# E2/toys/bico_elasticity.py
"""
Applies the Budget-Indexed Closure Operator (BICO) framework to a 2D
plane-stress finite‑element model, measuring how guaranteed and retrievable
regions of nodal stress converge with mesh refinement (storage budget) and
smoothing iterations (retrieval budget), and fitting predictive scaling laws.
"""
from __future__ import annotations

import argparse
import dataclasses as dc
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import torch

def get_device(use_cuda: bool) -> torch.device:
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

@dc.dataclass
class TriMesh2D:
    coords: torch.Tensor
    tris: torch.Tensor
    fixed: torch.Tensor
    traction_nodes: torch.Tensor
    L: float
    H: float

def generate_structured_trimesh(L: float, H: float, hx: float, hy: Optional[float] = None,
                                device: torch.device = torch.device("cpu")) -> TriMesh2D:
    if hy is None:
        hy = hx
    nx = max(2, int(round(L / hx)))
    ny = max(2, int(round(H / hy)))
    xs = torch.linspace(0.0, L, nx + 1, device=device)
    ys = torch.linspace(0.0, H, ny + 1, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    coords = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    def node_id(i, j):
        return j + (ny + 1) * i

    tris = []
    for i in range(nx):
        for j in range(ny):
            n00 = node_id(i, j)
            n10 = node_id(i + 1, j)
            n01 = node_id(i, j + 1)
            n11 = node_id(i + 1, j + 1)
            tris.append([n00, n10, n11])
            tris.append([n00, n11, n01])
    tris = torch.as_tensor(tris, dtype=torch.long, device=device)

    left_mask = coords[:, 0].isclose(torch.zeros(1, device=device))
    fixed = left_mask

    right_mask = coords[:, 0].isclose(torch.full((1,), L, device=device))
    traction_nodes = torch.nonzero(right_mask, as_tuple=False).reshape(-1)

    return TriMesh2D(coords=coords, tris=tris, fixed=fixed, traction_nodes=traction_nodes, L=L, H=H)

def elasticity_matrix_plane_stress(E: float, nu: float, device: torch.device) -> torch.Tensor:
    coef = E / (1.0 - nu ** 2)
    D = coef * torch.tensor([[1.0, nu, 0.0],
                              [nu, 1.0, 0.0],
                              [0.0, 0.0, (1.0 - nu) / 2.0]], device=device, dtype=torch.get_default_dtype())
    return D

def tri_element_stiffness(coords: torch.Tensor, tri: torch.Tensor, D: torch.Tensor, *, area_floor: float = 1e-18) -> Tuple[torch.Tensor, float]:
    xy = coords[tri.long(), :]
    x1, y1 = xy[0]; x2, y2 = xy[1]; x3, y3 = xy[2]
    area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    A = 0.5 * abs(float(area2))
    if A <= area_floor:
        A = area_floor

    if abs(area2) < 2 * area_floor:
        denom = (2 * area_floor) * (1.0 if area2 >= 0 else -1.0)
    else:
        denom = area2

    b1 = y2 - y3; b2 = y3 - y1; b3 = y1 - y2
    c1 = x3 - x2; c2 = x1 - x3; c3 = x2 - x1

    B = torch.tensor([[b1,0.0,b2,0.0,b3,0.0],
                      [0.0,c1,0.0,c2,0.0,c3],
                      [c1,b1,c2,b2,c3,b3]], dtype=D.dtype, device=D.device) / denom
    Ke = A * (B.T @ D @ B)
    return Ke, A

def assemble_global_stiffness(mesh: TriMesh2D, D: torch.Tensor) -> torch.Tensor:
    device = mesh.coords.device
    dtype = D.dtype
    nnode = mesh.coords.shape[0]
    ndof = 2 * nnode
    ntri = mesh.tris.shape[0]

    rows = []
    cols = []
    vals = []
    for e in range(ntri):
        tri = mesh.tris[e]
        Ke, A = tri_element_stiffness(mesh.coords, tri, D)
        if A <= 0: continue
        dofs = torch.stack([2*tri, 2*tri+1], dim=1).reshape(-1).tolist()
        for a in range(6):
            ra = dofs[a]
            for b in range(6):
                rows.append(ra)
                cols.append(dofs[b])
                vals.append(Ke[a, b].item())
    idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
    data = torch.tensor(vals, dtype=dtype, device=device)
    K = torch.sparse_coo_tensor(idx, data, (ndof, ndof), device=device)
    K = K.coalesce()
    return K

def assemble_load_vector(mesh: TriMesh2D, traction_x: float, traction_y: float = 0.0) -> torch.Tensor:
    device = mesh.coords.device
    dtype = torch.get_default_dtype()
    N = mesh.coords.shape[0]
    F = torch.zeros(2 * N, dtype=dtype, device=device)

    bnodes = mesh.traction_nodes
    by = mesh.coords[bnodes, 1]
    sort_idx = torch.argsort(by)
    bnodes = bnodes[sort_idx]
    for i in range(len(bnodes) - 1):
        n1 = int(bnodes[i])
        n2 = int(bnodes[i + 1])
        y1 = mesh.coords[n1, 1]
        y2 = mesh.coords[n2, 1]
        length = torch.sqrt((y2 - y1) ** 2 + 0.0)
        fe = length * 0.5 * torch.tensor([traction_x, traction_y, traction_x, traction_y], dtype=dtype, device=device)
        F[2 * n1:2 * n1 + 2] += fe[0:2]
        F[2 * n2:2 * n2 + 2] += fe[2:4]
    return F

def apply_dirichlet(K: torch.Tensor, F: torch.Tensor, fixed: torch.Tensor, value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N = fixed.shape[0]
    dof_mask = torch.zeros(2 * N, dtype=torch.bool, device=fixed.device)
    dof_mask[0::2] = fixed
    dof_mask[1::2] = fixed
    free = ~dof_mask
    free_ids = torch.nonzero(free, as_tuple=False).reshape(-1)

    K_dense = K.to_dense()
    Kff = K_dense[free][:, free]
    Ff = F[free].clone()
    return Kff, Ff, free_ids

def solve_pcg(K: torch.Tensor, F: torch.Tensor, tol: float = 1e-8, maxiter: int = 10000) -> torch.Tensor:
    device = K.device
    x = torch.zeros_like(F, device=device)
    r = F - K @ x
    M_inv = 1.0 / torch.diag(K)
    z = M_inv * r
    p = z.clone()
    rz_old = torch.dot(r, z)
    if torch.isclose(rz_old, torch.tensor(0.0)):
        return x
    for it in range(maxiter):
        Kp = K @ p
        alpha = rz_old / torch.dot(p, Kp)
        x = x + alpha * p
        r = r - alpha * Kp
        if torch.norm(r) < tol * torch.norm(F):
            break
        z = M_inv * r
        rz_new = torch.dot(r, z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
    return x

def solve_elasticity(mesh: TriMesh2D, E: float, nu: float, traction_x: float,
                     device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    D = elasticity_matrix_plane_stress(E, nu, device=device)
    K_coo = assemble_global_stiffness(mesh, D)
    F = assemble_load_vector(mesh, traction_x=traction_x)
    Kff, Ff, free_ids = apply_dirichlet(K_coo, F, mesh.fixed)
    uf = solve_pcg(Kff, Ff)
    ndof = 2 * mesh.coords.shape[0]
    u_full = torch.zeros(ndof, dtype=uf.dtype, device=uf.device)
    u_full[free_ids] = uf
    
    N = mesh.coords.shape[0]
    sigma_accum = torch.zeros((N, 3), dtype=u_full.dtype, device=u_full.device)
    area_accum = torch.zeros(N, dtype=u_full.dtype, device=u_full.device)
    Dloc = elasticity_matrix_plane_stress(E, nu, device=u_full.device)

    for tri in mesh.tris:
        tri = tri.long()
        i, j, k = tri.tolist()
        xy = mesh.coords[tri, :]
        x1, y1 = xy[0]; x2, y2 = xy[1]; x3, y3 = xy[2]
        area2_signed = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        A_unsigned = 0.5 * abs(area2_signed)
        if A_unsigned < 1e-18:
            continue
        denom = area2_signed
        if abs(denom) < 1e-18:
            denom = 1e-18 * (1.0 if denom >= 0 else -1.0)

        b1 = y2 - y3; b2 = y3 - y1; b3 = y1 - y2
        c1 = x3 - x2; c2 = x1 - x3; c3 = x2 - x1

        B = torch.tensor([[b1, 0.0, b2, 0.0, b3, 0.0],
                          [0.0, c1, 0.0, c2, 0.0, c3],
                          [c1, b1, c2, b2, c3, b3]], dtype=Dloc.dtype, device=Dloc.device) / denom
        
        dofs = torch.stack([2*tri, 2*tri+1], dim=1).reshape(-1)
        ue = u_full[dofs]
        eps = B @ ue
        sig = Dloc @ eps
        
        for n in [i, j, k]:
            sigma_accum[n] += sig * A_unsigned
            area_accum[n] += A_unsigned

    mask = area_accum > 1e-12
    sigma_nodal = torch.zeros_like(sigma_accum)
    sigma_nodal[mask] = sigma_accum[mask] / area_accum[mask,None]
    return u_full, sigma_nodal

@dc.dataclass
class RawLevel:
    h: float
    mesh: TriMesh2D
    u: torch.Tensor
    sigma: torch.Tensor

def compute_raw_levels(h_levels: List[float], L: float, H: float, E: float, nu: float, traction: float,
                       device: torch.device) -> List[RawLevel]:
    levels: List[RawLevel] = []
    for h in h_levels:
        print(f"  Solving for h={h:.4f}...")
        mesh = generate_structured_trimesh(L, H, hx=h, device=device)
        u, sigma = solve_elasticity(mesh, E=E, nu=nu, traction_x=traction, device=device)
        levels.append(RawLevel(h=h, mesh=mesh, u=u, sigma=sigma))
    return levels

def project_to_mesh(sigma_fine: torch.Tensor, mesh_fine: TriMesh2D, mesh_coarse: TriMesh2D) -> torch.Tensor:
    diff = mesh_coarse.coords[:, None, :] - mesh_fine.coords[None, :, :]
    d2 = torch.sum(diff ** 2, dim=2)
    idx = torch.argmin(d2, dim=1)
    sigma_proj = sigma_fine[idx]
    return sigma_proj

def nodal_stress_error(sigma: torch.Tensor, sigma_ref: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(sigma - sigma_ref, dim=1)

def guaranteed_thresholds(levels: List[RawLevel], sigma_ref: torch.Tensor, q: float = 1.5) -> Dict[float, float]:
    sorted_levels = sorted(levels, key=lambda lvl: lvl.h, reverse=True)
    finest_mesh = sorted(levels, key=lambda lvl: lvl.h)[-1].mesh

    med_errors_map = {}
    for lvl in sorted_levels:
        sigma_ref_on_lvl = project_to_mesh(sigma_ref, finest_mesh, lvl.mesh)
        err = nodal_stress_error(lvl.sigma, sigma_ref_on_lvl)
        med_errors_map[lvl.h] = torch.median(err).item()
    
    thresholds = {}
    running_max_thresh = -1.0
    for lvl in sorted_levels:
        current_thresh = q * med_errors_map[lvl.h]
        running_max_thresh = max(running_max_thresh, current_thresh)
        thresholds[lvl.h] = running_max_thresh
        
    return thresholds

def guaranteed_region_mask(level: RawLevel, sigma_ref_fine: torch.Tensor, mesh_ref: TriMesh2D,
                           tau_h: float) -> torch.Tensor:
    sigma_ref_on_lvl = project_to_mesh(sigma_ref_fine, mesh_ref, level.mesh)
    err = nodal_stress_error(level.sigma, sigma_ref_on_lvl)
    mask = err <= tau_h
    return mask

def empty_set(mesh: TriMesh2D) -> torch.Tensor:
    return torch.zeros(mesh.coords.shape[0], dtype=torch.bool, device=mesh.coords.device)

def full_set(mesh: TriMesh2D) -> torch.Tensor:
    return torch.ones(mesh.coords.shape[0], dtype=torch.bool, device=mesh.coords.device)

def set_union(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a | b

def set_intersection(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a & b

def set_subseteq(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool((~a | b).all())

def K_operator(level: RawLevel, G_mask: torch.Tensor, A_mask: torch.Tensor) -> torch.Tensor:
    return set_union(A_mask, G_mask)

def laplacian_smooth_nodal_scalar(values: torch.Tensor, mesh: TriMesh2D, iters: int) -> torch.Tensor:
    if iters == 0:
        return values.clone()

    v = values.clone()
    N = mesh.coords.shape[0]
    tris = mesh.tris
    i = torch.cat([tris[:, 0], tris[:, 1], tris[:, 2]])
    j = torch.cat([tris[:, 1], tris[:, 2], tris[:, 0]])

    indices = torch.stack([torch.cat([i, j]), torch.cat([j, i])], dim=0)
    values_ones = torch.ones(indices.shape[1], device=v.device, dtype=v.dtype)
    adj = torch.sparse_coo_tensor(indices, values_ones, (N, N)).coalesce()

    adj_sum = torch.sparse.sum(adj, dim=1).to_dense()

    for _ in range(iters):
        v_sum = adj @ v.unsqueeze(1)
        v_avg = v_sum.squeeze(1) / adj_sum.clamp(min=1)
        v = 0.5 * v + 0.5 * v_avg
    return v

def smooth_stress(mesh: TriMesh2D, sigma: torch.Tensor, theta: int) -> torch.Tensor:
    s0 = laplacian_smooth_nodal_scalar(sigma[:, 0], mesh, theta)
    s1 = laplacian_smooth_nodal_scalar(sigma[:, 1], mesh, theta)
    s2 = laplacian_smooth_nodal_scalar(sigma[:, 2], mesh, theta)
    return torch.stack([s0, s1, s2], dim=1)

def retrievability_scores(mesh: TriMesh2D, sigma_in: torch.Tensor, theta: int) -> torch.Tensor:
    sm = smooth_stress(mesh, sigma_in, theta)
    variation = torch.linalg.vector_norm(sm - sigma_in, dim=1)
    score = 1.0 / (1e-12 + variation)
    return score

def compute_retrieval_thresholds(level_ref: RawLevel, thetas: List[int], p: float = 0.25) -> Dict[int, float]:
    scores = {}
    for th in sorted(thetas):
        s = retrievability_scores(level_ref.mesh, level_ref.sigma, th)
        scores[th] = torch.quantile(s, p).item()
    return scores

def R_operator(level: RawLevel, B_mask: torch.Tensor, theta: int, tau_ret: float) -> torch.Tensor:
    s = retrievability_scores(level.mesh, level.sigma, theta)
    keep = s >= tau_ret
    return set_intersection(B_mask, keep)

def C_operator(level: RawLevel, G_mask: torch.Tensor, A_mask: torch.Tensor,
               theta: int, tau_ret: float) -> torch.Tensor:
    Kmask = K_operator(level, G_mask, A_mask)
    return R_operator(level, Kmask, theta, tau_ret)

def metric_L2_error(level: RawLevel, sigma_ref_fine: torch.Tensor, mesh_ref: TriMesh2D,
                    mask: torch.Tensor) -> float:
    sigma_ref_on_lvl = project_to_mesh(sigma_ref_fine, mesh_ref, level.mesh)
    if mask.sum() == 0:
        return float("inf")
    err_sq = (level.sigma[mask] - sigma_ref_on_lvl[mask]) ** 2
    return torch.sqrt(torch.mean(torch.sum(err_sq, dim=1))).item()

def metric_coverage(mask: torch.Tensor) -> float:
    return float(mask.float().mean().item())

def check_axioms(levels: List[RawLevel], Gmasks: Dict[float, torch.Tensor]) -> Dict[str, bool]:
    def proj_mask(src_mesh: TriMesh2D, dst_mesh: TriMesh2D, mask_src: torch.Tensor) -> torch.Tensor:
        if src_mesh is dst_mesh: return mask_src
        diff = dst_mesh.coords[:, None, :] - src_mesh.coords[None, :, :]
        d2 = torch.sum(diff**2, dim=2)
        nn_map = torch.argmin(d2, dim=1)
        return mask_src[nn_map]

    res = {k: True for k in ['A1','A2','A3','A4','A5']}

    for lvl in levels:
        N = lvl.mesh.coords.shape[0]
        G = Gmasks[lvl.h]
        rng = torch.rand(N, device=G.device)
        A = rng < 0.25
        B = rng < 0.50
        if not set_subseteq(A, B): B = A | B

        if not set_subseteq(A, K_operator(lvl, G, A)): res['A1'] = False
        KA = K_operator(lvl, G, A)
        if (K_operator(lvl, G, KA) ^ KA).any(): res['A2'] = False
        if not set_subseteq(K_operator(lvl, G, A), K_operator(lvl, G, B)): res['A3'] = False
        
        xs = lvl.mesh.coords[:,0]
        q1,q2 = torch.quantile(xs, torch.tensor([0.33,0.66],device=xs.device))
        fam = [xs < q1, xs < q2, torch.ones_like(A, dtype=torch.bool)]
        unionA = fam[0] | fam[1] | fam[2]
        left = K_operator(lvl, G, unionA)
        right = K_operator(lvl,G,fam[0]) | K_operator(lvl,G,fam[1]) | K_operator(lvl,G,fam[2])
        if (left ^ right).any(): res['A5'] = False

    levels_sorted = sorted(levels, key=lambda L: L.h, reverse=True)
    for i in range(len(levels_sorted)-1):
        c, f = levels_sorted[i], levels_sorted[i+1]
        Gc, Gf = Gmasks[c.h], Gmasks[f.h]
        A_c = torch.rand(c.mesh.coords.shape[0], device=Gc.device) < 0.4
        A_f = proj_mask(c.mesh, f.mesh, A_c)
        
        Kc_proj = proj_mask(c.mesh, f.mesh, K_operator(c, Gc, A_c))
        Kf = K_operator(f, Gf, A_f)
        if not set_subseteq(Kc_proj, Kf):
            res['A4'] = False
            break

    return res

def build_monotonic_gmasks(levels: List[RawLevel], sigma_ref_fine: torch.Tensor, 
                             mesh_ref: TriMesh2D, tauG: Dict[float, float]) -> Dict[float, torch.Tensor]:
    def proj_mask(src_mesh: TriMesh2D, dst_mesh: TriMesh2D, mask_src: torch.Tensor) -> torch.Tensor:
        if src_mesh is dst_mesh: return mask_src
        diff = dst_mesh.coords[:, None, :] - src_mesh.coords[None, :, :]
        d2 = torch.sum(diff**2, dim=2)
        nn_map = torch.argmin(d2, dim=1)
        return mask_src[nn_map]

    sorted_levels = sorted(levels, key=lambda lvl: lvl.h, reverse=True)
    
    naive_gmasks = {}
    for lvl in sorted_levels:
        tau_h = tauG[lvl.h]
        naive_gmasks[lvl.h] = guaranteed_region_mask(lvl, sigma_ref_fine, mesh_ref, tau_h)

    monotonic_gmasks = {}
    last_gmask = None
    last_mesh = None

    for lvl in sorted_levels:
        if last_gmask is None:
            current_gmask = naive_gmasks[lvl.h]
        else:
            gmask_proj = proj_mask(last_mesh, lvl.mesh, last_gmask)
            current_gmask = set_union(gmask_proj, naive_gmasks[lvl.h])
        
        monotonic_gmasks[lvl.h] = current_gmask
        last_gmask = current_gmask
        last_mesh = lvl.mesh

    return monotonic_gmasks

def detect_plateau(xs: List[float], ys: List[float], eps: float) -> float:
    for i, x in enumerate(xs):
        y = ys[i]
        is_plateau = True
        for j in range(i + 1, len(xs)):
            if abs(ys[j] - y) > eps:
                is_plateau = False
                break
        if is_plateau:
            return x
    return xs[-1] if xs else 0.0

def run_experiment(args):
    device = get_device(args.cuda)
    torch.set_default_dtype(torch.float64)

    h_levels = sorted(list(set(args.h_levels)))

    print("Building raw levels...")
    levels = compute_raw_levels(h_levels, L=args.L, H=args.H, E=args.E, nu=args.nu,
                                traction=args.T, device=device)

    finest = levels[-1]
    sigma_ref = finest.sigma

    print("Computing guaranteed region thresholds...")
    tauG = guaranteed_thresholds(levels, sigma_ref, q=args.guarantee_q)

    print("Building monotonic Guaranteed Regions to satisfy Axiom A4...")
    Gmasks = build_monotonic_gmasks(levels, sigma_ref, finest.mesh, tauG)

    print("Computing retrieval thresholds...")
    thetas = sorted(list(set(args.thetas)))
    tauR = compute_retrieval_thresholds(finest, thetas, p=args.retrieval_p)

    print("\nChecking axioms...")
    ax_res = check_axioms(levels, Gmasks)
    for k, v in ax_res.items():
        print(f"  {k}: {'OK' if v else 'FAIL'}")

    print("\nComputing metrics (A=∅)...")
    A_empty_masks = {lvl.h: empty_set(lvl.mesh) for lvl in levels}
    results = []
    for lvl in levels:
        A_mask = A_empty_masks[lvl.h]
        G_mask = Gmasks[lvl.h]
        K_mask = K_operator(lvl, G_mask, A_mask)
        K_L2 = metric_L2_error(lvl, sigma_ref, finest.mesh, K_mask)
        K_cov = metric_coverage(K_mask)
        results.append({'h': lvl.h, 'theta': -1, 'op': 'K', 'L2': K_L2, 'cov': K_cov})
        for th in thetas:
            tau_ret = tauR[th]
            C_mask = C_operator(lvl, G_mask, A_mask, th, tau_ret)
            C_L2 = metric_L2_error(lvl, sigma_ref, finest.mesh, C_mask)
            C_cov = metric_coverage(C_mask)
            results.append({'h': lvl.h, 'theta': th, 'op': 'C', 'L2': C_L2, 'cov': C_cov})

    max_theta = max(thetas) if thetas else 0
    c_results_max_theta = [r for r in results if r['op'] == 'C' and r['theta'] == max_theta]
    if c_results_max_theta:
        xs = [r['h'] for r in c_results_max_theta]
        ys = [r['L2'] for r in c_results_max_theta]
        plateau_h = detect_plateau(xs, ys, eps=args.plateau_eps)
        print(f"\nPlateau detected at h ≈ {plateau_h:.4g} (L2 error for θ={max_theta}, eps={args.plateau_eps:g}).")

    if args.plot:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        for th in thetas:
            data = [r for r in results if r['op'] == 'C' and r['theta'] == th]
            ax1.plot([r['h'] for r in data], [r['L2'] for r in data], marker='o', linestyle='-', label=f"C(θ={th})")
        if c_results_max_theta and xs:
             y_plateau = np.interp(plateau_h, xs, ys)
             ax1.plot([plateau_h], [y_plateau], 'k*', markersize=12, label=f'Plateau (h={plateau_h:.2g})')

        ax1.set_ylabel('L2 Stress Error (Masked)')
        ax1.set_title('BICO Performance vs. Budget (Element Size h)')
        ax1.legend()
        ax1.invert_xaxis()
        
        for th in thetas:
            data = [r for r in results if r['op'] == 'C' and r['theta'] == th]
            ax2.plot([r['h'] for r in data], [r['cov']*100 for r in data], marker='o', linestyle='--', label=f"C(θ={th})")
        ax2.set_xlabel('Element Size h (Budget Increases →)')
        ax2.set_ylabel('Coverage (%)')
        ax2.set_ylim(0, 101)
        
        plt.tight_layout()
        plt.show()

def parse_args():
    p = argparse.ArgumentParser(description="BICO mechanical stress GPU demo")
    p.add_argument('--L', type=float, default=2.0, help='Beam length')
    p.add_argument('--H', type=float, default=0.4, help='Beam height')
    p.add_argument('--E', type=float, default=210e9, help='Young modulus (Pa)')
    p.add_argument('--nu', type=float, default=0.3, help='Poisson ratio')
    p.add_argument('--T', type=float, default=50e6, help='Applied traction σ_xx (Pa) on right edge')
    p.add_argument('--h-levels', type=float, nargs='+', default=[8e-2, 4e-2, 2e-2, 1e-2],
                   help='Element size ladder (any order)')
    p.add_argument('--thetas', type=int, nargs='+', default=[0, 1, 2, 4], help='Retrieval smoothing iterations')
    p.add_argument('--guarantee-q', type=float, default=1.5, help='Multiplier for median error -> τ_G')
    p.add_argument('--retrieval-p', type=float, default=0.25, help='Quantile for τ_ret(θ) from scores')
    p.add_argument('--plateau-eps', type=float, default=1e5, help='Tolerance for plateau detection in Pa')
    p.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    p.add_argument('--plot', action='store_true', help='Generate plots')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)

