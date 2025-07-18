# E2/toys/bico_wave.py
"""
Implements BICO-based predictive scaling experiments on a damped 2D wave equation,
measuring how retrievable coverage scales with simulation time (storage budget)
and retrieval threshold, and fitting a two‑stage power‑law performance model.
"""

from __future__ import annotations
import argparse
from typing import Dict, List, Tuple
import math
import numpy as np
import torch
import torch.nn.functional as F


def get_device(use_cuda: bool) -> torch.device:
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_initial_pluck(
    width: int,
    height: int,
    device: torch.device,
    px: float = 0.2,
    py: float = 0.3,
    sigma: float = 0.05,
    dtype=torch.float64,
) -> torch.Tensor:
    x = torch.linspace(-1, 1, width, device=device, dtype=dtype)
    y = torch.linspace(-1, 1, height, device=device, dtype=dtype)
    xv, yv = torch.meshgrid(x, y, indexing="xy")
    dist_sq = (xv - px) ** 2 + (yv - py) ** 2
    return torch.exp(-dist_sq / (2 * sigma**2))


def propagate_wave(
    initial_grid: torch.Tensor,
    total_steps: int,
    record_h: List[int],
    c: float = 1.0,
    damping: float = 0.001,
) -> Dict[int, torch.Tensor]:
    dt = 0.001
    dx = 2.0 / initial_grid.shape[1]
    if c * dt / dx > 0.7:
        print(f"Warning: Courant number is {(c * dt / dx):.2f} > 0.7.")
    k = torch.tensor(
        [[0.25, 0.5, 0.25], [0.5, -3.0, 0.5], [0.25, 0.5, 0.25]],
        dtype=initial_grid.dtype,
        device=initial_grid.device,
    ).reshape(1, 1, 3, 3)
    u_prev = initial_grid.clone()
    u_curr = initial_grid.clone()
    rec_set = set(record_h)
    rec: Dict[int, torch.Tensor] = {}
    rec[0] = u_curr.clone()
    for step in range(1, total_steps + 1):
        u_curr_4d = u_curr.unsqueeze(0).unsqueeze(0)
        lap = F.conv2d(u_curr_4d, k, padding="same").squeeze(0).squeeze(0)
        u_next = (2.0 * u_curr - u_prev + (c * dt) ** 2 * lap / dx**2) * (1.0 - damping)
        u_prev, u_curr = u_curr, u_next
        if step in rec_set:
            rec[step] = u_curr.clone()
    if total_steps not in rec:
        rec[total_steps] = u_curr.clone()
    return rec


def calculate_stability_mask(
    history: List[torch.Tensor], threshold: float
) -> torch.Tensor:
    h = torch.stack(history, dim=0)
    std_dev = torch.std(h, dim=0)
    return std_dev < threshold


def raw_gmasks_from_history(
    states_by_t: Dict[int, torch.Tensor],
    h_levels: List[int],
    stability_window: int,
    stability_threshold: float,
) -> Dict[int, torch.Tensor]:
    ts = sorted(states_by_t)
    t_to_idx = {t: i for i, t in enumerate(ts)}
    all_states = [states_by_t[t] for t in ts]
    gmasks: Dict[int, torch.Tensor] = {}
    for h in h_levels:
        if h not in t_to_idx:
            idx = max(i for i, t in enumerate(ts) if t <= h)
        else:
            idx = t_to_idx[h]
        start = max(0, idx - stability_window + 1)
        hist = all_states[start : idx + 1]
        gmasks[h] = calculate_stability_mask(hist, stability_threshold)
    return gmasks


def build_monotonic_gmasks(
    budgets: List[int], stability_masks: Dict[int, torch.Tensor]
) -> Dict[int, torch.Tensor]:
    m: Dict[int, torch.Tensor] = {}
    last = None
    for h in sorted(budgets):
        cur = stability_masks[h]
        if last is None:
            m[h] = cur
        else:
            m[h] = last | cur
        last = m[h]
    return m


def K_factory(gmasks: Dict[int, torch.Tensor]):
    def K(h: int, A: torch.Tensor) -> torch.Tensor:
        return A | gmasks[h]

    return K


def R_factory():
    def tau(theta: float) -> float:
        t = max(1e-9, min(1.0, theta))
        return 1.0 - t

    def R(theta: float, B: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        s_min = torch.min(score)
        s_max = torch.max(score)
        if (s_max - s_min) <= 0:
            mask_score = torch.ones_like(B, dtype=torch.bool)
        else:
            norm = (score - s_min) / (s_max - s_min)
            thr = tau(theta)
            mask_score = norm >= thr
        return B & mask_score

    return R


def pluck_mask(initial_grid: torch.Tensor, alpha: float) -> torch.Tensor:
    mx = initial_grid.max()
    thr = alpha * mx
    return initial_grid >= thr


def recall(pred: torch.Tensor, gold: torch.Tensor) -> float:
    denom = gold.sum().item()
    if denom == 0:
        return float("nan")
    return (pred & gold).sum().item() / denom


def precision(pred: torch.Tensor, gold: torch.Tensor) -> float:
    denom = pred.sum().item()
    if denom == 0:
        return float("nan")
    return (pred & gold).sum().item() / denom


def f1(pred: torch.Tensor, gold: torch.Tensor) -> float:
    r = recall(pred, gold)
    p = precision(pred, gold)
    if r == 0 or p == 0 or math.isnan(r) or math.isnan(p):
        return 0.0
    return 2 * p * r / (p + r)


def fit_1d_scaling(hs: np.ndarray, Ms: np.ndarray):
    mask = ~np.isnan(Ms)
    hs = hs[mask]
    Ms = Ms[mask]
    if len(hs) < 3:
        return float("nan"), float("nan"), float("nan")
    x0 = np.array([Ms.max(), 0.5, 0.5], dtype=np.float64)

    def model(params, h):
        Mmax, a, gamma = params
        return np.clip(Mmax * (1.0 - a * np.power(h, -gamma)), 0.0, 1.0)

    def loss(params):
        pred = model(params, hs)
        return np.mean((pred - Ms) ** 2)

    params = x0
    lr = 1e-2
    for _ in range(10000):
        eps = 1e-6
        g = np.zeros_like(params)
        for i in range(len(params)):
            d = np.zeros_like(params)
            d[i] = eps
            g[i] = (loss(params + d) - loss(params - d)) / (2 * eps)
        params = params - lr * g
        if np.linalg.norm(g) < 1e-8:
            break
    Mmax, a, gamma = params
    if not np.isfinite(Mmax) or Mmax <= 0:
        Mmax = Ms.max()
    if not np.isfinite(a) or a <= 0:
        a = 1e-6
    if not np.isfinite(gamma) or gamma <= 0:
        gamma = 1e-6
    if Mmax > 1:
        Mmax = 1.0
    return float(Mmax), float(a), float(gamma)


def fit_2d_scaling(Ns: np.ndarray, thetas: np.ndarray, Qs: np.ndarray):
    mask = ~np.isnan(Qs)
    Ns = Ns[mask]
    thetas = thetas[mask]
    Qs = Qs[mask]
    if len(Ns) < 5:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    Qmax0 = np.nanmax(Qs)
    if not np.isfinite(Qmax0) or Qmax0 <= 0:
        Qmax0 = 1.0
    params = np.array([Qmax0, 0.5, 0.5, 0.5, 0.5], dtype=np.float64)

    def model(p, N, th):
        Qm, a, b, gam, del_ = p
        return np.clip(
            Qm * (1.0 - a * np.power(N, -gam)) * (1.0 - b * np.power(th, -del_)),
            0.0,
            1.0,
        )

    def loss(p):
        return np.mean((model(p, Ns, thetas) - Qs) ** 2)

    lr = 1e-3
    for _ in range(20000):
        eps = 1e-6
        g = np.zeros_like(params)
        for i in range(len(params)):
            d = np.zeros_like(params)
            d[i] = eps
            g[i] = (loss(params + d) - loss(params - d)) / (2 * eps)
        params = params - lr * g
        if np.linalg.norm(g) < 1e-7:
            break
    Qm, a, b, gam, del_ = params
    if not np.isfinite(Qm) or Qm <= 0:
        Qm = Qmax0
    if not np.isfinite(a) or a <= 0:
        a = 1e-6
    if not np.isfinite(b) or b <= 0:
        b = 1e-6
    if not np.isfinite(gam) or gam <= 0:
        gam = 1e-6
    if not np.isfinite(del_) or del_ <= 0:
        del_ = 1e-6
    if Qm > 1:
        Qm = 1.0
    return float(Qm), float(a), float(b), float(gam), float(del_)


def predicted_plateau_h(Mmax: float, a: float, gamma: float, eps: float) -> float:
    if not math.isfinite(Mmax) or Mmax <= 0 or a <= 0 or gamma <= 0:
        return float("inf")
    gap = eps / Mmax
    if gap <= 0:
        return float("inf")
    return (a / gap) ** (1.0 / gamma)


def run_single_resolution(
    device,
    width,
    height,
    sim_steps,
    h_levels,
    stability_window,
    stability_threshold,
    alpha_pluck,
    theta_grid,
    damping,
    eps_rel: float,
):
    init = create_initial_pluck(width, height, device=device)
    rec_ts = sorted(set(h_levels + [sim_steps]))
    states = propagate_wave(init, sim_steps, rec_ts, damping=damping)
    raw_g = raw_gmasks_from_history(
        states, h_levels, stability_window, stability_threshold
    )
    canon_g = build_monotonic_gmasks(h_levels, raw_g)
    K = K_factory(canon_g)
    R = R_factory()
    gold = pluck_mask(init, alpha_pluck)
    empty = torch.zeros_like(init, dtype=torch.bool)
    rows = []
    for h in h_levels:
        k_mask = K(h, empty)
        score = torch.abs(states[max(t for t in states if t <= h)])
        for theta in theta_grid:
            r_mask = R(theta, k_mask, score)
            q_rec = recall(r_mask, gold)
            q_f1 = f1(r_mask, gold)
            cs = k_mask.float().mean().item()
            cr = r_mask.float().mean().item()
            if cs <= eps_rel:
                covrel = float("nan")
            else:
                covrel = cr / cs
            rows.append((width * height, h, theta, q_rec, q_f1, cs, cr, covrel))
    return rows, canon_g


def aggregate_by_h(rows, use_col: int):
    d = {}
    for r in rows:
        N, h, theta, q_rec, q_f1, cs, cr, covrel = r
        if h not in d:
            d[h] = []
        if use_col == 0:
            d[h].append(q_rec)
        elif use_col == 1:
            d[h].append(cr)
        else:
            d[h].append(covrel)
    hs = sorted(d)
    Ms = []
    for h in hs:
        vals = [v for v in d[h] if not math.isnan(v)]
        if len(vals) == 0:
            Ms.append(np.nan)
        else:
            Ms.append(float(np.mean(vals)))
    return np.array(hs, dtype=np.float64), np.array(Ms, dtype=np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--heights", type=int, nargs="+", default=None)
    p.add_argument("--sim_steps", type=int, default=2000)
    p.add_argument(
        "--h_levels", type=int, nargs="+", default=[50, 200, 400, 800, 1600, 2000]
    )
    p.add_argument("--stability_window", type=int, default=50)
    p.add_argument("--stability_threshold", type=float, default=1e-4)
    p.add_argument("--alpha_pluck", type=float, default=0.5)
    p.add_argument(
        "--theta_grid", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 1.0]
    )
    p.add_argument("--damping", type=float, default=0.002)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--eps_plateau", type=float, default=0.01)
    p.add_argument("--eps_rel", type=float, default=1e-9)
    p.add_argument(
        "--metric", choices=["recall", "coverage", "relcov"], default="relcov"
    )
    args = p.parse_args()

    device = get_device(args.cuda)
    heights = args.heights if args.heights is not None else args.widths
    theta_grid = sorted(set(args.theta_grid))
    all_rows = []
    canon_cache = {}
    for w, hgt in zip(args.widths, heights):
        rows, canon_g = run_single_resolution(
            device,
            w,
            hgt,
            args.sim_steps,
            args.h_levels,
            args.stability_window,
            args.stability_threshold,
            args.alpha_pluck,
            theta_grid,
            args.damping,
            args.eps_rel,
        )
        all_rows.extend(rows)
        canon_cache[(w, hgt)] = canon_g

    Ns = np.array([r[0] for r in all_rows], dtype=np.float64)
    hs_arr = np.array([r[1] for r in all_rows], dtype=np.float64)
    th_arr = np.array([r[2] for r in all_rows], dtype=np.float64)
    qrec_arr = np.array([r[3] for r in all_rows], dtype=np.float64)
    covret_arr = np.array([r[6] for r in all_rows], dtype=np.float64)
    covrel_arr = np.array([r[7] for r in all_rows], dtype=np.float64)

    if args.metric == "recall":
        use_col = 0
        Q_input = qrec_arr
    elif args.metric == "coverage":
        use_col = 1
        Q_input = covret_arr
    else:
        use_col = 2
        Q_input = covrel_arr

    hfilt, Mfilt = aggregate_by_h(all_rows, use_col)
    Mmax, a1, g1 = fit_1d_scaling(hfilt, Mfilt)
    hp = predicted_plateau_h(Mmax, a1, g1, args.eps_plateau)

    qm, a2, b2, g2, d2 = fit_2d_scaling(Ns, th_arr, Q_input)

    print("rows_N_h_theta_qrec_qf1_covstore_covret_covrel")
    for r in all_rows:
        print(*r, sep=",")
    print("metric_used", args.metric)
    print("fit1d_Mmax,a,gamma,h_plateau", Mmax, a1, g1, hp)
    print("fit2d_Qmax,a,b,gamma,delta", qm, a2, b2, g2, d2)

    if args.plot:
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure(figsize=(8, 5))
        plt.scatter(hfilt, Mfilt, label="data")
        xs = np.linspace(min(hfilt), max(hfilt) * 1.2, 200)
        ys = np.clip(Mmax * (1.0 - a1 * xs ** (-g1)), 0.0, 1.0)
        plt.plot(xs, ys, label="fit")
        plt.axhline(Mmax, linestyle="--", color="gray")
        if math.isfinite(hp):
            plt.axvline(hp, linestyle="--", color="red", label="pred_plateau")
        plt.xlabel("h")
        plt.ylabel("M(h)")
        plt.legend()
        Ns_unique = np.unique(Ns)
        Th_unique = np.unique(th_arr)
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(np.log10(Ns), np.log10(th_arr), Q_input, c="C0", marker="o")
        NX, TX = np.meshgrid(
            np.logspace(np.log10(Ns_unique.min()), np.log10(Ns_unique.max()), 40),
            np.logspace(np.log10(Th_unique.min()), np.log10(Th_unique.max()), 40),
        )
        Q_pred = np.clip(
            qm * (1.0 - a2 * NX ** (-g2)) * (1.0 - b2 * TX ** (-d2)), 0.0, 1.0
        )
        ax.plot_surface(np.log10(NX), np.log10(TX), Q_pred, alpha=0.3)
        ax.set_xlabel("log10 N")
        ax.set_ylabel("log10 theta")
        ax.set_zlabel("Q")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
