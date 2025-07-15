import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from pathlib import Path
import gc
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QuantizationConfig:
    method: str = "sparsegpt"
    bits: int = 4
    group_size: int = 128
    sparsity: float = 0.5
    block_size: int = 128
    percdamp: float = 0.01
    nsamples: int = 128
    use_cuda_fp16: bool = True
    model_seqlen: int = 2048

class SparseGPTQuantizer:
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_calibration_data(self, tokenizer, dataset_name="wikitext", subset="wikitext-2-raw-v1"):
        from datasets import load_dataset
        dataset = load_dataset(dataset_name, subset, split="train")
        samples = []
        max_len = min(self.config.model_seqlen, tokenizer.model_max_length)
        for i in range(self.config.nsamples * 2):
            if len(samples) >= self.config.nsamples:
                break
            txt = dataset[i]["text"]
            if txt.strip():
                ids = tokenizer(txt, return_tensors="pt", truncation=True, max_length=max_len, padding="max_length").input_ids
                if ids.shape[1] > 10:
                    samples.append(ids.to(self.device))
        return samples

    def _prepare_model(self, model):
        if hasattr(model, "model"):
            return model.model.layers
        if hasattr(model, "transformer"):
            if hasattr(model.transformer, "h"):
                return model.transformer.h
            if hasattr(model.transformer, "blocks"):
                return model.transformer.blocks
        raise RuntimeError("unsupported model architecture")

    def _find_linear(self, module):
        res = {}
        for n, m in module.named_modules():
            if isinstance(m, nn.Linear):
                res[n] = m
        return res

    def _capture_activations(self, model, layers, samples):
        acts = {i: [] for i in range(len(layers))}
        hooks = []
        for idx, l in enumerate(layers):
            def save_inp(i):
                def hook(mod, inp, out):
                    acts[i].append(inp[0].detach())
                return hook
            hooks.append(l.register_forward_hook(save_inp(idx)))
        for x in samples:
            with torch.no_grad():
                _ = model(x)
        for h in hooks:
            h.remove()
        for k in acts:
            acts[k] = torch.cat(acts[k], dim=0)
        return acts

    def _apply_sparsegpt(self, model, layers, samples):
        activations = self._capture_activations(model, layers, samples)
        for lid, layer in enumerate(layers):
            linear_layers = self._find_linear(layer)
            for _, lin in linear_layers.items():
                W = lin.weight.data
                inps = activations[lid]
                H = (inps.T @ inps) / inps.shape[0]
                damp = self.config.percdamp * torch.mean(torch.diag(H))
                H += torch.eye(H.shape[0], device=H.device) * damp
                Hinv = torch.linalg.inv(H)
                score = torch.sum((W @ Hinv) * W, dim=1)
                k = int(score.numel() * (1 - self.config.sparsity))
                _, keep_idx = torch.topk(score, k)
                mask = torch.zeros_like(score, dtype=torch.bool)
                mask[keep_idx] = True
                W[~mask] = 0
                self._quantize_weights(lin)

    def _quantize_weights(self, layer):
        g = self.config.group_size
        b = self.config.bits
        w = layer.weight.data
        out, inp = w.shape
        pad = (g - (inp % g)) % g
        if pad:
            w = torch.cat([w, torch.zeros((out, pad), device=w.device, dtype=w.dtype)], dim=1)
        w = w.view(out, -1, g)
        mn = w.min(dim=2, keepdim=True)[0]
        mx = w.max(dim=2, keepdim=True)[0]
        scale = (mx - mn) / (2 ** b - 1)
        scale[scale == 0] = 1e-4
        zp = (-mn / scale).round()
        qw = ((w - mn) / scale + zp).round().clamp(0, 2 ** b - 1)
        dq = (qw - zp) * scale + mn
        dq = dq.view(out, -1)[:, :inp]
        layer.weight.data = dq

    def quantize_model(self, model_path: str, output_path: str):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.config.use_cuda_fp16 else torch.float32
        ).to(self.device)
        max_pos = getattr(model.config, "n_positions", getattr(model.config, "max_position_embeddings", 1024))
        self.config.model_seqlen = min(self.config.model_seqlen, max_pos)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        samples = self._get_calibration_data(tokenizer)
        layers = self._prepare_model(model)
        model.eval()
        with torch.no_grad():
            self._apply_sparsegpt(model, layers, samples)
        Path(output_path).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        with open(Path(output_path) / "quantization_config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

