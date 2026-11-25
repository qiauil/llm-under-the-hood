#!/usr/bin/env python3
"""E-2: CPU correctness checks for a tiny Decoder-only Transformer."""
import json
import math
import platform
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


CFG = dict(L=2, d=32, H=4, H_KV=2, d_h=8, d_ff=64, V=64, B=2, T=6)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope(x):
    # x is [B,H,T,d_h]; RoPE is applied only to Q/K.
    d_h = x.shape[-1]
    position = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
    inverse = 10000 ** (-torch.arange(0, d_h, 2, device=x.device, dtype=torch.float32) / d_h)
    angle = position[:, None] * inverse[None, :]
    cosine, sine = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, kv, dh = cfg["d"], cfg["H_KV"], cfg["d_h"]
        self.cfg = cfg
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, kv * dh, bias=False)
        self.v = nn.Linear(d, kv * dh, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, trace=None):
        b, t, _ = x.shape
        c = self.cfg
        q = self.q(x).view(b, t, c["H"], c["d_h"]).transpose(1, 2)
        k = self.k(x).view(b, t, c["H_KV"], c["d_h"]).transpose(1, 2)
        v = self.v(x).view(b, t, c["H_KV"], c["d_h"]).transpose(1, 2)
        q_rot, k_rot = rope(q), rope(k)
        groups = c["H"] // c["H_KV"]
        k_grouped = k_rot.repeat_interleave(groups, dim=1)
        v_grouped = v.repeat_interleave(groups, dim=1)
        score = q_rot @ k_grouped.transpose(-1, -2) / math.sqrt(c["d_h"])
        future = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
        probability = score.masked_fill(future, float("-inf")).softmax(-1)
        merged = (probability @ v_grouped).transpose(1, 2).contiguous().view(b, t, c["d"])
        output = self.o(merged)
        if trace is not None:
            trace.update({
                "q": list(q.shape), "k": list(k.shape), "v": list(v.shape),
                "q_rope": list(q_rot.shape), "k_rope": list(k_rot.shape),
                "score": list(score.shape), "probability": list(probability.shape),
                "attention_merged": list(merged.shape), "attention_output": list(output.shape),
                "max_future_probability": float(probability.masked_select(future[None, None]).max()),
            })
        return output


class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x, trace=None):
        gate, up = self.gate(x), self.up(x)
        product = F.silu(gate) * up
        output = self.down(product)
        if trace is not None:
            trace.update({"gate": list(gate.shape), "up": list(up.shape), "swiglu_product": list(product.shape), "mlp_output": list(output.shape)})
        return output


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg["d"]), RMSNorm(cfg["d"])
        self.attention = Attention(cfg)
        self.mlp = SwiGLU(cfg["d"], cfg["d_ff"])

    def forward(self, x, trace=None):
        attention_input = self.n1(x)
        after_attention = x + self.attention(attention_input, trace)
        mlp_input = self.n2(after_attention)
        output = after_attention + self.mlp(mlp_input, trace)
        if trace is not None:
            trace.update({"attention_input": list(attention_input.shape), "after_attention_residual": list(after_attention.shape), "mlp_input": list(mlp_input.shape), "block_output": list(output.shape)})
        return output


class MiniGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg["V"], cfg["d"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["L"])])
        self.final_norm = RMSNorm(cfg["d"])

    def forward(self, tokens, trace=None):
        x = self.embedding(tokens)
        if trace is not None:
            trace["tokens"] = list(tokens.shape)
            trace["embedding"] = list(x.shape)
        for index, block in enumerate(self.blocks):
            x = block(x, trace if index == 0 else None)
        normalized = self.final_norm(x)
        logits = F.linear(normalized, self.embedding.weight)
        if trace is not None:
            trace["final_norm"] = list(normalized.shape)
            trace["logits"] = list(logits.shape)
        return logits


def main():
    torch.manual_seed(20261102)
    model = MiniGPT(CFG).eval()
    tokens = torch.randint(0, CFG["V"], (CFG["B"], CFG["T"]))
    trace = {}
    with torch.inference_mode():
        logits = model(tokens, trace)
        changed = tokens.clone()
        changed[:, 3:] = (changed[:, 3:] + 17) % CFG["V"]
        changed_logits = model(changed)
        causal_prefix_max_abs_diff = float((logits[:, :3] - changed_logits[:, :3]).abs().max())

        rope_input = torch.randn(CFG["B"], CFG["H"], CFG["T"], CFG["d_h"])
        rope_output = rope(rope_input)
        rope_position_zero_max_abs_diff = float((rope_input[:, :, 0] - rope_output[:, :, 0]).abs().max())
        rope_norm_max_abs_diff = float((rope_input.norm(dim=-1) - rope_output.norm(dim=-1)).abs().max())

        mlp_input = torch.randn(CFG["B"], CFG["T"], CFG["d"])
        mlp = model.blocks[0].mlp
        module_output = mlp(mlp_input)
        explicit_output = F.linear(F.silu(F.linear(mlp_input, mlp.gate.weight)) * F.linear(mlp_input, mlp.up.weight), mlp.down.weight)
        swiglu_max_abs_diff = float((module_output - explicit_output).abs().max())

    checks = {
        "shape_ledger": trace == {
            "tokens": [2, 6], "embedding": [2, 6, 32],
            "q": [2, 4, 6, 8], "k": [2, 2, 6, 8], "v": [2, 2, 6, 8],
            "q_rope": [2, 4, 6, 8], "k_rope": [2, 2, 6, 8],
            "score": [2, 4, 6, 6], "probability": [2, 4, 6, 6],
            "attention_merged": [2, 6, 32], "attention_output": [2, 6, 32],
            "max_future_probability": 0.0, "gate": [2, 6, 64], "up": [2, 6, 64],
            "swiglu_product": [2, 6, 64], "mlp_output": [2, 6, 32],
            "attention_input": [2, 6, 32], "after_attention_residual": [2, 6, 32],
            "mlp_input": [2, 6, 32], "block_output": [2, 6, 32],
            "final_norm": [2, 6, 32], "logits": [2, 6, 64],
        },
        "causal_prefix": causal_prefix_max_abs_diff <= 1e-7,
        "rope_position_zero": rope_position_zero_max_abs_diff <= 1e-7,
        "rope_norm": rope_norm_max_abs_diff <= 1e-5,
        "swiglu_explicit": swiglu_max_abs_diff <= 1e-7,
    }
    result = {
        "task": "E-2", "status": "PASS" if all(checks.values()) else "FAIL",
        "seed": 20261102,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "dtype": "torch.float32"},
        "config": CFG, "shape_trace": trace,
        "numeric": {"causal_prefix_max_abs_diff": causal_prefix_max_abs_diff, "rope_position_zero_max_abs_diff": rope_position_zero_max_abs_diff, "rope_norm_max_abs_diff": rope_norm_max_abs_diff, "swiglu_max_abs_diff": swiglu_max_abs_diff},
        "checks": checks,
        "notes": ["PyTorch Linear stores [out,in]; F.linear applies its transpose to row-vector inputs.", "CPU correctness run only; no GPU timing or performance conclusion."],
    }
    output = Path(__file__).resolve().parents[1] / "experiments" / "ch02_transformer_shapes.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
