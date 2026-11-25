#!/usr/bin/env python3
"""CPU FP64 checks for blockwise Online Softmax attention.

This validates algebra and PyTorch's CPU SDPA math path. It does not run or
benchmark a CUDA FlashAttention kernel.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import torch
import torch.nn.functional as F


def explicit_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        mask = torch.ones(scores.shape[-2:], dtype=torch.bool, device=scores.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def blockwise_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_q: int,
    block_k: int,
    causal: bool,
) -> torch.Tensor:
    """Pedagogical differentiable implementation; not a fused GPU kernel."""
    batch, heads, length, head_dim = q.shape
    outputs = []
    scale = 1.0 / math.sqrt(head_dim)
    for q_start in range(0, length, block_q):
        q_end = min(q_start + block_q, length)
        qi = q[:, :, q_start:q_end, :]
        rows = q_end - q_start
        m = torch.full((batch, heads, rows), float("-inf"), dtype=q.dtype, device=q.device)
        ell = torch.zeros_like(m)
        z = torch.zeros((batch, heads, rows, head_dim), dtype=q.dtype, device=q.device)
        for k_start in range(0, length, block_k):
            k_end = min(k_start + block_k, length)
            kj = k[:, :, k_start:k_end, :]
            vj = v[:, :, k_start:k_end, :]
            raw = (qi @ kj.transpose(-1, -2)) * scale
            if causal:
                q_pos = torch.arange(q_start, q_end, device=q.device)[:, None]
                k_pos = torch.arange(k_start, k_end, device=q.device)[None, :]
                local_mask = k_pos <= q_pos
            else:
                local_mask = torch.ones((rows, k_end - k_start), dtype=torch.bool, device=q.device)
            valid = local_mask.any(dim=-1).view(1, 1, rows)
            masked = raw.masked_fill(~local_mask, float("-inf"))
            tile_m = masked.amax(dim=-1)
            safe_tile_m = torch.where(valid, tile_m, torch.zeros_like(tile_m))
            p_tilde = torch.where(
                local_mask,
                torch.exp(raw - safe_tile_m.unsqueeze(-1)),
                torch.zeros_like(raw),
            )
            tile_ell = p_tilde.sum(dim=-1)
            tile_z = p_tilde @ vj
            new_m = torch.maximum(m, tile_m)
            old_valid = torch.isfinite(m)
            old_delta = torch.where(old_valid, m - new_m, torch.zeros_like(m))
            tile_delta = torch.where(valid, tile_m - new_m, torch.zeros_like(tile_m))
            alpha = torch.where(old_valid, torch.exp(old_delta), torch.zeros_like(m))
            beta = torch.where(valid, torch.exp(tile_delta), torch.zeros_like(tile_m))
            ell = alpha * ell + beta * tile_ell
            z = alpha.unsqueeze(-1) * z + beta.unsqueeze(-1) * tile_z
            m = new_m
        outputs.append(z / ell.unsqueeze(-1))
    return torch.cat(outputs, dim=-2)


def output_and_grads(impl, q0: torch.Tensor, k0: torch.Tensor, v0: torch.Tensor, probe: torch.Tensor):
    q = q0.detach().clone().requires_grad_(True)
    k = k0.detach().clone().requires_grad_(True)
    v = v0.detach().clone().requires_grad_(True)
    out = impl(q, k, v)
    (out * probe).sum().backward()
    return out.detach(), (q.grad.detach(), k.grad.detach(), v.grad.detach())


def max_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/experiments/ch06_online_softmax_correctness.json")
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    torch.manual_seed(20261105)
    shape = (2, 3, 17, 8)
    q0 = torch.randn(shape, dtype=torch.float64)
    k0 = torch.randn(shape, dtype=torch.float64)
    v0 = torch.randn(shape, dtype=torch.float64)
    probe = torch.randn(shape, dtype=torch.float64)
    cases = []
    for causal in (False, True):
        reference = lambda q, k, v, c=causal: explicit_attention(q, k, v, c)
        blocked = lambda q, k, v, c=causal: blockwise_attention(q, k, v, 4, 5, c)
        sdpa = lambda q, k, v, c=causal: F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=c, scale=1.0 / math.sqrt(shape[-1])
        )
        ref_out, ref_grads = output_and_grads(reference, q0, k0, v0, probe)
        block_out, block_grads = output_and_grads(blocked, q0, k0, v0, probe)
        sdpa_out, sdpa_grads = output_and_grads(sdpa, q0, k0, v0, probe)
        case = {
            "causal": causal,
            "block_q": 4,
            "block_k": 5,
            "blockwise_output_max_abs_error": max_error(block_out, ref_out),
            "blockwise_gradient_max_abs_errors": {
                name: max_error(got, expected)
                for name, got, expected in zip(("q", "k", "v"), block_grads, ref_grads)
            },
            "cpu_sdpa_output_max_abs_error": max_error(sdpa_out, ref_out),
            "cpu_sdpa_gradient_max_abs_errors": {
                name: max_error(got, expected)
                for name, got, expected in zip(("q", "k", "v"), sdpa_grads, ref_grads)
            },
        }
        values = [
            case["blockwise_output_max_abs_error"],
            *case["blockwise_gradient_max_abs_errors"].values(),
            case["cpu_sdpa_output_max_abs_error"],
            *case["cpu_sdpa_gradient_max_abs_errors"].values(),
        ]
        case["pass"] = all(math.isfinite(x) and x <= args.tolerance for x in values)
        cases.append(case)

    output = {
        "status": "PASS" if all(case["pass"] for case in cases) else "FAIL",
        "scope": (
            "single-process CPU float64 algebra/autograd check of a pedagogical blockwise "
            "Online Softmax attention implementation plus PyTorch CPU SDPA math; no CUDA, "
            "FlashAttention fused kernel, GPU memory, timing, profiler, Nsight, or Triton claim"
        ),
        "seed": 20261105,
        "shape_b_h_t_dh": list(shape),
        "dtype": "torch.float64",
        "tolerance": args.tolerance,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        },
        "cases": cases,
        "gpu_status": "not run: CUDA unavailable",
        "triton_status": "not run: optional challenge only",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    if output["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
