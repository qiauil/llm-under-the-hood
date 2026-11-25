#!/usr/bin/env python3
"""GPU-only SDPA math-vs-flash correctness, memory, and timing harness.

This file is delivered unrun in the CPU-only authoring environment. It requires
an explicit --allow-run so a syntax check cannot accidentally become a benchmark.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def call_sdpa(q, k, v, backend, causal):
    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)


def measure(q, k, v, backend, causal, warmup, repeats):
    for _ in range(warmup):
        call_sdpa(q, k, v, backend, causal)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    output = None
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = call_sdpa(q, k, v, backend, causal)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return output, {
        "median_ms": statistics.median(times),
        "p10_ms": percentile(times, 0.10),
        "p90_ms": percentile(times, 0.90),
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-run", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--max-abs-tol", type=float, default=5e-2)
    parser.add_argument("--rel-l2-tol", type=float, default=1e-2)
    parser.add_argument("--output", default="docs/experiments/ch06_sdpa_gpu_benchmark.json")
    args = parser.parse_args()
    if not args.allow_run:
        raise SystemExit("Refusing to benchmark without --allow-run")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; no benchmark artifact written")

    torch.manual_seed(20261105)
    dtype = getattr(torch, args.dtype)
    shape = (args.batch, args.heads, args.length, args.head_dim)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)
    math_out, math_metrics = measure(q, k, v, SDPBackend.MATH, args.causal, args.warmup, args.repeats)
    flash_out, flash_metrics = measure(q, k, v, SDPBackend.FLASH_ATTENTION, args.causal, args.warmup, args.repeats)
    error = (math_out.float() - flash_out.float()).abs()
    rel_l2 = error.norm() / math_out.float().norm().clamp_min(1e-12)
    passed = error.max().item() <= args.max_abs_tol and rel_l2.item() <= args.rel_l2_tol
    record = {
        "status": "PASS" if passed else "FAIL",
        "seed": 20261105,
        "environment": {
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "cuda_runtime": torch.version.cuda,
        },
        "shape": list(shape),
        "dtype": args.dtype,
        "causal": args.causal,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "math": math_metrics,
        "flash": flash_metrics,
        "flash_over_math_speedup": math_metrics["median_ms"] / flash_metrics["median_ms"],
        "output_max_abs_error": error.max().item(),
        "output_mean_abs_error": error.mean().item(),
        "output_relative_l2_error": rel_l2.item(),
        "max_abs_tolerance": args.max_abs_tol,
        "relative_l2_tolerance": args.rel_l2_tol,
        "scope": "CUDA forward-only SDPA forced-backend benchmark; profiler and gradients not included",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
