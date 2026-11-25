#!/usr/bin/env python3
"""W-10.4 low-resource CPU baseline; never labels CPU SDPA as FlashAttention."""
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def explicit_attention(q, k, v):
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    mask = torch.ones(scores.shape[-2:], dtype=torch.bool).tril()
    return torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1) @ v


def sdpa_attention(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def measure(fn, q, k, v, warmup=3, repeats=15):
    with torch.inference_mode():
        for _ in range(warmup):
            fn(q, k, v)
        values = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            output = fn(q, k, v)
            values.append((time.perf_counter_ns() - start) / 1e6)
    return output, {"median_ms": statistics.median(values),
                    "p10_ms": percentile(values, 0.10),
                    "p90_ms": percentile(values, 0.90),
                    "raw_ms": values}


def svg_chart(rows, output):
    selected = [row for row in rows if row["B"] == 1 and row["d_h"] == 64 and row["dtype"] == "float32"]
    width, height, left, top, bottom = 760, 430, 72, 44, 80
    plot_h = height - top - bottom
    max_y = max(max(row["explicit"]["median_ms"], row["sdpa"]["median_ms"]) for row in selected) * 1.15
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
             '<title id="title">CPU causal Attention baseline, not FlashAttention</title>',
             '<desc id="desc">Representative single-thread CPU timings compare an explicit causal Attention implementation with PyTorch CPU SDPA; the chart contains no CUDA, HBM, GPU memory, or FlashAttention result.</desc>',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:sans-serif;fill:#222}.axis{stroke:#333}.grid{stroke:#ddd}.e{fill:#4c78a8}.s{fill:#f58518}</style>',
             '<text x="380" y="24" text-anchor="middle" font-size="16">CPU causal Attention baseline (not FlashAttention)</text>',
             f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>',
             f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="730" y2="{height-bottom}"/>']
    for i in range(5):
        value = max_y * i / 4
        y = height - bottom - plot_h * i / 4
        parts += [f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="730" y2="{y:.1f}"/>',
                  f'<text x="64" y="{y+4:.1f}" text-anchor="end" font-size="11">{value:.3f}</text>']
    group_w = 620 / len(selected)
    for i, row in enumerate(selected):
        center = left + 42 + i * group_w
        for offset, key, css in ((-18, "explicit", "e"), (4, "sdpa", "s")):
            value = row[key]["median_ms"]
            bar_h = plot_h * value / max_y
            parts.append(f'<rect class="{css}" x="{center+offset:.1f}" y="{height-bottom-bar_h:.1f}" width="18" height="{bar_h:.1f}"/>')
        parts.append(f'<text x="{center:.1f}" y="{height-bottom+20}" text-anchor="middle" font-size="12">T={row["T"]}</text>')
    parts += [f'<text x="18" y="{top+plot_h/2}" transform="rotate(-90 18 {top+plot_h/2})" text-anchor="middle" font-size="12">median runtime (ms)</text>',
              '<rect class="e" x="245" y="382" width="14" height="14"/><text x="265" y="394" font-size="12">explicit score matrix</text>',
              '<rect class="s" x="430" y="382" width="14" height="14"/><text x="450" y="394" font-size="12">PyTorch CPU SDPA</text>',
              '<text x="380" y="420" text-anchor="middle" font-size="10">B=1, H=2, d_h=64, FP32; actual CPU timing, no CUDA/HBM/Flash claim</text>',
              '</svg>']
    Path(output).write_text("\n".join(parts) + "\n", encoding="utf-8")


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20261111)
    rows = []
    for dtype_name, dtype, tolerance in (("float32", torch.float32, 5e-6),
                                         ("float64", torch.float64, 1e-12)):
        for batch in (1, 2):
            for length in (64, 128):
                for head_dim in (32, 64):
                    shape = (batch, 2, length, head_dim)
                    q = torch.randn(shape, dtype=dtype)
                    k = torch.randn(shape, dtype=dtype)
                    v = torch.randn(shape, dtype=dtype)
                    explicit_out, explicit_time = measure(explicit_attention, q, k, v)
                    sdpa_out, sdpa_time = measure(sdpa_attention, q, k, v)
                    error = (explicit_out - sdpa_out).abs()
                    rel_l2 = error.norm() / explicit_out.norm().clamp_min(1e-30)
                    rows.append({"B": batch, "H": 2, "T": length, "d_h": head_dim,
                                 "dtype": dtype_name, "causal": True,
                                 "explicit_score_logical_bytes": batch * 2 * length * length * torch.tensor([], dtype=dtype).element_size(),
                                 "explicit": explicit_time, "sdpa": sdpa_time,
                                 "output_max_abs_error": error.max().item(),
                                 "output_relative_l2_error": rel_l2.item(),
                                 "tolerance": tolerance,
                                 "pass": math.isfinite(error.max().item()) and error.max().item() <= tolerance})
    checks = {"all_numeric": all(row["pass"] for row in rows),
              "matrix_has_16_cases": len(rows) == 16,
              "all_timings_positive": all(row[key]["median_ms"] > 0 for row in rows for key in ("explicit", "sdpa")),
              "logical_score_scales_quadratically": (
                  next(r for r in rows if r["dtype"] == "float32" and r["B"] == 1 and r["T"] == 128 and r["d_h"] == 32)["explicit_score_logical_bytes"] ==
                  4 * next(r for r in rows if r["dtype"] == "float32" and r["B"] == 1 and r["T"] == 64 and r["d_h"] == 32)["explicit_score_logical_bytes"])}
    result = {"task": "W-10.4", "status": "PASS" if all(checks.values()) else "FAIL",
              "evidence_type": "actual_single_thread_cpu_forward_timing_and_correctness_not_flashattention",
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "device": "cpu", "threads": 1, "cuda_available": torch.cuda.is_available()},
              "protocol": {"seed": 20261111, "warmup": 3, "repeats": 15,
                           "matrix": "B={1,2}, H=2, T={64,128}, d_h={32,64}, dtype={float32,float64}, causal"},
              "rows": rows, "checks": checks,
              "physical_peak_memory": {"status": "not_available_for_torch_cpu_allocator", "allocated": None, "reserved": None},
              "gpu_flashattention": {"status": "not_run_no_cuda", "runtime": None, "peak_memory": None,
                                     "profiler": None, "triton": "not_run_optional"},
              "limitations": ["CPU SDPA is not a CUDA FlashAttention kernel.",
                              "The score bytes are an analytic logical intermediate, not measured peak memory.",
                              "CPU timings do not predict GPU speedup or HBM traffic."]}
    Path("docs/experiments/ch10_attention_cpu_baseline.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    svg_chart(rows, "docs/experiments/ch10_attention_cpu_baseline.svg")
    print(json.dumps({"status": result["status"], "checks": checks,
                      "max_abs_error": max(row["output_max_abs_error"] for row in rows)}, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
