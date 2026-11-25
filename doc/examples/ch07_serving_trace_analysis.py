#!/usr/bin/env python3
"""Analyze a serving event log without claiming it is a model benchmark.

The bundled input is a synthetic fixture used to validate metric definitions.
For a real benchmark, feed raw client-side events with the same schema and keep
the workload/environment manifest beside the output.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Iterable


def nearest_rank(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        row["_line"] = line_number
        rows.append(row)
    return rows


def validate_row(row: dict, start: float, end: float) -> None:
    required = {"request_id", "arrival_s", "prompt_tokens", "output_tokens", "token_event_s", "status"}
    missing = required - row.keys()
    if missing:
        raise ValueError(f"line {row['_line']}: missing {sorted(missing)}")
    if not start <= row["arrival_s"] < end:
        raise ValueError(f"line {row['_line']}: arrival outside measurement window")
    if row["status"] == "completed":
        events = row["token_event_s"]
        if row["output_tokens"] < 2:
            raise ValueError(f"line {row['_line']}: this exercise requires at least two output tokens")
        if len(events) != row["output_tokens"] or events != sorted(events):
            raise ValueError(f"line {row['_line']}: token events/count mismatch")
        if row["first_token_s"] != events[0] or row["last_token_s"] != events[-1]:
            raise ValueError(f"line {row['_line']}: first/last token timestamp mismatch")
        if row["completion_s"] < row["last_token_s"] or events[0] < row["arrival_s"]:
            raise ValueError(f"line {row['_line']}: non-causal timestamps")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="doc/examples/ch07_synthetic_serving_trace.jsonl")
    parser.add_argument("--output", default="doc/experiments/ch07_synthetic_trace_metrics.json")
    parser.add_argument("--window-start", type=float, default=0.0)
    parser.add_argument("--window-end", type=float, default=10.0)
    parser.add_argument("--ttft-slo", type=float, default=1.5)
    parser.add_argument("--tpot-slo", type=float, default=0.4)
    parser.add_argument("--e2e-slo", type=float, default=4.0)
    args = parser.parse_args()

    if not args.window_end > args.window_start:
        raise ValueError("window-end must exceed window-start")
    source = Path(args.input)
    rows = load_jsonl(source)
    for row in rows:
        validate_row(row, args.window_start, args.window_end)
    completed = [row for row in rows if row["status"] == "completed"]
    duration = args.window_end - args.window_start
    per_request = []
    for row in completed:
        ttft = row["first_token_s"] - row["arrival_s"]
        tpot = (row["last_token_s"] - row["first_token_s"]) / (row["output_tokens"] - 1)
        e2e = row["completion_s"] - row["arrival_s"]
        slo = ttft <= args.ttft_slo and tpot <= args.tpot_slo and e2e <= args.e2e_slo
        per_request.append({
            "request_id": row["request_id"], "ttft_s": ttft, "tpot_s": tpot,
            "e2e_s": e2e, "meets_joint_slo": slo,
        })

    percentiles = {}
    for metric in ("ttft_s", "tpot_s", "e2e_s"):
        values = [row[metric] for row in per_request]
        percentiles[metric] = {f"p{int(p * 100)}": nearest_rank(values, p) for p in (0.5, 0.95, 0.99)}
    status_counts = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    output_tokens = sum(row["output_tokens"] for row in completed)
    joint_slo = sum(row["meets_joint_slo"] for row in per_request)
    result = {
        "status": "PASS",
        "evidence_type": "synthetic_fixture_metric_pipeline_check",
        "measurement_claim": "not a model, GPU, framework, network, or service performance measurement",
        "input": str(source),
        "window": {"start_s": args.window_start, "end_s": args.window_end, "duration_s": duration},
        "slo": {"ttft_s": args.ttft_slo, "tpot_s": args.tpot_slo, "e2e_s": args.e2e_slo},
        "counts": {"arrived": len(rows), "completed": len(completed), "joint_slo": joint_slo, "status": status_counts},
        "rates": {
            "offered_requests_per_s": len(rows) / duration,
            "completed_requests_per_s": len(completed) / duration,
            "output_tokens_per_s": output_tokens / duration,
            "joint_slo_goodput_per_s": joint_slo / duration,
        },
        "percentile_method": "nearest-rank over completed requests",
        "percentiles": percentiles,
        "per_request": per_request,
        "environment": {"python": platform.python_version()},
        "real_service_route": "not_run: CUDA unavailable and vLLM/SGLang/TensorRT-LLM not installed",
        "prefix_cache_route": "not_run",
        "quantization_route": "not_run",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
