#!/usr/bin/env python3
"""W-10.3: deterministic analytic ledger for distributed-training candidates.

This is not a GPU benchmark.  It turns explicit hypothetical throughput/link
inputs into compute, communication, bubble and logical-state estimates.
"""
import json
import math
import platform
from pathlib import Path


P = 5_802_037_248
L = 32
D = 4096
H = 32
H_KV = 8
D_FF = 11_008
T = 4096
TRAIN_FLOPS_PER_SEQUENCE = 168_972_603_359_232
BF16_BYTES = 2
GIB = 2**30

HARDWARE = {
    "device_peak_flops_per_s": 300e12,
    "assumed_mfu_without_communication": 0.40,
    "intra_effective_bytes_per_s": 300e9,
    "inter_effective_bytes_per_s": 50e9,
    "intra_latency_s": 5e-6,
    "inter_latency_s": 20e-6,
    "dp_bucket_bytes": 128 * 2**20,
}


def ring(payload_bytes, group_size, bandwidth, latency, calls=1, bucket_bytes=None):
    if group_size == 1 or calls == 0:
        return {"logical_send_bytes_per_rank": 0, "rounds": 0, "time_s": 0.0}
    chunks = 1 if bucket_bytes is None else math.ceil(payload_bytes / bucket_bytes)
    logical_bytes = calls * 2 * (group_size - 1) / group_size * payload_bytes
    rounds = calls * chunks * 2 * (group_size - 1)
    return {
        "logical_send_bytes_per_rank": int(logical_bytes),
        "rounds": rounds,
        "time_s": rounds * latency + logical_bytes / bandwidth,
    }


def analyze(candidate):
    dp, tp, pp = candidate["dp"], candidate["tp"], candidate["pp"]
    micro, accum = candidate["micro_batch"], candidate["accumulation"]
    zero_stage = candidate["zero_stage"]
    if dp * tp * pp != candidate["devices"]:
        raise ValueError("devices must equal dp*tp*pp")
    if D % tp or H % tp or H_KV % tp or D_FF % tp or L % pp:
        raise ValueError("candidate violates d/H/H_KV/d_ff TP or L PP divisibility")
    if zero_stage not in (0, 1, 3):
        raise ValueError("only simplified ZeRO stages 0, 1 and 3 are modeled")

    throughput = (HARDWARE["device_peak_flops_per_s"] *
                  HARDWARE["assumed_mfu_without_communication"])
    compute_flops = accum * micro * TRAIN_FLOPS_PER_SEQUENCE / (tp * pp)
    compute_s = compute_flops / throughput
    bubble_fraction = (pp - 1) / (accum + pp - 1)
    bubble_s = compute_s * (pp - 1) / accum

    model_shard_params = P / (tp * pp)
    if zero_stage == 0:
        state_bytes = 12 * model_shard_params
    elif zero_stage == 1:
        state_bytes = 4 * model_shard_params + 8 * model_shard_params / dp
    else:
        state_bytes = 12 * model_shard_params / dp

    gradient_bytes = BF16_BYTES * model_shard_params
    dp_comm = ring(gradient_bytes, dp, HARDWARE["inter_effective_bytes_per_s"],
                   HARDWARE["inter_latency_s"], bucket_bytes=HARDWARE["dp_bucket_bytes"])

    hidden_bytes = micro * T * D * BF16_BYTES
    # Teaching approximation inherited from 5.9.2: attention and MLP each
    # contribute one hidden-sized TP collective per layer.  Factor 3 extends
    # the forward-only ledger to a forward/backward main-path approximation.
    tp_calls = accum * 2 * (L // pp) * 3
    tp_comm = ring(hidden_bytes, tp, HARDWARE["intra_effective_bytes_per_s"],
                   HARDWARE["intra_latency_s"], calls=tp_calls)

    critical_edges = min(2, pp - 1)
    pp_calls = 2 * accum * critical_edges
    pp_payload_per_rank = hidden_bytes / tp
    pp_comm = {
        "logical_send_bytes_per_rank": int(pp_calls * pp_payload_per_rank),
        "rounds": pp_calls,
        "time_s": (pp_calls * HARDWARE["inter_latency_s"] +
                   pp_calls * pp_payload_per_rank / HARDWARE["inter_effective_bytes_per_s"]),
    }
    communication_s = dp_comm["time_s"] + tp_comm["time_s"] + pp_comm["time_s"]
    lower_s = max(compute_s, communication_s) + bubble_s
    upper_s = compute_s + communication_s + bubble_s
    global_batch = micro * accum * dp
    return {
        "name": candidate["name"],
        "parallelism": {"devices": candidate["devices"], "dp": dp, "tp": tp,
                        "pp": pp, "zero_stage": zero_stage},
        "batch": {"micro_batch_sequences_per_dp_replica": micro,
                  "accumulation_micro_batches": accum,
                  "global_batch_sequences": global_batch,
                  "input_tokens_per_update": global_batch * T},
        "divisibility": {"d_per_tp": D // tp, "heads_per_tp": H // tp,
                         "kv_heads_per_tp": H_KV // tp, "d_ff_per_tp": D_FF // tp,
                         "layers_per_pp": L // pp},
        "logical_memory": {"parameter_state_bytes_per_rank": int(state_bytes),
                           "parameter_state_gib_per_rank": state_bytes / GIB,
                           "excludes": "activations, temp aggregation, communication buffers, workspace, runtime, fragmentation"},
        "analytic_time": {
            "active_compute_flops_per_rank": int(compute_flops),
            "active_compute_s": compute_s,
            "pipeline_bubble_fraction": bubble_fraction,
            "pipeline_idle_s": bubble_s,
            "communication_serial_s": communication_s,
            "full_overlap_lower_bound_s": lower_s,
            "no_overlap_upper_bound_s": upper_s,
            "not_a_measured_step_time": True,
        },
        "communication": {"dp_ring": dp_comm, "tp_simplified_ring": tp_comm,
                          "pp_critical_stage_p2p": pp_comm,
                          "scope": ("gradient-ring baseline only; ZeRO-3 parameter all-gather and "
                                    "reduce-scatter are unmodeled" if zero_stage == 3 else
                                    "declared DP/TP/PP teaching approximation")},
        "performance_comparison_eligible": zero_stage != 3,
    }


def main():
    candidates = [
        dict(name="single_gpu_fixed_global_batch_16", devices=1, dp=1, tp=1, pp=1,
             zero_stage=0, micro_batch=1, accumulation=16),
        dict(name="ddp8_fixed_global_batch_16", devices=8, dp=8, tp=1, pp=1,
             zero_stage=0, micro_batch=1, accumulation=2),
        dict(name="fsdp8_zero3_ideal_fixed_global_batch_16", devices=8, dp=8, tp=1, pp=1,
             zero_stage=3, micro_batch=1, accumulation=2),
        dict(name="128gpu_A_tp4_pp4_dp8", devices=128, dp=8, tp=4, pp=4,
             zero_stage=0, micro_batch=1, accumulation=16),
        dict(name="128gpu_B_tp8_pp2_dp8", devices=128, dp=8, tp=8, pp=2,
             zero_stage=0, micro_batch=1, accumulation=16),
    ]
    reports = [analyze(candidate) for candidate in candidates]
    by_name = {report["name"]: report for report in reports}
    a = by_name["128gpu_A_tp4_pp4_dp8"]
    b = by_name["128gpu_B_tp8_pp2_dp8"]
    checks = {
        "all_device_products_close": all(r["parallelism"]["devices"] ==
                                         r["parallelism"]["dp"] * r["parallelism"]["tp"] *
                                         r["parallelism"]["pp"] for r in reports),
        "fixed_batch_triplet_is_16": all(by_name[name]["batch"]["global_batch_sequences"] == 16
                                         for name in list(by_name)[:3]),
        "fsdp_ideal_state_is_one_eighth_ddp": (
            by_name["fsdp8_zero3_ideal_fixed_global_batch_16"]["logical_memory"]["parameter_state_bytes_per_rank"] * 8 ==
            by_name["ddp8_fixed_global_batch_16"]["logical_memory"]["parameter_state_bytes_per_rank"]),
        "candidate_A_bubble_is_3_over_19": abs(a["analytic_time"]["pipeline_bubble_fraction"] - 3/19) < 1e-15,
        "candidate_B_bubble_is_1_over_17": abs(b["analytic_time"]["pipeline_bubble_fraction"] - 1/17) < 1e-15,
        "candidate_A_tp_forward_ledger_bytes": (
            a["communication"]["tp_simplified_ring"]["logical_send_bytes_per_rank"] // 3 // 16 == 768 * 2**20),
        "candidate_B_tp_forward_ledger_bytes": (
            b["communication"]["tp_simplified_ring"]["logical_send_bytes_per_rank"] // 3 // 16 == 1792 * 2**20),
        "all_bounds_ordered": all(r["analytic_time"]["full_overlap_lower_bound_s"] <=
                                  r["analytic_time"]["no_overlap_upper_bound_s"] for r in reports),
    }
    result = {
        "task": "W-10.3",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "evidence_type": "analytic_distributed_strategy_ledger_not_gpu_benchmark",
        "environment": {"python": platform.python_version(), "device": "cpu",
                        "multi_process": False, "gpu": False, "nccl": False},
        "model": {"P": P, "L": L, "d": D, "H": H, "H_KV": H_KV,
                  "d_ff": D_FF, "T": T, "training_flops_per_sequence": TRAIN_FLOPS_PER_SEQUENCE},
        "hypothetical_hardware_inputs": HARDWARE,
        "reports": reports,
        "checks": checks,
        "measurement_fields": {"step_time_s": None, "peak_allocated_bytes": None,
                               "peak_reserved_bytes": None, "tokens_per_s": None,
                               "communication_time_s": None,
                               "status": "not_run_no_multi_gpu"},
        "limitations": [
            "All seconds are model outputs from hypothetical throughput, bandwidth and latency inputs, not measurements.",
            "The TP communication ledger is a declared hidden-sized ring approximation, not a framework trace.",
            "The memory field is logical parameter-state payload, not allocated or reserved peak memory.",
            "FSDP/ZeRO-3 includes an ideal uniform state shard but omits transient parameter materialization.",
            "The bounds omit input stalls, topology contention, load imbalance and kernel efficiency changes.",
        ],
    }
    output = Path("doc/experiments/ch10_distributed_strategy_analysis.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
