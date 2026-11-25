#!/usr/bin/env python3
"""W-10.2: analytic parameter/FLOPs/state/KV ledger with MiniGPT validation."""
import importlib.util
import json
import platform
from pathlib import Path

import torch


CONFIGS = {
    "book_reference": dict(L=32, d=4096, H=32, H_KV=8, d_h=128, d_ff=11008, V=32000, B=1, T=4096),
    "minigpt": dict(L=2, d=32, H=4, H_KV=2, d_h=8, d_ff=64, V=64, B=2, T=24),
}


def load_model_class():
    source = Path(__file__).with_name("ch02_transformer_shapes.py")
    spec = importlib.util.spec_from_file_location("ch02_model_for_analyzer", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MiniGPT


def analyze(c):
    if c["d"] != c["H"] * c["d_h"] or c["H"] % c["H_KV"]:
        raise ValueError("require d=H*d_h and H divisible by H_KV")
    embed = c["V"] * c["d"]
    q_o = 2 * c["d"] ** 2
    k_v = 2 * c["d"] * c["H_KV"] * c["d_h"]
    mlp = 3 * c["d"] * c["d_ff"]
    norms = 2 * c["d"]
    per_layer = q_o + k_v + mlp + norms
    params = embed + c["L"] * per_layer + c["d"]
    block_flops = (4 * c["B"] * c["T"] * c["d"] ** 2 +
                   4 * c["B"] * c["T"] * c["d"] * c["H_KV"] * c["d_h"] +
                   6 * c["B"] * c["T"] * c["d"] * c["d_ff"] +
                   4 * c["B"] * c["T"] ** 2 * c["d"])
    forward_flops = c["L"] * block_flops + 2 * c["B"] * c["T"] * c["d"] * c["V"]
    kv_bytes = 2 * c["B"] * c["L"] * c["T"] * c["H_KV"] * c["d_h"] * 2
    return {
        "parameter_breakdown": {"tied_embedding": embed, "per_layer_q_o": q_o,
            "per_layer_k_v": k_v, "per_layer_swiglu": mlp, "per_layer_norms": norms,
            "per_layer_total": per_layer, "final_norm": c["d"], "total": params},
        "flops": {"per_block_full_square_forward_main": block_flops,
            "model_full_sequence_forward_main": forward_flops,
            "training_forward_backward_3x_main": 3 * forward_flops},
        "logical_memory_bytes": {"bf16_weights_grad_fp32_adam_no_master_12B_per_param": 12 * params,
            "bf16_weights_grad_fp32_master_adam_16B_per_param": 16 * params,
            "bf16_kv_cache": kv_bytes},
    }


def main():
    reports = {name: analyze(config) for name, config in CONFIGS.items()}
    mini_model = load_model_class()(CONFIGS["minigpt"])
    actual = sum(parameter.numel() for parameter in mini_model.parameters())
    expected_reference = 5_802_037_248
    checks = {
        "reference_parameter_total": reports["book_reference"]["parameter_breakdown"]["total"] == expected_reference,
        "reference_forward_flops": reports["book_reference"]["flops"]["model_full_sequence_forward_main"] == 56_324_201_119_744,
        "reference_kv_bytes": reports["book_reference"]["logical_memory_bytes"]["bf16_kv_cache"] == 536_870_912,
        "minigpt_actual_parameter_match": reports["minigpt"]["parameter_breakdown"]["total"] == actual,
        "all_positive": all(value > 0 for report in reports.values() for group in report.values() for value in group.values()),
    }
    result = {
        "task": "W-10.2", "status": "PASS" if all(checks.values()) else "FAIL",
        "evidence_type": "analytic_resource_ledger_with_cpu_parameter_validation",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu",
                        "cuda_available": torch.cuda.is_available()},
        "assumptions": {"architecture": "bias-free GQA SwiGLU Pre-RMSNorm tied embedding",
            "flop_convention": "2 FLOPs per MAC; full-square causal implementation; all-position LM head",
            "state_memory": "logical payload only; no activation/temp/workspace/allocator/communication",
            "kv_memory": "BF16 K and V, all layers, equal lengths; no paging/metadata/fragmentation"},
        "configs": CONFIGS, "reports": reports,
        "actual_validation": {"minigpt_sum_parameter_numel": actual}, "checks": checks,
        "physical_gpu_peak": {"status": "not_run_no_cuda", "allocated": None, "reserved": None},
        "limitations": ["Analytic FLOPs are not profiler coverage or wall-clock performance.",
            "Logical bytes are not physical allocated/reserved peak memory.",
            "The 3x training FLOPs factor omits recomputation and lower-order operations."],
    }
    output = Path("doc/experiments/ch10_resource_analyzer.json" if result["status"] == "PASS"
                  else "doc/experiments/ch10_resource_analyzer_failed_attempt_01.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
