#!/usr/bin/env python3
"""E-4: honest CPU training matrix for dtype, micro-batch, and accumulation.

The parent process launches one child per configuration so ru_maxrss is isolated.
This is a CPU teaching experiment; it does not measure GPU memory or throughput.
"""
import argparse
import gc
import importlib.util
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


SEED = 20261103
CFG = dict(L=2, d=64, H=4, H_KV=2, d_h=16, d_ff=176, V=256, B=8, T=32)
MATRIX = [
    ("fp32-b8-a1", "fp32", 8, 1),
    ("fp32-b2-a4", "fp32", 2, 4),
    ("bf16-b8-a1", "bf16", 8, 1),
    ("bf16-b2-a4", "bf16", 2, 4),
]


def load_model_class():
    source = Path(__file__).with_name("ch02_transformer_shapes.py")
    spec = importlib.util.spec_from_file_location("ch02_model", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MiniGPT


def make_data(steps=6):
    generator = torch.Generator().manual_seed(SEED + 1)
    return torch.randint(0, CFG["V"], (steps, CFG["B"], CFG["T"] + 1), generator=generator)


def train_once(dtype_name, micro_batch, accumulation, steps=6, measured=True):
    if micro_batch * accumulation != CFG["B"]:
        raise ValueError("micro_batch * accumulation must equal global batch")
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    model = load_model_class()(CFG).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.01)
    data = make_data(steps)
    use_bf16 = dtype_name == "bf16"
    losses, grad_norms = [], []
    start = time.perf_counter()
    for update in range(steps):
        optimizer.zero_grad(set_to_none=True)
        nll_sum = 0.0
        denominator = CFG["B"] * CFG["T"]
        for offset in range(0, CFG["B"], micro_batch):
            batch = data[update, offset:offset + micro_batch]
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(batch[:, :-1])
                nll = F.cross_entropy(logits.reshape(-1, CFG["V"]), batch[:, 1:].reshape(-1), reduction="sum")
                loss = nll / denominator
            loss.backward()
            nll_sum += float(nll.detach())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        losses.append(nll_sum / denominator)
        grad_norms.append(grad_norm)
    elapsed = time.perf_counter() - start
    checksum = float(sum(parameter.detach().double().sum() for parameter in model.parameters()))
    result = {
        "dtype_path": "cpu_bf16_autocast" if use_bf16 else "cpu_fp32",
        "micro_batch_sequences": micro_batch,
        "accumulation_steps": accumulation,
        "global_batch_sequences": CFG["B"],
        "updates": steps,
        "effective_target_tokens": steps * CFG["B"] * CFG["T"],
        "loss_curve": losses,
        "grad_norm_curve_before_clipping": grad_norms,
        "elapsed_seconds": elapsed,
        "effective_tokens_per_second": steps * CFG["B"] * CFG["T"] / elapsed,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "final_parameter_checksum_fp64_sum": checksum,
        "measured": measured,
    }
    del model, optimizer, data
    gc.collect()
    return result


def child(args):
    # Dry run in an independent model warms common CPU paths; it is excluded from time.
    train_once(args.dtype, args.micro_batch, args.accumulation, steps=1, measured=False)
    result = train_once(args.dtype, args.micro_batch, args.accumulation, steps=args.steps, measured=True)
    print(json.dumps(result, ensure_ascii=False))


def parent(args):
    runs = []
    for name, dtype_name, micro_batch, accumulation in MATRIX:
        command = [sys.executable, __file__, "--child", "--name", name, "--dtype", dtype_name,
                   "--micro-batch", str(micro_batch), "--accumulation", str(accumulation),
                   "--steps", str(args.steps)]
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        record = json.loads(completed.stdout.strip().splitlines()[-1])
        record["name"] = name
        runs.append(record)
    same_fp32_loss = max(abs(a - b) for a, b in zip(runs[0]["loss_curve"], runs[1]["loss_curve"]))
    same_bf16_loss = max(abs(a - b) for a, b in zip(runs[2]["loss_curve"], runs[3]["loss_curve"]))
    result = {
        "task": "E-4",
        "status": "PASS" if same_fp32_loss < 1e-5 and same_bf16_loss < 1e-2 else "FAIL",
        "seed": SEED,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__, "device": "cpu",
            "cuda_available": torch.cuda.is_available(), "threads_per_child": 4,
            "platform": platform.platform(),
        },
        "config": CFG,
        "protocol": {
            "same_initialization_and_data": True, "optimizer": "AdamW(lr=3e-3, betas=(0.9,0.95), wd=0.01)",
            "gradient_clip_norm": 1.0, "warmup_updates_excluded_from_time": 1,
            "rss_scope": "per-child process peak RSS; not GPU memory and not allocator-only model memory",
        },
        "runs": runs,
        "checks": {
            "fp32_accumulation_loss_max_abs_diff": same_fp32_loss,
            "bf16_accumulation_loss_max_abs_diff": same_bf16_loss,
            "fp32_accumulation_tolerance": 1e-5,
            "bf16_accumulation_tolerance": 1e-2,
            "all_finite": all(all(torch.isfinite(torch.tensor(run["loss_curve"]))) for run in runs),
            "all_loss_decreased": all(run["loss_curve"][-1] < run["loss_curve"][0] for run in runs),
        },
        "limitations": [
            "CPU measurements do not establish GPU throughput, CUDA memory, or Tensor Core behavior.",
            "Six updates are a correctness/smoke comparison, not a convergence-quality study.",
            "ru_maxrss is a process high-water mark and includes runtime/library memory.",
        ],
    }
    if not result["checks"]["all_finite"] or not result["checks"]["all_loss_decreased"]:
        result["status"] = "FAIL"
    output = Path(__file__).resolve().parents[1] / "experiments" / "ch04_train_matrix.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--name", default="")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()
    child(args) if args.child else parent(args)


if __name__ == "__main__":
    main()
