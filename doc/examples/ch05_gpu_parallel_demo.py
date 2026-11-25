#!/usr/bin/env python3
"""Unrun 2-8 GPU correctness demo for DDP, FSDP, and manual TP.

Launch with torchrun. This file is syntax-checked in the authoring environment,
but the repository's current machine has no usable CUDA device.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP


def fixed_linear(device: torch.device) -> nn.Linear:
    layer = nn.Linear(4, 2, bias=False, device=device, dtype=torch.float64)
    weight = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [-0.2, 0.1, 0.0, 0.5]],
        device=device,
        dtype=torch.float64,
    )
    with torch.no_grad():
        layer.weight.copy_(weight)
    return layer


def local_batch(rank: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    xs = (
        [[1.0, 2.0, 3.0, 4.0]],
        [[2.0, 0.0, -1.0, 1.0]],
    )
    ys = (
        [[0.5, -1.0]],
        [[1.0, 0.25]],
    )
    index = rank % 2
    return (
        torch.tensor(xs[index], device=device, dtype=torch.float64),
        torch.tensor(ys[index], device=device, dtype=torch.float64),
    )


def train_one_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, lr: float) -> float:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
    return loss.item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="doc/experiments/ch05_gpu_parallel.json")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; use ch05_distributed_correctness.py for CPU algebra")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world < 2 or world > 8:
        raise ValueError("Run with 2-8 processes")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(20261105)
    x, y = local_batch(rank, device)

    ddp_module = fixed_linear(device)
    ddp = DDP(ddp_module, device_ids=[local_rank])
    ddp_loss = train_one_step(ddp, x, y, args.lr)
    ddp_weight = ddp_module.weight.detach().clone()

    fsdp_module = FSDP(fixed_linear(device), device_id=device, use_orig_params=True)
    fsdp_loss = train_one_step(fsdp_module, x, y, args.lr)
    with FSDP.summon_full_params(fsdp_module, recurse=True, writeback=False):
        fsdp_weight = fsdp_module.module.weight.detach().clone()
    fsdp_ddp_error = (fsdp_weight - ddp_weight).abs().max()
    dist.all_reduce(fsdp_ddp_error, op=dist.ReduceOp.MAX)

    tp_x = torch.arange(16, device=device, dtype=torch.float64).reshape(2, 8) / 10
    tp_w = (
        torch.arange(64, device=device, dtype=torch.float64).reshape(8, 8) / 100
        - 0.3
    )
    if 8 % world != 0:
        raise ValueError("The fixture requires world size 2, 4, or 8")
    in_width = 8 // world
    row_partial = (
        tp_x[:, rank * in_width : (rank + 1) * in_width]
        @ tp_w[rank * in_width : (rank + 1) * in_width, :]
    )
    dist.all_reduce(row_partial, op=dist.ReduceOp.SUM)
    row_error = (row_partial - tp_x @ tp_w).abs().max()

    out_width = 8 // world
    col_local = tp_x @ tp_w[:, rank * out_width : (rank + 1) * out_width]
    gathered = [torch.empty_like(col_local) for _ in range(world)]
    dist.all_gather(gathered, col_local)
    col_error = (torch.cat(gathered, dim=1) - tp_x @ tp_w).abs().max()
    dist.all_reduce(row_error, op=dist.ReduceOp.MAX)
    dist.all_reduce(col_error, op=dist.ReduceOp.MAX)

    result = {
        "status": "PASS"
        if max(fsdp_ddp_error.item(), row_error.item(), col_error.item())
        <= args.tolerance
        else "FAIL",
        "scope": "GPU/NCCL correctness; timing intentionally omitted",
        "world_size": world,
        "dtype": "torch.float64",
        "seed": 20261105,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(local_rank),
        },
        "observed": {
            "rank0_ddp_loss": ddp_loss if rank == 0 else None,
            "rank0_fsdp_loss": fsdp_loss if rank == 0 else None,
            "fsdp_ddp_weight_max_abs_error": fsdp_ddp_error.item(),
            "row_parallel_max_abs_error": row_error.item(),
            "column_parallel_max_abs_error": col_error.item(),
        },
        "tolerance": args.tolerance,
    }
    if rank == 0:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    dist.destroy_process_group()
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
