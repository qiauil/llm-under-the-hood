#!/usr/bin/env python3
"""Two-process CPU correctness checks for DDP and manual TP collectives.

This is a low-resource fallback for Chapter 5. It does not benchmark GPUs,
NCCL, FSDP, or distributed performance.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


WEIGHT = torch.tensor(
    [[0.1, 0.2, 0.3, 0.4], [-0.2, 0.1, 0.0, 0.5]], dtype=torch.float64
)
X_BY_RANK = [
    torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64),
    torch.tensor([[2.0, 0.0, -1.0, 1.0]], dtype=torch.float64),
]
Y_BY_RANK = [
    torch.tensor([[0.5, -1.0]], dtype=torch.float64),
    torch.tensor([[1.0, 0.25]], dtype=torch.float64),
]
TP_X = torch.tensor(
    [[1.0, 2.0, 3.0, 4.0], [2.0, 0.0, -1.0, 1.0]], dtype=torch.float64
)
TP_W = torch.tensor(
    [
        [0.1, 0.2, 0.3, 0.4],
        [-0.2, 0.5, 0.7, -0.1],
        [0.0, -0.3, 0.2, 0.6],
        [0.8, 0.1, -0.4, 0.2],
    ],
    dtype=torch.float64,
)


def serial_ddp_reference(lr: float) -> torch.Tensor:
    model = nn.Linear(4, 2, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(WEIGHT)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss = nn.functional.mse_loss(
        model(torch.cat(X_BY_RANK, dim=0)), torch.cat(Y_BY_RANK, dim=0)
    )
    loss.backward()
    optimizer.step()
    return model.weight.detach().clone()


def worker(rank: int, world_size: int, init_file: str, out_dir: str, lr: float) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method="file://" + init_file,
        rank=rank,
        world_size=world_size,
    )
    torch.manual_seed(20261105)

    model = nn.Linear(4, 2, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(WEIGHT)
    ddp = DDP(model)
    optimizer = torch.optim.SGD(ddp.parameters(), lr=lr)
    loss = nn.functional.mse_loss(ddp(X_BY_RANK[rank]), Y_BY_RANK[rank])
    loss.backward()
    optimizer.step()
    ddp_weight = model.weight.detach().clone()
    gathered_weights = [torch.empty_like(ddp_weight) for _ in range(world_size)]
    dist.all_gather(gathered_weights, ddp_weight)
    reference_weight = serial_ddp_reference(lr)
    ddp_rank_error = max(
        (weight - gathered_weights[0]).abs().max().item()
        for weight in gathered_weights
    )
    ddp_reference_error = (ddp_weight - reference_weight).abs().max().item()

    in_per_rank = TP_X.shape[1] // world_size
    x_shard = TP_X[:, rank * in_per_rank : (rank + 1) * in_per_rank]
    w_row_shard = TP_W[rank * in_per_rank : (rank + 1) * in_per_rank, :]
    row_output = x_shard @ w_row_shard
    dist.all_reduce(row_output, op=dist.ReduceOp.SUM)
    full_output = TP_X @ TP_W
    row_error = (row_output - full_output).abs().max().item()

    out_per_rank = TP_W.shape[1] // world_size
    w_col_shard = TP_W[:, rank * out_per_rank : (rank + 1) * out_per_rank]
    col_output = TP_X @ w_col_shard
    gathered_columns = [torch.empty_like(col_output) for _ in range(world_size)]
    dist.all_gather(gathered_columns, col_output)
    joined_output = torch.cat(gathered_columns, dim=1)
    col_error = (joined_output - full_output).abs().max().item()

    fsdp = {"status": "not_run", "reason": "probe not attempted"}
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        probe = FSDP(nn.Linear(4, 2), device_id=torch.device("cpu"))
        del probe
        fsdp = {
            "status": "construction_only",
            "reason": "CPU construction succeeded; no FSDP training claim",
        }
    except Exception as exc:
        fsdp = {
            "status": "skipped",
            "reason": type(exc).__name__ + ": " + str(exc),
        }

    record = {
        "rank": rank,
        "local_loss": loss.item(),
        "ddp_rank_max_abs_error": ddp_rank_error,
        "ddp_serial_reference_max_abs_error": ddp_reference_error,
        "row_parallel_max_abs_error": row_error,
        "column_parallel_max_abs_error": col_error,
        "fsdp": fsdp,
    }
    Path(out_dir, "rank-" + str(rank) + ".json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    dist.destroy_process_group()


def run_single_process_simulation(lr: float, output_path: Path) -> None:
    local_grads = []
    local_losses = []
    for rank in range(2):
        model = nn.Linear(4, 2, bias=False, dtype=torch.float64)
        with torch.no_grad():
            model.weight.copy_(WEIGHT)
        loss = nn.functional.mse_loss(model(X_BY_RANK[rank]), Y_BY_RANK[rank])
        loss.backward()
        local_grads.append(model.weight.grad.detach().clone())
        local_losses.append(loss.item())
    averaged_grad = torch.stack(local_grads).mean(dim=0)
    simulated_ddp_weight = WEIGHT - lr * averaged_grad
    reference_weight = serial_ddp_reference(lr)
    ddp_reference_error = (
        simulated_ddp_weight - reference_weight
    ).abs().max().item()

    row_partials = []
    col_outputs = []
    for rank in range(2):
        in_per_rank = TP_X.shape[1] // 2
        row_partials.append(
            TP_X[:, rank * in_per_rank : (rank + 1) * in_per_rank]
            @ TP_W[rank * in_per_rank : (rank + 1) * in_per_rank, :]
        )
        out_per_rank = TP_W.shape[1] // 2
        col_outputs.append(
            TP_X @ TP_W[:, rank * out_per_rank : (rank + 1) * out_per_rank]
        )
    dense_output = TP_X @ TP_W
    row_error = (sum(row_partials) - dense_output).abs().max().item()
    col_error = (torch.cat(col_outputs, dim=1) - dense_output).abs().max().item()
    tolerance = 1e-12
    checks = {
        "simulated_ddp_matches_serial_global_batch": ddp_reference_error
        <= tolerance,
        "row_parallel_algebra_matches_dense": row_error <= tolerance,
        "column_parallel_algebra_matches_dense": col_error <= tolerance,
    }
    output = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": (
            "single-process CPU algebra simulation; Gloo process-group runs were "
            "attempted but blocked by sandbox socket permissions; no distributed "
            "runtime, GPU, FSDP, NCCL, or performance claim"
        ),
        "seed": 20261105,
        "world_size_simulated": 2,
        "dtype": "torch.float64",
        "learning_rate": lr,
        "tolerance": tolerance,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "gloo_available_but_not_runnable_in_sandbox": dist.is_gloo_available(),
        },
        "runtime_status": {
            "cpu_gloo": "blocked: sandbox denied socket operation",
            "fsdp": "not run: no accelerator and no process group",
            "gpu_ddp_tp": "not run: CUDA unavailable",
        },
        "checks": checks,
        "observed": {
            "local_losses": local_losses,
            "ddp_reference_max_abs_error": ddp_reference_error,
            "row_parallel_max_abs_error": row_error,
            "column_parallel_max_abs_error": col_error,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    if output["status"] != "PASS":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="docs/experiments/ch05_distributed_correctness.json",
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--mode",
        choices=("simulate", "gloo"),
        default="simulate",
        help="simulate is the sandbox-safe CPU algebra route",
    )
    args = parser.parse_args()
    if args.world_size != 2:
        raise ValueError("This deterministic fixture requires --world-size 2")
    output_path = Path(args.output)
    if args.mode == "simulate":
        run_single_process_simulation(args.lr, output_path)
        return

    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    with tempfile.TemporaryDirectory(prefix="ch05-dist-") as tmp:
        init_file = os.path.join(tmp, "process-group-init")
        mp.spawn(
            worker,
            args=(args.world_size, init_file, tmp, args.lr),
            nprocs=args.world_size,
            join=True,
        )
        ranks = [
            json.loads(Path(tmp, "rank-" + str(rank) + ".json").read_text())
            for rank in range(args.world_size)
        ]

    tolerance = 1e-12
    checks = {
        "ddp_ranks_equal": max(x["ddp_rank_max_abs_error"] for x in ranks)
        <= tolerance,
        "ddp_matches_serial_global_batch": max(
            x["ddp_serial_reference_max_abs_error"] for x in ranks
        )
        <= tolerance,
        "row_parallel_matches_dense": max(
            x["row_parallel_max_abs_error"] for x in ranks
        )
        <= tolerance,
        "column_parallel_matches_dense": max(
            x["column_parallel_max_abs_error"] for x in ranks
        )
        <= tolerance,
    }
    output = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "2-process CPU/Gloo correctness only; no GPU or performance claim",
        "seed": 20261105,
        "world_size": args.world_size,
        "dtype": "torch.float64",
        "learning_rate": args.lr,
        "tolerance": tolerance,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "gloo_available": dist.is_gloo_available(),
        },
        "checks": checks,
        "ranks": ranks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    if output["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
