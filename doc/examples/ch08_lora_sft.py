#!/usr/bin/env python3
"""E-8: deterministic CPU LoRA/SFT smoke experiment on the book's MiniGPT.

This verifies masking, frozen-base LoRA training, a fixed held-out synthetic test,
and merge arithmetic. It is not evidence about real instruction-tuning quality.
"""
import importlib.util
import json
import math
import platform
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


SEED = 20261111
CFG = dict(L=1, d=32, H=4, H_KV=2, d_h=8, d_ff=64, V=64, B=12, T=6)
RANK = 4
ALPHA = 8.0


def load_model_class():
    source = Path(__file__).with_name("ch02_transformer_shapes.py")
    spec = importlib.util.spec_from_file_location("ch02_model_for_e8", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MiniGPT


class LoRALinear(nn.Module):
    """Row-vector LoRA: Y=X W_0 + (alpha/r) X A B."""

    def __init__(self, base, rank=RANK, alpha=ALPHA):
        super().__init__()
        if base.bias is not None:
            raise ValueError("This experiment expects bias-free book-model linears")
        self.base = base
        self.base.weight.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.a = nn.Parameter(torch.empty(base.in_features, rank))
        self.b = nn.Parameter(torch.zeros(rank, base.out_features))
        nn.init.normal_(self.a, mean=0.0, std=0.02)

    @property
    def scale(self):
        return self.alpha / self.rank

    def forward(self, x):
        return F.linear(x, self.base.weight) + self.scale * (x @ self.a @ self.b)

    def merged_weight_storage(self):
        # PyTorch Linear stores [out,in], the transpose of the book's W_math.
        delta_storage = (self.scale * (self.a @ self.b)).transpose(0, 1)
        return self.base.weight + delta_storage


def inject_lora(model):
    wrappers = []
    for block in model.blocks:
        for owner, names in ((block.attention, ("q", "k", "v", "o")),
                             (block.mlp, ("gate", "up", "down"))):
            for name in names:
                wrapped = LoRALinear(getattr(owner, name))
                setattr(owner, name, wrapped)
                wrappers.append(wrapped)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for wrapper in wrappers:
        wrapper.a.requires_grad_(True)
        wrapper.b.requires_grad_(True)
    return wrappers


def make_split(fillers):
    # [BOS, instruction, class, irrelevant filler, SEP, answer, EOS].
    rows = []
    for index, filler in enumerate(fillers):
        cls = 10 + (index % 2)
        answer = 20 + (index % 2)
        rows.append([1, 4, cls, filler, 2, answer, 3])
    return torch.tensor(rows, dtype=torch.long)


def masked_nll(model, sequences):
    inputs, targets = sequences[:, :-1], sequences[:, 1:]
    logits = model(inputs)
    # Targets at original sequence positions 5 and 6 are answer and EOS.
    mask = torch.zeros_like(targets, dtype=torch.bool)
    mask[:, 4:] = True
    per_token = F.cross_entropy(logits.reshape(-1, CFG["V"]), targets.reshape(-1), reduction="none")
    per_token = per_token.view_as(targets)
    return per_token.masked_select(mask).sum() / mask.sum(), logits, mask


def evaluate(model, sequences):
    model.eval()
    with torch.inference_mode():
        loss, logits, mask = masked_nll(model, sequences)
        predictions = logits.argmax(-1)
        targets = sequences[:, 1:]
        token_accuracy = float((predictions[mask] == targets[mask]).float().mean())
        answer_accuracy = float((predictions[:, 4] == targets[:, 4]).float().mean())
        exact_response_accuracy = float((predictions[:, 4:] == targets[:, 4:]).all(-1).float().mean())
    return {
        "assistant_token_loss": float(loss),
        "assistant_token_accuracy": token_accuracy,
        "answer_token_accuracy": answer_accuracy,
        "exact_answer_plus_eos_accuracy": exact_response_accuracy,
    }


def checksum(parameters):
    return float(sum(parameter.detach().double().sum() for parameter in parameters))


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    model = load_model_class()(CFG)
    wrappers = inject_lora(model)
    train = make_split(range(30, 42))
    test = make_split(range(42, 50))
    frozen_parameters = [p for p in model.parameters() if not p.requires_grad]
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    expected_trainable = sum(RANK * (w.base.in_features + w.base.out_features) for w in wrappers)
    actual_trainable = sum(p.numel() for p in trainable_parameters)
    frozen_checksum_before = checksum(frozen_parameters)
    before = evaluate(model, test)

    optimizer = torch.optim.AdamW(trainable_parameters, lr=3e-2, betas=(0.9, 0.95), weight_decay=0.0)
    losses = []
    model.train()
    for step in range(160):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = masked_nll(model, train)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite SFT loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        optimizer.step()
        if step % 10 == 0 or step == 159:
            losses.append({"update": step + 1, "assistant_token_loss": float(loss.detach())})

    after = evaluate(model, test)
    frozen_checksum_after = checksum(frozen_parameters)

    probe = torch.randn(3, 5, wrappers[0].base.in_features)
    with torch.inference_mode():
        separate = wrappers[0](probe)
        merged = F.linear(probe, wrappers[0].merged_weight_storage())
        merge_max_abs_diff = float((separate - merged).abs().max())

    checks = {
        "trainable_count": actual_trainable == expected_trainable,
        "only_lora_trainable": all(p.requires_grad for p in trainable_parameters) and
            all(not p.requires_grad for p in frozen_parameters),
        "frozen_checksum_exact": frozen_checksum_before == frozen_checksum_after,
        "finite_losses": all(math.isfinite(point["assistant_token_loss"]) for point in losses),
        "train_loss_decreased": losses[-1]["assistant_token_loss"] < losses[0]["assistant_token_loss"],
        "heldout_loss_improved": after["assistant_token_loss"] < before["assistant_token_loss"],
        "heldout_answer_accuracy_improved": after["answer_token_accuracy"] > before["answer_token_accuracy"],
        "heldout_exact_response_accuracy": after["exact_answer_plus_eos_accuracy"] == 1.0,
        "merge_equivalence": merge_max_abs_diff <= 1e-5,
    }
    result = {
        "task": "E-8",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "evidence_type": "cpu_synthetic_lora_sft_pipeline_check",
        "seed": SEED,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "torch.float32",
            "cuda_available": torch.cuda.is_available(),
            "threads": 2,
        },
        "config": CFG,
        "lora": {
            "rank": RANK,
            "alpha": ALPHA,
            "target_modules": ["q", "k", "v", "o", "gate", "up", "down"],
            "wrapped_linear_count": len(wrappers),
            "expected_trainable_parameters": expected_trainable,
            "actual_trainable_parameters": actual_trainable,
        },
        "data": {
            "kind": "synthetic classification serialized as chat-like token sequences",
            "train_sequences": len(train),
            "heldout_test_sequences": len(test),
            "sequence_tokens": 7,
            "supervised_target_positions_per_sequence": 2,
            "test_fixed_before_training": True,
        },
        "optimizer": "AdamW(lr=0.03, betas=(0.9,0.95), weight_decay=0)",
        "updates": 160,
        "before": before,
        "after": after,
        "training_trace_every_10_updates": losses,
        "numeric": {
            "frozen_checksum_before": frozen_checksum_before,
            "frozen_checksum_after": frozen_checksum_after,
            "merge_max_abs_diff": merge_max_abs_diff,
        },
        "checks": checks,
        "limitations": [
            "Synthetic token task only; this does not establish natural-language instruction quality.",
            "Held-out fillers test the fixed pipeline but are not a public benchmark.",
            "CPU float32 correctness run only; no GPU memory, throughput, QLoRA, or quality claim.",
            "No preference optimization was run; the optional extension remains unrun.",
        ],
    }
    output = Path("doc/experiments/ch08_lora_sft.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
