#!/usr/bin/env python3
"""W-10.1: deterministic CPU MiniGPT train/generate/checkpoint project smoke.

This reuses the book's Chapter 2 model.  The built-in character corpus is a
pipeline fixture, not evidence of natural-language quality or generalization.
"""
import importlib.util
import io
import json
import platform
from pathlib import Path

import torch
import torch.nn.functional as F


SEED = 20261111
CORPUS_UNIT = "to be or not to be.\n"
CORPUS = CORPUS_UNIT * 128
UPDATES = 320
BLOCK_SIZE = 24
BATCH_SIZE = 8


def load_model_class():
    source = Path(__file__).with_name("ch02_transformer_shapes.py")
    spec = importlib.util.spec_from_file_location("ch02_model_for_ch10", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MiniGPT


def make_tokenizer(text):
    symbols = sorted(set(text))
    stoi = {symbol: index for index, symbol in enumerate(symbols)}
    return symbols, stoi


def encode(text, stoi):
    return torch.tensor([stoi[char] for char in text], dtype=torch.long)


def decode(ids, symbols):
    return "".join(symbols[int(index)] for index in ids)


def make_batch(tokens, starts):
    rows = [tokens[start:start + BLOCK_SIZE + 1] for start in starts]
    batch = torch.stack(rows)
    return batch[:, :-1], batch[:, 1:]


@torch.inference_mode()
def evaluate(model, x, y):
    logits = model(x)
    return float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)))


@torch.inference_mode()
def greedy_generate(model, prompt_ids, new_tokens):
    sequence = prompt_ids.clone()
    decisions = []
    for _ in range(new_tokens):
        context = sequence[-BLOCK_SIZE:].unsqueeze(0)
        logits = model(context)
        next_id = int(logits[0, -1].argmax())
        decisions.append({"context_length": int(context.shape[1]), "argmax_id": next_id})
        sequence = torch.cat((sequence, torch.tensor([next_id], dtype=torch.long)))
    return sequence, decisions


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    symbols, stoi = make_tokenizer(CORPUS)
    tokens = encode(CORPUS, stoi)
    cfg = dict(L=2, d=32, H=4, H_KV=2, d_h=8, d_ff=64,
               V=len(symbols), B=BATCH_SIZE, T=BLOCK_SIZE)
    model = load_model_class()(cfg).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.01)

    validation_starts = [0, 1, 2, 3, 4, 5, 6, 7]
    val_x, val_y = make_batch(tokens, validation_starts)
    before_loss = evaluate(model, val_x, val_y)
    before_ids, _ = greedy_generate(model.eval(), encode("to be ", stoi), 24)
    model.train()

    generator = torch.Generator().manual_seed(SEED + 1)
    loss_curve = []
    max_start = len(tokens) - BLOCK_SIZE - 1
    for update in range(UPDATES):
        if update == 120:
            optimizer.param_groups[0]["lr"] = 1e-3
        if update == 240:
            optimizer.param_groups[0]["lr"] = 3e-4
        starts = torch.randint(0, max_start + 1, (BATCH_SIZE,), generator=generator).tolist()
        x, y = make_batch(tokens, starts)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg["V"]), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update % 10 == 0 or update == UPDATES - 1:
            loss_curve.append({"update": update + 1, "loss": float(loss.detach())})

    model.eval()
    after_loss = evaluate(model, val_x, val_y)
    generated_ids, decisions = greedy_generate(model, encode("to be ", stoi), 48)
    generated_text = decode(generated_ids, symbols)
    before_text = decode(before_ids, symbols)

    trace = {}
    with torch.inference_mode():
        reference_logits = model(val_x[:2], trace)
        changed = val_x[:2].clone()
        changed[:, 12:] = (changed[:, 12:] + 1) % cfg["V"]
        changed_logits = model(changed)
        causal_prefix_diff = float((reference_logits[:, :12] - changed_logits[:, :12]).abs().max())

    checkpoint = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "update": UPDATES,
        "config": cfg, "symbols": symbols, "seed": SEED,
    }
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    checkpoint_bytes = buffer.tell()
    buffer.seek(0)
    restored = torch.load(buffer, map_location="cpu", weights_only=False)
    restored_model = load_model_class()(restored["config"]).eval()
    restored_model.load_state_dict(restored["model"])
    with torch.inference_mode():
        restored_logits = restored_model(val_x[:2])
    checkpoint_max_diff = float((reference_logits - restored_logits).abs().max())

    expected_shapes = {
        "tokens": [2, BLOCK_SIZE], "embedding": [2, BLOCK_SIZE, cfg["d"]],
        "q": [2, cfg["H"], BLOCK_SIZE, cfg["d_h"]],
        "k": [2, cfg["H_KV"], BLOCK_SIZE, cfg["d_h"]],
        "v": [2, cfg["H_KV"], BLOCK_SIZE, cfg["d_h"]],
        "logits": [2, BLOCK_SIZE, cfg["V"]],
    }
    checks = {
        "all_logged_losses_finite": all(torch.isfinite(torch.tensor(point["loss"])) for point in loss_curve),
        "validation_loss_decreased": after_loss < before_loss,
        "validation_loss_below_0_05": after_loss < 0.05,
        "causal_prefix_max_abs_diff_le_1e_7": causal_prefix_diff <= 1e-7,
        "shape_trace_matches": all(trace.get(name) == shape for name, shape in expected_shapes.items()),
        "autoregressive_argmax_steps": len(decisions) == 48 and all("argmax_id" in item for item in decisions),
        "generated_contains_training_pattern": "to be or not to be" in generated_text,
        "checkpoint_roundtrip_exact": checkpoint_max_diff == 0.0,
        "checkpoint_has_training_state": all(key in restored for key in ["model", "optimizer", "update", "config", "symbols", "seed"]),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    result = {
        "task": "W-10.1", "status": status, "evidence_type": "cpu_character_minigpt_pipeline_check",
        "seed": SEED,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu",
                        "dtype": "fp32", "cuda_available": torch.cuda.is_available(), "threads": 2},
        "config": cfg,
        "corpus": {"kind": "built_in_repeated_character_fixture", "unit": CORPUS_UNIT,
                   "characters": len(CORPUS), "vocabulary": symbols, "block_size": BLOCK_SIZE},
        "training": {"optimizer": "AdamW(lr=3e-3, betas=(0.9,0.95), wd=0.01); lr=1e-3@121,3e-4@241",
                     "updates": UPDATES, "batch_size": BATCH_SIZE, "loss_curve": loss_curve},
        "observed": {"validation_loss_before": before_loss, "validation_loss_after": after_loss,
                     "generated_before": before_text, "generated_after": generated_text,
                     "shape_trace": trace, "causal_prefix_max_abs_diff": causal_prefix_diff,
                     "checkpoint_bytes_in_memory": checkpoint_bytes,
                     "checkpoint_roundtrip_max_abs_logit_diff": checkpoint_max_diff},
        "checks": checks,
        "limitations": [
            "The repeated character fixture tests the pipeline and memorization, not natural-language quality or generalization.",
            "CPU FP32 measurements do not establish GPU throughput, memory, mixed-precision, or large-model behavior.",
            "Generation recomputes the cropped prefix and does not implement or benchmark KV caching.",
            "The checkpoint round-trip is in memory; crash-safe atomic file commit and interrupted-resume equivalence are not tested here.",
        ],
    }
    output = Path("docs/experiments/ch10_minigpt_project.json" if status == "PASS"
                  else "docs/experiments/ch10_minigpt_project_failed_attempt_01.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
