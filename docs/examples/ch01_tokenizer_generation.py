#!/usr/bin/env python3
"""E-1: multilingual token counts and controlled sampling grid.

Downloads are disabled by default. Pass --allow-download only after selecting and
recording a suitable small causal language model checkpoint.
"""
import argparse
import json
import platform
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


TEXTS = [
    {"language": "zh", "text": "人工智能帮助人们解决问题。"},
    {"language": "en", "text": "Artificial intelligence helps people solve problems."},
    {"language": "de", "text": "Künstliche Intelligenz hilft Menschen, Probleme zu lösen."},
    {"language": "es", "text": "La inteligencia artificial ayuda a resolver problemas."},
]


def environment():
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": "cpu",
    }


def write_result(path, result):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()

    result = {
        "task": "E-1",
        "status": "preflight",
        "environment": environment(),
        "model_requested": args.model,
        "local_files_only": not args.allow_download,
        "comparison_note": (
            "Parallel short statements are illustrative, not a proof of translation equivalence. "
            "Compare token count together with Unicode code points and UTF-8 bytes."
        ),
        "byte_baseline": [
            {
                **item,
                "unicode_code_points": len(item["text"]),
                "utf8_bytes": len(item["text"].encode("utf-8")),
                "byte_token_count_if_one_byte_per_token": len(item["text"].encode("utf-8")),
            }
            for item in TEXTS
        ],
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, local_files_only=not args.allow_download
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=not args.allow_download
        ).to("cpu")
    except Exception as error:
        result.update({
            "status": "not_run_missing_checkpoint",
            "error_type": type(error).__name__,
            "error": str(error),
            "token_counts": [],
            "generations": [],
        })
        write_result(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    model.eval()
    result["resolved_model_type"] = model.config.model_type
    result["tokenizer_class"] = type(tokenizer).__name__
    result["vocab_size"] = len(tokenizer)
    counts = []
    for item in TEXTS:
        ids = tokenizer(item["text"], add_special_tokens=False)["input_ids"]
        counts.append({
            **item,
            "unicode_code_points": len(item["text"]),
            "utf8_bytes": len(item["text"].encode("utf-8")),
            "token_count": len(ids),
            "token_ids": ids,
            "tokens": tokenizer.convert_ids_to_tokens(ids),
            "round_trip_text": tokenizer.decode(ids, skip_special_tokens=False),
        })

    prompt = "The future of language models is"
    encoded = tokenizer(prompt, return_tensors="pt")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    generations = []
    for temperature in (0.7, 1.0):
        for top_p in (0.8, 1.0):
            for seed in (20261102, 20261103):
                torch.manual_seed(seed)
                with torch.inference_mode():
                    output = model.generate(
                        **encoded,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=0,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=pad_id,
                    )[0]
                continuation = output[encoded["input_ids"].shape[1]:]
                generations.append({
                    "prompt": prompt,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": 0,
                    "seed": seed,
                    "max_new_tokens": args.max_new_tokens,
                    "generated_token_ids": continuation.tolist(),
                    "generated_text": tokenizer.decode(continuation, skip_special_tokens=False),
                    "stopped_on_eos": bool(
                        tokenizer.eos_token_id is not None
                        and tokenizer.eos_token_id in continuation.tolist()
                    ),
                })

    result.update({"status": "measured_cpu", "token_counts": counts, "generations": generations})
    write_result(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
