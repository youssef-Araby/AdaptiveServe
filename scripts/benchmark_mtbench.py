#!/usr/bin/env python3
"""
MT-Bench generation phase — produces model answers ONLY (no API / no judge).

Generations are cached to disk so the (expensive) judge phase can be re-run
many times against different judges without re-doing GPU work.

Why MT-Bench is a separate axis from LongBench:
  - LongBench: long-context (3.5k–7.5k tokens) extractive / summarization tasks.
              KV-cache compression matters → drives our C0..C5 + C6 router.
  - MT-Bench : short open-ended chat prompts (~50–300 tokens). KV compression
              is largely irrelevant. Used here to evaluate *model* routing
              (Phi-3 vs LLaMA-3) and as a published baseline for BEST-Route.

Output:
  runs/mtbench/{config}/{model}/generations.jsonl

Each row:
  {prompt_id, category, turn, model, config, prompt, response,
   prompt_tokens, response_tokens, ttft_ms, total_time_s}

Usage:
  python scripts/benchmark_mtbench.py --model phi3        --config C0
  python scripts/benchmark_mtbench.py --model llama3      --config C0
  python scripts/benchmark_mtbench.py --model llama31_8b  --config C0
  python scripts/benchmark_mtbench.py --model llama32_3b  --config C0
  python scripts/benchmark_mtbench.py --model phi3        --config C0 --turn 2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from _common import (
    MODELS,
    MTBENCH_MAX_NEW_TOKENS,
    apply_chat_template,
    load_mtbench,
)

SUPPORTED_CONFIGS = {"C0"}  # generation under FP16 full KV cache.
# Note: extending to C1..C5 requires importing each script's KV strategy.
# Track A (router work) does not need them — MT-Bench prompts are short
# enough that per-config KV compression is a no-op for quality.


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  required=True, choices=list(MODELS.keys()))
    p.add_argument("--config", default="C0", choices=sorted(SUPPORTED_CONFIGS),
                   help="KV-cache config tag for output dir (only C0 supported "
                        "in this version — short prompts make compression moot).")
    p.add_argument("--turn",   type=int, default=1, choices=[1, 2],
                   help="MT-Bench turn (1 = single-turn pointwise; 2 requires "
                        "turn-1 generations as context, not yet implemented).")
    p.add_argument("--output", default="runs/mtbench")
    p.add_argument("--limit",  type=int, default=None,
                   help="Generate only the first N prompts (for smoke tests).")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-generate even if generations.jsonl already exists.")
    return p.parse_args()


@torch.no_grad()
def generate_one(model, tokenizer, prompt: str, device: str) -> dict:
    text  = apply_chat_template(tokenizer, prompt)
    ids   = tokenizer.encode(text, add_special_tokens=False)
    ids_t = torch.tensor([ids], device=device)
    attn  = torch.ones_like(ids_t)
    n_in  = ids_t.shape[1]

    # Time-to-first-token via prefill + 1 decode step.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pre = model(ids_t, attention_mask=attn, use_cache=True)
    first = pre.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1000

    # Full generation for the response text + tokens.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(
        ids_t,
        attention_mask=attn,
        max_new_tokens=MTBENCH_MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    torch.cuda.synchronize()
    total_s = time.perf_counter() - t0

    response_ids = out[0, n_in:]
    response     = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    # Suppress unused first-token warning.
    del first, pre

    return {
        "response":        response,
        "prompt_tokens":   n_in,
        "response_tokens": int(response_ids.shape[0]),
        "ttft_ms":         round(ttft_ms, 2),
        "total_time_s":    round(total_s, 3),
    }


def main():
    args     = parse_args()
    model_id = MODELS[args.model]
    out_dir  = Path(args.output) / args.config / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generations.jsonl"

    if out_path.exists() and not args.overwrite:
        print(f"Already exists (use --overwrite to redo): {out_path}")
        return

    if args.turn != 1:
        raise NotImplementedError("Turn 2 requires conditioning on cached turn-1 "
                                  "responses; not implemented in this version.")

    prompts = load_mtbench(turn=args.turn)
    if args.limit is not None:
        prompts = prompts[:args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nMT-Bench generation")
    print(f"  Config : {args.config} (FP16 full KV cache)")
    print(f"  Model  : {model_id}")
    print(f"  Turn   : {args.turn}")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Output : {out_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, clean_up_tokenization_spaces=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    out_path.write_text("")  # truncate
    with out_path.open("a") as f:
        for i, p in enumerate(prompts, 1):
            res = generate_one(model, tokenizer, p["prompt"], device)
            row = {
                "prompt_id": p["prompt_id"],
                "category":  p["category"],
                "turn":      args.turn,
                "model":     args.model,
                "config":    args.config,
                "prompt":    p["prompt"],
                "reference": p["reference"],
                **res,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"  [{i:>3}/{len(prompts)}] q{p['prompt_id']:<3} "
                  f"{p['category']:<14} "
                  f"{res['response_tokens']:>4}tok  "
                  f"{res['total_time_s']:>5.2f}s")

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
