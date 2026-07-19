#!/usr/bin/env python3
"""
C0 baseline benchmark — HuggingFace FP16, full KV cache, no compression.

Measures:
  Speed  : TTFT (ms), TPOT (ms), tokens/sec, peak VRAM (MB), KV cache size (MB)
  Quality: WikiText-2 perplexity, LongBench accuracy (7 tasks)

Output: runs/p0/C0/{model}/results.json

Usage:
    python scripts/benchmark_c0_baseline.py --model phi3
    python scripts/benchmark_c0_baseline.py --model llama3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from _common import (
    LB_MAX_INPUT,
    LB_TASKS,
    LB_TEMPLATES,
    MODELS,
    SPEED_N_DECODE,
    SPEED_TEXT,
    PerPromptLogger,
    apply_chat_template,
    assert_not_p0_output_path,
    compute_score,
    extract_prompt_features,
    kv_cache_bytes,
    kv_cache_mb,
    load_longbench_task,
    postprocess_pred,
    run_ppl,
)


# ---------------------------------------------------------------------------
# Phase 1: Speed
# ---------------------------------------------------------------------------

def run_speed(model, tokenizer, model_key: str, device: str) -> dict:
    print("\n=== Speed benchmark ===")

    # Match the speed-phase prompt length used by compression configs (LongBench
    # max_input) so KV-cache numbers across configs are directly comparable.
    target_len = LB_MAX_INPUT[model_key]
    all_ids = tokenizer.encode(SPEED_TEXT, add_special_tokens=False)
    if len(all_ids) < target_len:
        raise RuntimeError(
            f"SPEED_TEXT only has {len(all_ids)} tokens; need {target_len}."
        )
    prompt_ids = all_ids[:target_len]
    input_ids  = torch.tensor([prompt_ids], device=device)
    print(f"  Prompt: {len(prompt_ids)} tokens")

    # Warm-up pass (discarded)
    with torch.no_grad():
        _ = model(input_ids[:, :32], use_cache=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # TTFT: time the full prefill
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        prefill_out = model(input_ids, use_cache=True)
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1000
    print(f"  TTFT: {ttft_ms:.1f} ms")

    kv_mb = kv_cache_mb(prefill_out.past_key_values)
    print(f"  KV cache after prefill: {kv_mb:.1f} MB")

    past     = prefill_out.past_key_values
    next_tok = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # TPOT: time each decode step
    decode_times = []
    with torch.no_grad():
        for _ in range(SPEED_N_DECODE):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            step = model(next_tok, past_key_values=past, use_cache=True)
            torch.cuda.synchronize()
            decode_times.append((time.perf_counter() - t0) * 1000)
            past     = step.past_key_values
            next_tok = step.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    tpot_ms        = sum(decode_times) / len(decode_times)
    tokens_per_sec = 1000.0 / tpot_ms
    peak_vram_mb   = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"  TPOT: {tpot_ms:.2f} ms/tok  ({tokens_per_sec:.1f} tok/s)")
    print(f"  Peak VRAM: {peak_vram_mb:.0f} MB")

    return {
        "ttft_ms":              round(ttft_ms, 2),
        "tpot_ms":              round(tpot_ms, 3),
        "tokens_per_sec":       round(tokens_per_sec, 2),
        "peak_vram_mb":         round(peak_vram_mb, 1),
        "kv_cache_mb":          round(kv_mb, 2),
        "kv_compression_ratio": 1.0,
        "speed_prompt_tokens":  len(prompt_ids),
        "speed_n_decode":       SPEED_N_DECODE,
    }


# ---------------------------------------------------------------------------
# Phase 3: LongBench
# ---------------------------------------------------------------------------

def _generate(model, tokenizer, prompt: str, max_new: int, max_input: int, device: str,
              use_chat: bool = True) -> tuple[str, int]:
    """Greedy-decode a prediction.

    Returns (raw_pred_text, kv_bytes) where kv_bytes is the measured size of the
    KV cache at the end of generation. For C0 (FP16, no compression) this is
    both the effective and the FP16-reference byte count.
    """
    if use_chat:
        prompt = apply_chat_template(tokenizer, prompt)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > max_input:
        half = max_input // 2
        ids  = ids[:half] + ids[-half:]
    ids_t = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model.generate(
            ids_t,
            max_new_tokens=max_new,
            max_length=None,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=True,
        )
    pred     = tokenizer.decode(out.sequences[0, len(ids):], skip_special_tokens=True)
    kv_bytes = kv_cache_bytes(out.past_key_values)
    return pred, kv_bytes


def run_longbench(model, tokenizer, model_key: str, device: str,
                  pp_logger=None) -> dict:
    print("\n=== LongBench ===")
    max_input = LB_MAX_INPUT[model_key]
    results   = {}

    for task, cfg in LB_TASKS.items():
        try:
            samples = load_longbench_task(task)
        except FileNotFoundError:
            print(f"  SKIP {task}: file not found")
            continue

        metric  = cfg["metric"]
        max_new = cfg["max_new_tokens"]
        scores  = []

        for j, sample in enumerate(samples):
            golds  = sample["answers"] if isinstance(sample["answers"], list) else [sample["answers"]]
            prompt = LB_TEMPLATES[task].format(context=sample.get("context", ""), input=sample["input"])
            pred_raw, kv_bytes = _generate(model, tokenizer, prompt, max_new, max_input,
                                           device, use_chat=cfg["chat"])
            pred = postprocess_pred(task, pred_raw)
            s = compute_score(metric, pred, golds,
                              all_classes=sample.get("all_classes"))
            scores.append(s)
            if pp_logger is not None:
                # C0 does not compress: effective bytes == FP16-reference bytes.
                pp_logger.log(task=task, sample_idx=j, metric=metric, score=s,
                              features=extract_prompt_features(prompt, tokenizer),
                              compression=1.0, pred=pred,
                              kv_bytes=float(kv_bytes),
                              kv_bytes_fp16=float(kv_bytes))
            if (j + 1) % 5 == 0:
                print(f"  {task} [{j+1}/{len(samples)}]  {metric}={sum(scores)/len(scores):.4f}")

        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"  {task}  FINAL {metric}={avg:.4f}  (n={len(scores)})")
        results[task] = {"metric": metric, "score": round(avg, 4), "n": len(scores)}

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  required=True, choices=list(MODELS.keys()))
    p.add_argument("--output", default="runs/p0/C0")
    p.add_argument("--skip-speed-ppl", action="store_true",
                   help="skip speed + perplexity phases; preserve old fields from existing results.json")
    return p.parse_args()


def main():
    args     = parse_args()
    model_id = MODELS[args.model]
    out_dir  = Path(args.output) / args.model
    assert_not_p0_output_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nConfig : C0 (FP16 full KV cache, no compression)")
    print(f"Model  : {model_id}")
    print(f"Device : {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    print(f"Output : {out_dir}/results.json")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    results = {
        "config":   "C0",
        "method":   "none (FP16 full KV cache)",
        "model":    args.model,
        "model_id": model_id,
    }

    if not args.skip_speed_ppl:
        results.update(run_speed(model, tokenizer, args.model, device))
        torch.cuda.empty_cache()

        results.update(run_ppl(model, tokenizer, args.model, device))
        torch.cuda.empty_cache()

    lb = run_longbench(model, tokenizer, args.model, device,
                       pp_logger=PerPromptLogger(out_dir / "per_prompt.jsonl",
                                                 config="C0", model=args.model))
    results["longbench"] = lb
    if lb:
        results["longbench_avg"] = round(
            sum(v["score"] for v in lb.values()) / len(lb), 4
        )

    out_path = out_dir / "results.json"
    if out_path.exists():
        old = json.loads(out_path.read_text())
        results = {**old, **results}
    assert_not_p0_output_path(out_path)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out_path}")
    print(json.dumps({k: v for k, v in results.items() if k != "longbench"}, indent=2))


if __name__ == "__main__":
    main()
