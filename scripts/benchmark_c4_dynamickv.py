#!/usr/bin/env python3
"""
C4 benchmark — DynamicKV: per-layer attention-driven token retention.

DynamicKV (Zhou et al., 2024). arXiv:2412.14838  |  github.com/DreamMr/DynamicKV

Algorithm (this script, simplified-but-faithful):
  1. Run full prefill with output_attentions=True.
  2. For each layer, score every token by summing softmaxed attention from the
     last `window_size` queries across all heads.
  3. Smooth scores with 1D avgpool (kernel_size=7) along the sequence axis.
  4. Always retain the last `window_size` tokens (recent context).
  5. From the remaining tokens, keep the top (max_capacity_prompt - window_size)
     by smoothed score; index sets differ per layer (per-layer dynamic
     selection).
  6. Replace K/V tensors per layer with the gathered (compressed) versions.
     Each layer ends with the same length so HF's uniform past_seq_len
     invariant holds.
  7. Decode normally — new tokens append to the compressed prefix.

Simplifications versus the paper:
  - Uniform per-layer budget (the paper additionally redistributes a *global*
    budget across layers based on attention concentration, which requires a
    custom Cache implementation; deferred).
  - No re-pruning during decode (paper re-prunes every L tokens for very long
    generations; for our LongBench max_new_tokens ≤ 512 the impact is small).

Measures:
  Speed  : TTFT (ms incl. compression), TPOT (ms), tokens/sec, peak VRAM (MB),
           compressed KV cache size (MB)
  Quality: WikiText-2 perplexity (teacher-forcing, unaffected by compression),
           LongBench accuracy (7 tasks, with DynamicKV-compressed prefill)

Output: runs/C4/{model}/results.json

Usage:
    python scripts/benchmark_c4_dynamickv.py --model phi3
    python scripts/benchmark_c4_dynamickv.py --model llama3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from _common import (
    LB_MAX_INPUT,
    LB_TASKS,
    LB_TEMPLATES,
    MODELS,
    SPEED_N_DECODE,
    SPEED_TEXT,
    PerPromptLogger,
    apply_chat_template,
    compute_score,
    extract_prompt_features,
    kv_cache_mb,
    load_longbench_task,
    run_ppl,
)

# ---------------------------------------------------------------------------
# DynamicKV hyperparameters (paper defaults)
# ---------------------------------------------------------------------------

# Per-layer budget after compression. Paper sweeps {128, 512, 1024, 2048}.
# Picked so the compression ratio is meaningful at the LongBench input length:
#   phi3   max input ~3500 → 512  ≈ 6.8× compression
#   llama3 max input ~7500 → 1024 ≈ 7.3× compression
DKV_MAX_CAPACITY = {
    "phi3":   512,
    "llama3": 1024,
}

# Number of recent tokens always retained (paper default 32)
DKV_WINDOW_SIZE = 32

# Smoothing kernel for attention scores (paper default 7, uses avgpool)
DKV_KERNEL_SIZE = 7


# ---------------------------------------------------------------------------
# Compression core
# ---------------------------------------------------------------------------

def _layer_token_scores(attn: torch.Tensor, window: int, kernel: int,
                        k_len: int | None = None) -> torch.Tensor:
    """
    attn : [batch=1, n_heads, q_len, k_len_attn] softmaxed attention.
    k_len: optional length of the actual K cache. If smaller than attn's k
           dimension (e.g. phi-3 sliding-window attention truncates the K
           cache), only the last `k_len` columns of attn are scored so the
           returned indices align with cache layout.

    Returns a 1-D score over `k_len` (or full) positions.
    """
    if k_len is not None and attn.shape[-1] > k_len:
        attn = attn[..., -k_len:]
    q_len = attn.shape[-2]
    win   = min(window, q_len)
    scores = attn[0, :, -win:, :].sum(dim=(0, 1)).float()

    pad = kernel // 2
    scores = F.avg_pool1d(
        scores.unsqueeze(0).unsqueeze(0),
        kernel_size=kernel, stride=1, padding=pad,
    ).squeeze()
    return scores


def _select_indices(scores: torch.Tensor, budget: int, window: int) -> torch.Tensor:
    """
    Pick `budget` indices to retain: always keep the last `window` tokens, then
    fill the remaining (budget - window) slots with the top-scoring tokens from
    the prefix. Indices returned sorted ascending (preserves chronological
    order so RoPE-aligned keys stay in their original relative positions).
    """
    seq_len = scores.shape[0]
    if seq_len <= budget:
        return torch.arange(seq_len, device=scores.device)

    n_recent = min(window, seq_len)
    n_top    = budget - n_recent
    if n_top <= 0:
        return torch.arange(seq_len - n_recent, seq_len, device=scores.device)

    # Mask out the recent window so it isn't double-counted.
    prefix_scores = scores.clone()
    prefix_scores[-n_recent:] = float("-inf")
    top_idx  = torch.topk(prefix_scores, n_top, largest=True).indices
    recent   = torch.arange(seq_len - n_recent, seq_len, device=scores.device)
    keep     = torch.cat([top_idx, recent])
    return keep.sort().values


def _compress_cache(
    past: DynamicCache,
    attentions: tuple,
    budget: int,
    window: int,
    kernel: int,
) -> tuple[DynamicCache, list[int]]:
    """
    Build a new DynamicCache where each layer's K and V are gathered along the
    sequence dim using per-layer top-`budget` indices.

    Returns the new cache and a list of per-layer kept-token counts.
    """
    new_cache = DynamicCache()
    kept = []

    for L, attn in enumerate(attentions):
        # transformers >= 4.55 stores keys/values per-layer on cache.layers[L]
        K = past.layers[L].keys     # [B, n_kv_heads, S, D]
        V = past.layers[L].values
        k_len = K.shape[2]

        scores = _layer_token_scores(attn, window=window, kernel=kernel, k_len=k_len)
        idx    = _select_indices(scores, budget=budget, window=window)
        kept.append(idx.shape[0])

        Kc = K.index_select(dim=2, index=idx)
        Vc = V.index_select(dim=2, index=idx)
        new_cache.update(Kc, Vc, layer_idx=L)

    return new_cache, kept


def _prefill_and_compress(
    model, ids_t: torch.Tensor, budget: int, window: int, kernel: int,
    device: str,
) -> tuple[DynamicCache, torch.Tensor, dict]:
    """
    Two-pass prefill that captures the per-layer attention scores DynamicKV
    needs without materializing the full [N, N] attention matrix.

    Pass 1 (SDPA, no attentions): build the prefix KV cache cheaply.
    Pass 2 (eager, output_attentions=True): forward only the last `window`
    queries with that prefix as past_key_values. The resulting attentions
    have shape [B, H, window, N] — small enough to fit even for N=7500.

    DynamicKV scoring uses exactly these last-`window` query rows, so the
    two-pass result is mathematically identical to the single-pass eager
    prefill.

    Returns (compressed_cache, last_logits[:, -1, :], stats) where stats has
    full_kv_mb / compressed_kv_mb / avg_kept / seq_len.
    """
    seq_len = ids_t.shape[1]
    w = min(window, seq_len)

    if seq_len <= w:
        # Tiny prompt: single eager pass fits trivially.
        model.set_attn_implementation("eager")
        out = model(ids_t, use_cache=True, output_attentions=True)
        full_mb = kv_cache_mb(out.past_key_values)
        past, kept = _compress_cache(
            out.past_key_values, out.attentions,
            budget=budget, window=window, kernel=kernel,
        )
        comp_mb = kv_cache_mb(past)
        stats = {"full_kv_mb": full_mb, "compressed_kv_mb": comp_mb,
                 "avg_kept": sum(kept) / len(kept), "seq_len": seq_len}
        return past, out.logits[:, -1, :], stats

    prefix_ids = ids_t[:, :-w]
    last_ids   = ids_t[:, -w:]

    # Pass 1: prefix prefill on SDPA (no attentions kept).
    model.set_attn_implementation("sdpa")
    with torch.no_grad():
        pref = model(prefix_ids, use_cache=True)
    prefix_past = pref.past_key_values
    del pref

    # Pass 2: eager over just the last `window` tokens. Explicit position_ids
    # so RoPE uses uncompressed indices [seq_len-w .. seq_len-1].
    model.set_attn_implementation("eager")
    pos = torch.arange(seq_len - w, seq_len, device=device).unsqueeze(0)
    with torch.no_grad():
        out = model(last_ids, past_key_values=prefix_past, use_cache=True,
                    output_attentions=True, position_ids=pos)
    full_mb = kv_cache_mb(out.past_key_values)
    past, kept = _compress_cache(
        out.past_key_values, out.attentions,
        budget=budget, window=window, kernel=kernel,
    )
    comp_mb = kv_cache_mb(past)
    stats = {"full_kv_mb": full_mb, "compressed_kv_mb": comp_mb,
             "avg_kept": sum(kept) / len(kept), "seq_len": seq_len}
    return past, out.logits[:, -1, :], stats


# ---------------------------------------------------------------------------
# Phase 1: Speed
# ---------------------------------------------------------------------------

def run_speed(model, tokenizer, model_key: str, device: str) -> dict:
    print("\n=== Speed benchmark (DynamicKV) ===")

    # Use a realistic long prompt so compression actually triggers (a 1024-tok
    # prompt with budget>=1024 reports 1.0x and hides what the method does).
    target_len = LB_MAX_INPUT[model_key]
    all_ids    = tokenizer.encode(SPEED_TEXT, add_special_tokens=False)
    if len(all_ids) < target_len:
        raise RuntimeError(
            f"SPEED_TEXT only has {len(all_ids)} tokens; need {target_len}. "
            "Increase the repetition factor in scripts/_common.py."
        )
    prompt_ids = all_ids[:target_len]
    input_ids  = torch.tensor([prompt_ids], device=device)
    print(f"  Prompt: {len(prompt_ids)} tokens  budget/layer: {DKV_MAX_CAPACITY[model_key]}")

    # Warm-up
    with torch.no_grad():
        _ = model(input_ids[:, :32], use_cache=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # TTFT: full prefill + compression overhead (two-pass, same as LongBench).
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    past, last_logits, stats = _prefill_and_compress(
        model, input_ids,
        budget=DKV_MAX_CAPACITY[model_key],
        window=DKV_WINDOW_SIZE, kernel=DKV_KERNEL_SIZE, device=device,
    )
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1000
    print(f"  TTFT (incl. compression): {ttft_ms:.1f} ms")

    full_kv_mb       = stats["full_kv_mb"]
    compressed_kv_mb = stats["compressed_kv_mb"]
    ratio            = full_kv_mb / compressed_kv_mb if compressed_kv_mb > 0 else 1.0
    avg_kept         = stats["avg_kept"]
    print(f"  Full KV: {full_kv_mb:.1f} MB → compressed: {compressed_kv_mb:.1f} MB "
          f"({ratio:.2f}× compression, avg {avg_kept:.0f} tok/layer)")

    # Free anything we don't need before timing decode.
    next_tok = last_logits.argmax(dim=-1, keepdim=True)
    logical_len = input_ids.shape[1]
    del last_logits
    torch.cuda.empty_cache()

    # TPOT: time each decode step. position_ids must reflect the LOGICAL
    # sequence length (not the compressed cache length) so RoPE for the new
    # query aligns with the original-position-encoded keys we retained.
    decode_times = []
    with torch.no_grad():
        for step_i in range(SPEED_N_DECODE):
            pos = torch.tensor([[logical_len + step_i]], device=device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            step = model(next_tok, past_key_values=past, use_cache=True,
                         position_ids=pos)
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
        "kv_cache_mb_full":     round(full_kv_mb, 2),
        "kv_cache_mb":          round(compressed_kv_mb, 2),
        "kv_compression_ratio": round(ratio, 3),
        "avg_kept_per_layer":   round(avg_kept, 1),
        "speed_prompt_tokens":  len(prompt_ids),
        "speed_n_decode":       SPEED_N_DECODE,
    }


# ---------------------------------------------------------------------------
# Phase 3: LongBench (DynamicKV-compressed prefill)
# ---------------------------------------------------------------------------

def _generate_dkv(
    model, tokenizer, prompt: str, max_new: int, max_input: int,
    device: str, use_chat: bool, budget: int,
) -> tuple[str, dict]:
    if use_chat:
        prompt = apply_chat_template(tokenizer, prompt)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > max_input:
        half = max_input // 2
        ids  = ids[:half] + ids[-half:]
    ids_t = torch.tensor([ids], device=device)
    logical_len = ids_t.shape[1]   # original prompt length, kept across compression

    eos_id = tokenizer.eos_token_id
    past, last_logits, stats = _prefill_and_compress(
        model, ids_t, budget=budget,
        window=DKV_WINDOW_SIZE, kernel=DKV_KERNEL_SIZE, device=device,
    )
    next_tok = last_logits.argmax(dim=-1, keepdim=True)
    del last_logits

    with torch.no_grad():
        generated = []
        for step_i in range(max_new):
            tok_id = int(next_tok.item())
            if tok_id == eos_id:
                break
            generated.append(tok_id)

            # The new token sits at LOGICAL position `logical_len + step_i`,
            # not at the compressed cache length. Pass position_ids explicitly
            # so RoPE matches the keys retained from prefill.
            pos = torch.tensor([[logical_len + step_i]], device=device)
            step = model(next_tok, past_key_values=past, use_cache=True,
                         position_ids=pos)
            past     = step.past_key_values
            next_tok = step.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    return tokenizer.decode(generated, skip_special_tokens=True).strip(), stats


def run_longbench(model, tokenizer, model_key: str, device: str,
                  pp_logger=None) -> dict:
    print("\n=== LongBench (DynamicKV) ===")
    max_input = LB_MAX_INPUT[model_key]
    budget    = DKV_MAX_CAPACITY[model_key]
    results   = {}
    full_mbs, comp_mbs, kept_list, seq_lens = [], [], [], []

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
            prompt = LB_TEMPLATES[task].format(context=sample.get("context", ""),
                                               input=sample["input"])
            pred, st = _generate_dkv(model, tokenizer, prompt, max_new, max_input,
                                     device, use_chat=cfg["chat"], budget=budget)
            full_mbs.append(st["full_kv_mb"])
            comp_mbs.append(st["compressed_kv_mb"])
            kept_list.append(st["avg_kept"])
            seq_lens.append(st["seq_len"])
            if cfg["first_line"]:
                pred = pred.split("\n")[0].strip()
            s = compute_score(metric, pred, golds)
            scores.append(s)
            if pp_logger is not None:
                pp_logger.log(task=task, sample_idx=j, metric=metric, score=s,
                              features=extract_prompt_features(prompt, tokenizer))
            if (j + 1) % 5 == 0:
                print(f"  {task} [{j+1}/{len(samples)}]  {metric}={sum(scores)/len(scores):.4f}")

        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"  {task}  FINAL {metric}={avg:.4f}  (n={len(scores)})")
        results[task] = {"metric": metric, "score": round(avg, 4), "n": len(scores)}

    if full_mbs:
        avg_full = sum(full_mbs) / len(full_mbs)
        avg_comp = sum(comp_mbs) / len(comp_mbs)
        avg_ratio = avg_full / avg_comp if avg_comp > 0 else 1.0
        avg_seq  = sum(seq_lens) / len(seq_lens)
        avg_kept = sum(kept_list) / len(kept_list)
        print(f"\n  LongBench compression: full {avg_full:.1f} MB → "
              f"compressed {avg_comp:.1f} MB ({avg_ratio:.2f}×, "
              f"avg seq {avg_seq:.0f} → {avg_kept:.0f} kept)")
        results["_compression"] = {
            "avg_seq_len":          round(avg_seq, 1),
            "avg_kept_per_layer":   round(avg_kept, 1),
            "avg_kv_cache_mb_full": round(avg_full, 2),
            "avg_kv_cache_mb":      round(avg_comp, 2),
            "avg_compression_ratio": round(avg_ratio, 3),
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  required=True, choices=list(MODELS.keys()))
    p.add_argument("--output", default="runs/C4")
    p.add_argument("--budget", type=int, default=None,
                   help="override per-layer budget (default: 512 phi3 / 1024 llama3)")
    p.add_argument("--skip-speed-ppl", action="store_true",
                   help="skip speed + perplexity phases; preserve old fields from existing results.json")
    return p.parse_args()


def main():
    args     = parse_args()
    model_id = MODELS[args.model]
    out_dir  = Path(args.output) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.budget is not None:
        DKV_MAX_CAPACITY[args.model] = args.budget

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nConfig : C4 (DynamicKV, per-layer budget={DKV_MAX_CAPACITY[args.model]}, "
          f"window={DKV_WINDOW_SIZE}, kernel={DKV_KERNEL_SIZE})")
    print(f"Model  : {model_id}")
    print(f"Device : {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    print(f"Output : {out_dir}/results.json")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load with eager attention so prefill calls with output_attentions=True
    # return real attention scores (SDPA returns empty in this transformers
    # version). For PPL teacher-forcing we don't need attentions, so we
    # temporarily switch to SDPA — eager attention on llama3-8B at 8192
    # tokens is O(N^2) and dominates total runtime otherwise.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    results = {
        "config":            "C4",
        "method":            "DynamicKV (per-layer dynamic token retention)",
        "model":             args.model,
        "model_id":          model_id,
        "max_capacity":      DKV_MAX_CAPACITY[args.model],
        "window_size":       DKV_WINDOW_SIZE,
        "kernel_size":       DKV_KERNEL_SIZE,
    }

    if not args.skip_speed_ppl:
        results.update(run_speed(model, tokenizer, args.model, device))
        torch.cuda.empty_cache()

        # PPL doesn't need attention scores; SDPA is much faster (eager on
        # llama3-8B at 8192 tokens is O(N^2) per layer).
        model.set_attn_implementation("sdpa")
        results.update(run_ppl(model, tokenizer, args.model, device))
        torch.cuda.empty_cache()
        model.set_attn_implementation("eager")

    lb = run_longbench(model, tokenizer, args.model, device,
                       pp_logger=PerPromptLogger(out_dir / "per_prompt.jsonl",
                                                 config="C4", model=args.model))
    results["longbench"] = lb
    task_scores = [v["score"] for k, v in lb.items() if not k.startswith("_")]
    if task_scores:
        results["longbench_avg"] = round(sum(task_scores) / len(task_scores), 4)
    if "_compression" in lb:
        # Surface LongBench compression at the top level so it's visible in
        # the summary alongside the speed-phase numbers.
        for k, v in lb["_compression"].items():
            results[f"longbench_{k}"] = v

    out_path = out_dir / "results.json"
    if out_path.exists():
        old = json.loads(out_path.read_text())
        results = {**old, **results}
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out_path}")
    print(json.dumps({k: v for k, v in results.items() if k != "longbench"}, indent=2))


if __name__ == "__main__":
    main()
