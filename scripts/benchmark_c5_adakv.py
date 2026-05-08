#!/usr/bin/env python3
"""
C5 benchmark — Ada-KV: head-wise adaptive budget allocation for KV eviction.

Ada-KV (Feng et al., 2024). arXiv:2407.11550  |  github.com/FFY0/AdaKV

Algorithm (this script, simplified-but-faithful):
  1. Run a two-pass prefill (SDPA prefix + eager last-window) to get the
     last-`window_size` queries' attention scores per layer — same trick as C4.
  2. Per layer, compute per-head attention sums [n_kv_heads, S] and smooth with
     1D avgpool over the sequence axis (kernel_size=7).
  3. Allocate a head-wise budget: total per-layer pool = budget_per_head ×
     n_kv_heads. The pool is split across heads proportional to each head's
     "concentration mass" — the sum of the head's top-`budget_per_head`
     smoothed scores. Heads with concentrated attention get more slots.
  4. Per head, take the top-k_h prefix indices.
  5. Vote tally: each selected prefix index accumulates the selecting head's
     score. Take the top `budget_per_head − window_size` positions by vote
     and add the recent window — final layer length is exactly
     `budget_per_head` (uniform across layers, so HF's DynamicCache invariant
     is preserved for sliding-window models).
  6. Gather K/V along the sequence dim with the chosen indices.
  7. Decode normally.

Differences vs. C4 (DynamicKV):
  - C4 sums attentions across heads BEFORE selecting → uniform per-head
    contribution, single layer-wide top-K.
  - C5 selects PER HEAD with adaptive k_h then vote-aggregates → heads with
    concentrated attention drive which positions are kept (head-wise
    importance is preserved through the vote weights).

Simplifications versus the paper:
  - Paper uses FlashAttention-2 varlen so each head can keep different
    positions; we vote-aggregate to a uniform per-layer budget (compatible
    with HF DynamicCache's shared seq dim across heads).
  - No re-pruning during decode.
  - Question-agnostic scoring (last-window queries), like SnapKV/DynamicKV.

Measures:
  Speed  : TTFT (ms incl. compression), TPOT (ms), tokens/sec, peak VRAM (MB),
           compressed KV cache size (MB)
  Quality: WikiText-2 perplexity (teacher-forcing, unaffected by compression),
           LongBench accuracy (7 tasks, with Ada-KV-compressed prefill)

Output: runs/C5/{model}/results.json

Usage:
    python scripts/benchmark_c5_adakv.py --model phi3
    python scripts/benchmark_c5_adakv.py --model llama3
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
    apply_chat_template,
    compute_score,
    kv_cache_mb,
    load_longbench_task,
    run_ppl,
)

# ---------------------------------------------------------------------------
# Ada-KV hyperparameters
# ---------------------------------------------------------------------------

# Per-head average budget. Total per-layer budget pool is this × n_kv_heads,
# adaptively allocated across heads, then unioned. Matched to C4's per-layer
# budget so the two methods are directly comparable.
ADA_BUDGET_PER_HEAD = {
    "phi3":   512,
    "llama3": 1024,
}

# Recent window always retained (per layer, after union).
ADA_WINDOW_SIZE = 32

# Smoothing kernel (matches SnapKV/DynamicKV convention).
ADA_KERNEL_SIZE = 7


# ---------------------------------------------------------------------------
# Compression core
# ---------------------------------------------------------------------------

def _per_head_scores(attn: torch.Tensor, window: int, kernel: int,
                     k_len: int | None = None) -> torch.Tensor:
    """
    attn : [batch=1, n_heads_q, q_len, k_len_attn] softmaxed attention.
    k_len: optional length of the actual K cache (K may be smaller than attn's
           k dim, e.g. phi-3 sliding-window). Only the last `k_len` columns
           are scored so indices align with cache layout.

    Returns smoothed scores of shape [n_heads_q, k_len].
    """
    if k_len is not None and attn.shape[-1] > k_len:
        attn = attn[..., -k_len:]
    q_len = attn.shape[-2]
    win = min(window, q_len)
    # [n_heads_q, k_len] — sum over the last `win` query rows
    scores = attn[0, :, -win:, :].sum(dim=1).float()

    pad = kernel // 2
    # avg_pool1d expects [N, C, L]; treat heads as the batch dim.
    smoothed = F.avg_pool1d(
        scores.unsqueeze(1),  # [H, 1, k_len]
        kernel_size=kernel, stride=1, padding=pad,
    ).squeeze(1)
    return smoothed


def _adakv_layer_indices(
    per_head_scores: torch.Tensor,    # [n_heads_q, S]
    n_kv_heads: int,
    budget_per_head: int,
    window: int,
) -> torch.Tensor:
    """
    Ada-KV head-wise adaptive allocation, with a UNIFORM per-layer output
    size so HF's DynamicCache invariant (same past length across layers) is
    preserved — required for sliding-window models like phi-3.

    Per-head adaptive budgets are still computed (heads with concentrated
    attention get more slots), each head picks its own top-k_h from the
    prefix, the selections are voted across heads (each head contributes a
    weight = its smoothed score), and the layer-wide top `budget_per_head`
    positions by aggregate vote are kept. The recent window is always kept.

    For GQA, multiple query heads share a single KV head; we fold query
    heads down to KV-head groups by mean-pooling their scores so budgets
    and votes are at the KV-cache granularity.

    Returns sorted ascending indices (RoPE-aligned) for this layer.
    """
    H_q, S = per_head_scores.shape
    target = budget_per_head
    if S <= target:
        return torch.arange(S, device=per_head_scores.device)

    if H_q != n_kv_heads:
        assert H_q % n_kv_heads == 0, f"q_heads {H_q} not divisible by kv_heads {n_kv_heads}"
        group = H_q // n_kv_heads
        scores = per_head_scores.view(n_kv_heads, group, S).mean(dim=1)
    else:
        scores = per_head_scores

    H = n_kv_heads
    n_recent = min(window, S)
    avail = S - n_recent
    if avail <= 0:
        return torch.arange(S - n_recent, S, device=scores.device)

    # Mask out recent window in the prefix-scoring tensor.
    masked = scores.clone()
    masked[:, -n_recent:] = float("-inf")

    # Per-head concentration = sum of top-`budget_per_head` smoothed scores
    # in the prefix region. Heads with peaky attention score high.
    k_probe = min(target, avail)
    top_vals, _ = torch.topk(masked, k_probe, dim=1)
    concentration = top_vals.sum(dim=1).clamp_min(1e-6)
    weights = concentration / concentration.sum()        # [H]

    # Adaptive per-head prefix budgets: sum_h k_h = H * target - H * n_recent.
    prefix_budget = max(0, target * H - H * n_recent)
    raw = (weights * prefix_budget).round().long().clamp(min=0, max=avail)

    # Vote tally: each prefix position p accumulates sum_h (score_h[p]) over
    # heads h that included p in their top-k_h. Positions chosen by many
    # heads (or by heads with strong scores) win.
    votes = torch.zeros(avail, device=scores.device)
    prefix_scores = scores[:, :avail].clone()  # head scores over prefix region only
    for h in range(H):
        kh = int(raw[h].item())
        if kh <= 0:
            continue
        idx_h = torch.topk(masked[h, :avail], kh, largest=True).indices
        votes.index_add_(0, idx_h, prefix_scores[h, idx_h])

    # Take top (target - n_recent) prefix positions by vote.
    n_top = target - n_recent
    if n_top <= 0:
        return torch.arange(S - n_recent, S, device=scores.device)
    n_top = min(n_top, avail)
    top_idx = torch.topk(votes, n_top, largest=True).indices
    recent  = torch.arange(S - n_recent, S, device=scores.device)
    keep    = torch.cat([top_idx, recent]).sort().values
    return keep


def _compress_cache(
    past: DynamicCache,
    attentions: tuple,
    n_kv_heads: int,
    budget_per_head: int,
    window: int,
    kernel: int,
) -> tuple[DynamicCache, list[int]]:
    """
    Build a new DynamicCache where each layer's K and V are gathered along the
    sequence dim using Ada-KV per-head adaptive selection (then unioned).

    Returns the new cache and a list of per-layer kept-token counts.
    """
    new_cache = DynamicCache()
    kept = []

    for L, attn in enumerate(attentions):
        K = past.layers[L].keys     # [B, n_kv_heads, S, D]
        V = past.layers[L].values
        k_len = K.shape[2]

        scores = _per_head_scores(attn, window=window, kernel=kernel, k_len=k_len)
        idx    = _adakv_layer_indices(
            scores, n_kv_heads=n_kv_heads,
            budget_per_head=budget_per_head, window=window,
        )
        kept.append(idx.shape[0])

        Kc = K.index_select(dim=2, index=idx)
        Vc = V.index_select(dim=2, index=idx)
        new_cache.update(Kc, Vc, layer_idx=L)

    return new_cache, kept


def _prefill_and_compress(
    model, ids_t: torch.Tensor, n_kv_heads: int, budget_per_head: int,
    window: int, kernel: int, device: str,
) -> tuple[DynamicCache, torch.Tensor, dict]:
    """
    Two-pass prefill (same structure as C4):
      Pass 1 (SDPA): build prefix KV cache for all but the last `window` tokens.
      Pass 2 (eager, output_attentions=True): forward the last `window`
              queries with the prefix as past_key_values, capturing
              attentions of shape [B, H, window, S].

    Then run Ada-KV per-head adaptive selection on those attentions.
    """
    seq_len = ids_t.shape[1]
    w = min(window, seq_len)

    if seq_len <= w:
        model.set_attn_implementation("eager")
        out = model(ids_t, use_cache=True, output_attentions=True)
        full_mb = kv_cache_mb(out.past_key_values)
        past, kept = _compress_cache(
            out.past_key_values, out.attentions,
            n_kv_heads=n_kv_heads,
            budget_per_head=budget_per_head, window=window, kernel=kernel,
        )
        comp_mb = kv_cache_mb(past)
        stats = {"full_kv_mb": full_mb, "compressed_kv_mb": comp_mb,
                 "avg_kept": sum(kept) / len(kept), "seq_len": seq_len}
        return past, out.logits[:, -1, :], stats

    prefix_ids = ids_t[:, :-w]
    last_ids   = ids_t[:, -w:]

    model.set_attn_implementation("sdpa")
    with torch.no_grad():
        pref = model(prefix_ids, use_cache=True)
    prefix_past = pref.past_key_values
    del pref

    model.set_attn_implementation("eager")
    pos = torch.arange(seq_len - w, seq_len, device=device).unsqueeze(0)
    with torch.no_grad():
        out = model(last_ids, past_key_values=prefix_past, use_cache=True,
                    output_attentions=True, position_ids=pos)
    full_mb = kv_cache_mb(out.past_key_values)
    past, kept = _compress_cache(
        out.past_key_values, out.attentions,
        n_kv_heads=n_kv_heads,
        budget_per_head=budget_per_head, window=window, kernel=kernel,
    )
    comp_mb = kv_cache_mb(past)
    stats = {"full_kv_mb": full_mb, "compressed_kv_mb": comp_mb,
             "avg_kept": sum(kept) / len(kept), "seq_len": seq_len}
    return past, out.logits[:, -1, :], stats


# ---------------------------------------------------------------------------
# Phase 1: Speed
# ---------------------------------------------------------------------------

def _get_n_kv_heads(model) -> int:
    cfg = model.config
    return getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads


def run_speed(model, tokenizer, model_key: str, n_kv_heads: int, device: str) -> dict:
    print("\n=== Speed benchmark (Ada-KV) ===")

    target_len = LB_MAX_INPUT[model_key]
    all_ids    = tokenizer.encode(SPEED_TEXT, add_special_tokens=False)
    if len(all_ids) < target_len:
        raise RuntimeError(
            f"SPEED_TEXT only has {len(all_ids)} tokens; need {target_len}. "
            "Increase the repetition factor in scripts/_common.py."
        )
    prompt_ids = all_ids[:target_len]
    input_ids  = torch.tensor([prompt_ids], device=device)
    print(f"  Prompt: {len(prompt_ids)} tokens  budget/head: "
          f"{ADA_BUDGET_PER_HEAD[model_key]}  kv_heads: {n_kv_heads}")

    # Warm-up
    with torch.no_grad():
        _ = model(input_ids[:, :32], use_cache=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    past, last_logits, stats = _prefill_and_compress(
        model, input_ids, n_kv_heads=n_kv_heads,
        budget_per_head=ADA_BUDGET_PER_HEAD[model_key],
        window=ADA_WINDOW_SIZE, kernel=ADA_KERNEL_SIZE, device=device,
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

    next_tok = last_logits.argmax(dim=-1, keepdim=True)
    logical_len = input_ids.shape[1]
    del last_logits
    torch.cuda.empty_cache()

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
# Phase 3: LongBench
# ---------------------------------------------------------------------------

def _generate_ada(
    model, tokenizer, prompt: str, max_new: int, max_input: int,
    device: str, use_chat: bool, n_kv_heads: int, budget_per_head: int,
) -> tuple[str, dict]:
    if use_chat:
        prompt = apply_chat_template(tokenizer, prompt)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > max_input:
        half = max_input // 2
        ids  = ids[:half] + ids[-half:]
    ids_t = torch.tensor([ids], device=device)
    logical_len = ids_t.shape[1]

    eos_id = tokenizer.eos_token_id
    past, last_logits, stats = _prefill_and_compress(
        model, ids_t, n_kv_heads=n_kv_heads,
        budget_per_head=budget_per_head,
        window=ADA_WINDOW_SIZE, kernel=ADA_KERNEL_SIZE, device=device,
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

            pos = torch.tensor([[logical_len + step_i]], device=device)
            step = model(next_tok, past_key_values=past, use_cache=True,
                         position_ids=pos)
            past     = step.past_key_values
            next_tok = step.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    return tokenizer.decode(generated, skip_special_tokens=True).strip(), stats


def run_longbench(model, tokenizer, model_key: str, n_kv_heads: int,
                  device: str) -> dict:
    print("\n=== LongBench (Ada-KV) ===")
    max_input = LB_MAX_INPUT[model_key]
    budget_per_head = ADA_BUDGET_PER_HEAD[model_key]
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
            pred, st = _generate_ada(model, tokenizer, prompt, max_new, max_input,
                                     device, use_chat=cfg["chat"],
                                     n_kv_heads=n_kv_heads,
                                     budget_per_head=budget_per_head)
            full_mbs.append(st["full_kv_mb"])
            comp_mbs.append(st["compressed_kv_mb"])
            kept_list.append(st["avg_kept"])
            seq_lens.append(st["seq_len"])
            if cfg["first_line"]:
                pred = pred.split("\n")[0].strip()
            scores.append(compute_score(metric, pred, golds))
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
    p.add_argument("--output", default="runs/C5")
    p.add_argument("--budget", type=int, default=None,
                   help="override per-head budget (default: 512 phi3 / 1024 llama3)")
    return p.parse_args()


def main():
    args     = parse_args()
    model_id = MODELS[args.model]
    out_dir  = Path(args.output) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.budget is not None:
        ADA_BUDGET_PER_HEAD[args.model] = args.budget

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nConfig : C5 (Ada-KV, head-wise adaptive budget, "
          f"per-head={ADA_BUDGET_PER_HEAD[args.model]}, window={ADA_WINDOW_SIZE}, "
          f"kernel={ADA_KERNEL_SIZE})")
    print(f"Model  : {model_id}")
    print(f"Device : {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    print(f"Output : {out_dir}/results.json")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    n_kv_heads = _get_n_kv_heads(model)

    results = {
        "config":            "C5",
        "method":            "Ada-KV (head-wise adaptive budget allocation, vote-aggregated to uniform layer length)",
        "model":             args.model,
        "model_id":          model_id,
        "budget_per_head":   ADA_BUDGET_PER_HEAD[args.model],
        "n_kv_heads":        n_kv_heads,
        "window_size":       ADA_WINDOW_SIZE,
        "kernel_size":       ADA_KERNEL_SIZE,
    }

    results.update(run_speed(model, tokenizer, args.model, n_kv_heads, device))
    torch.cuda.empty_cache()

    # PPL: SDPA is much faster than eager (no attention scores needed).
    model.set_attn_implementation("sdpa")
    results.update(run_ppl(model, tokenizer, args.model, device))
    torch.cuda.empty_cache()
    model.set_attn_implementation("eager")

    lb = run_longbench(model, tokenizer, args.model, n_kv_heads, device)
    results["longbench"] = lb
    task_scores = [v["score"] for k, v in lb.items() if not k.startswith("_")]
    if task_scores:
        results["longbench_avg"] = round(sum(task_scores) / len(task_scores), 4)
    if "_compression" in lb:
        c = lb["_compression"]
        results["longbench_avg_seq_len"]          = c["avg_seq_len"]
        results["longbench_avg_kept_per_layer"]   = c["avg_kept_per_layer"]
        results["longbench_avg_kv_cache_mb_full"] = c["avg_kv_cache_mb_full"]
        results["longbench_avg_kv_cache_mb"]      = c["avg_kv_cache_mb"]
        results["longbench_avg_compression_ratio"] = c["avg_compression_ratio"]

    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out_path}")
    summary = {k: v for k, v in results.items() if k != "longbench"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
