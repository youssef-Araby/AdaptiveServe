#!/usr/bin/env python3
"""
C4 benchmark — DynamicKV: cross-layer adaptive KV budget reallocation.

DynamicKV (Zhou et al., 2024). arXiv:2412.14838  |  github.com/DreamMr/DynamicKV

Algorithm (this script, following the paper's Eq. 2/4-6 and the official
`dynamic_v11.py` reference implementation):
  1. Two-pass prefill (SDPA prefix + eager last-window) captures each layer's
     softmaxed attention from the last `window_size` queries to all tokens.
  2. Per layer & head: sum attention over the window queries on the PREFIX
     columns only (the recent window is always retained separately), smooth
     with 1D avgpool (kernel_size=7)  [official budget_compute_per_layer].
  3. Global budget reallocation [paper Eq. 4-6, official
     update_and_reset_budget]: flatten the smoothed scores of ALL
     (layer, head, prefix-position) entries, take the global top
     (budget - window) × n_heads × n_layers, count winners per layer, and
     allocate the global prefix budget (budget - window) × n_layers
     proportionally to those counts (water-filled against per-layer caps
     min(ratio_max × (budget - window), prefix_len) so the configured total
     budget is preserved whenever the caps allow). On sliding-window models
     (phi-3) an additional per-layer cap keeps every layer below
     config.sliding_window throughout decode — see _prefill_and_compress.
  4. Per layer: keep the top budget_l prefix tokens by head-summed smoothed
     score, plus always the last `window_size` tokens. Layer cache lengths
     therefore DIFFER across layers — a GPU feasibility test confirmed that
     transformers 5.7 SDPA decode handles per-layer variable cache lengths
     (llama32_3b and phi3, layer-0 shortest and longest, coherent output).
  5. Decode normally under SDPA — the paper applies no decode-time
     modification; new tokens append to the compressed prefix at fp16.

The global budget equals n_layers × DKV_MAX_CAPACITY (the uniform total used
by the earlier SnapKV-style variant of this config), so results stay
comparable; only its distribution across layers is adaptive.

Simplifications versus the official implementation:
  - The allocation runs ONCE at the end of prefill over all layers'
    statistics (equivalent to the paper's progressive update with
    m = n_layers). The official every-m=4-layers progressive update exists to
    bound prefill memory, which our two-pass capture already avoids, and is
    otherwise a path-dependent trimming of the same rule.
  - Token retention uses a single shared index set per layer (head-summed
    scores); the official code keeps per-(query-)head token sets, which under
    HF requires materializing the KV cache at query-head granularity (its
    llama implementation stores the cache repeat_kv-expanded).

Measures:
  Speed  : TTFT (ms incl. compression), TPOT (ms, SDPA decode), tokens/sec,
           peak VRAM (MB), compressed KV cache size (MB)
  Quality: WikiText-2 perplexity (teacher-forcing, unaffected by compression),
           LongBench accuracy (with DynamicKV-compressed prefill)

Output: runs/p0/C4/{model}/results.json

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
    assert_not_p0_output_path,
    compute_score,
    extract_prompt_features,
    fp16_decode_growth_bytes,
    kv_cache_bytes,
    load_longbench_task,
    postprocess_pred,
    run_ppl,
)

# ---------------------------------------------------------------------------
# DynamicKV hyperparameters
# ---------------------------------------------------------------------------

# MEAN per-layer budget (wt in the paper): the global budget is this value
# × n_layers, redistributed adaptively across layers. Picked so the
# compression ratio is meaningful at the LongBench input length:
#   phi3   max input ~3500 → 512  ≈ 6.8× compression
#   llama3 max input ~7500 → 1024 ≈ 7.3× compression
DKV_MAX_CAPACITY = {
    "phi3":       512,
    "llama3":     1024,
    "llama31_8b": 1024,
    "llama32_3b": 1024,
}

# Number of recent tokens always retained per layer (ws in the paper)
DKV_WINDOW_SIZE = 32

# Smoothing kernel for attention scores (official default 7, avgpool)
DKV_KERNEL_SIZE = 7

# Per-layer cap on the reallocated prefix budget, as a multiple of the mean
# prefix budget (radio_max in the official code, default 10). Non-binding at
# our sequence lengths — the available prefix length binds first.
DKV_RATIO_MAX = 10


# ---------------------------------------------------------------------------
# Compression core
# ---------------------------------------------------------------------------

def _layer_scores(attn: torch.Tensor, window: int, kernel: int,
                  k_len: int) -> torch.Tensor:
    """
    Per-head smoothed window-attention scores over PREFIX columns
    (official budget_compute_per_layer).

    attn : [batch=1, n_heads_q, q_len, k_len_attn] softmaxed attention.
    k_len: length of the actual K cache. If smaller than attn's k dimension
           (e.g. phi-3 sliding-window attention truncates the K cache), only
           the last `k_len` columns are scored so the returned indices align
           with cache layout.

    Returns [n_heads_q, P] float32 scores, P = k_len - min(window, k_len);
    the recent window columns are excluded (they are always retained).
    """
    if attn.shape[-1] > k_len:
        attn = attn[..., -k_len:]
    q_len = attn.shape[-2]
    win = min(window, q_len)
    P = k_len - min(window, k_len)
    if P <= 0:
        return torch.zeros((attn.shape[1], 0), dtype=torch.float32,
                           device=attn.device)
    scores = attn[0, :, -win:, :P].sum(dim=1).float()   # [H, P]

    pad = kernel // 2
    scores = F.avg_pool1d(
        scores.unsqueeze(1),                            # [H, 1, P]
        kernel_size=kernel, stride=1, padding=pad,
    ).squeeze(1)
    return scores


def _proportional_with_caps(weights: list[float], caps: list[int],
                            total: int) -> list[int]:
    """
    Split integer `total` across slots proportionally to `weights`, never
    exceeding per-slot `caps` (water-filling). Guarantees
    sum(result) == min(total, sum(caps)).
    """
    n = len(caps)
    budgets = [0] * n
    remaining = min(total, sum(caps))
    active = [i for i in range(n) if caps[i] > 0]
    while remaining > 0 and active:
        w = [max(float(weights[i]), 0.0) for i in active]
        wsum = sum(w)
        if wsum <= 0:                       # no signal left: split evenly
            w = [1.0] * len(active)
            wsum = float(len(active))
        shares = [remaining * wi / wsum for wi in w]
        floors = [min(int(s), caps[a] - budgets[a])
                  for s, a in zip(shares, active)]
        given = sum(floors)
        if given == 0:
            # Fewer tokens left than active slots: hand out singles by share.
            order = sorted(range(len(active)), key=lambda j: shares[j],
                           reverse=True)
            for j in order:
                if remaining == 0:
                    break
                a = active[j]
                if caps[a] - budgets[a] > 0:
                    budgets[a] += 1
                    remaining -= 1
        else:
            for f, a in zip(floors, active):
                budgets[a] += f
            remaining -= given
        active = [a for a in active if caps[a] - budgets[a] > 0]
    return budgets


def _dkv_allocate(scores: list[torch.Tensor], base: int,
                  ratio_max: int, prefix_cap: int | None = None) -> list[int]:
    """
    DynamicKV global budget reallocation (paper Eq. 4-6, official
    update_and_reset_budget):

      * flatten all layers' per-head smoothed prefix scores,
      * global top-k with k = base × n_heads × n_layers,
      * count winning entries per layer,
      * allocate the global prefix budget base × n_layers proportionally to
        those counts, capped per layer at min(ratio_max × base, prefix_len,
        prefix_cap).

    scores    : list (one per layer) of [n_heads_q, P_l] tensors.
    base      : mean per-layer PREFIX budget (budget - window).
    prefix_cap: optional absolute per-layer cap (sliding-window models; see
                _prefill_and_compress).
    Returns per-layer prefix budgets (list of int).
    """
    n_layers = len(scores)
    total = base * n_layers
    hard = int(ratio_max * base) if prefix_cap is None else min(
        int(ratio_max * base), prefix_cap)
    caps = [min(hard, s.shape[1]) for s in scores]
    if total <= 0:
        return [0] * n_layers
    sizes = [s.numel() for s in scores]
    n_entries = sum(sizes)
    if n_entries == 0:
        return [0] * n_layers

    flat = torch.cat([s.reshape(-1) for s in scores])
    n_heads = max(s.shape[0] for s in scores)
    k_global = min(base * n_heads * n_layers, n_entries)
    top_idx = torch.topk(flat, k_global).indices
    bounds = torch.tensor(sizes, device=flat.device).cumsum(0)
    layer_of = torch.searchsorted(bounds, top_idx, right=True)
    counts = torch.bincount(layer_of, minlength=n_layers).tolist()
    return _proportional_with_caps(counts, caps, total)


def _compress_cache(
    past: DynamicCache,
    attentions: tuple,
    budget: int,
    window: int,
    kernel: int,
    ratio_max: int = DKV_RATIO_MAX,
    prefix_cap: int | None = None,
) -> tuple[DynamicCache, list[int]]:
    """
    Build a new DynamicCache holding each layer's DynamicKV selection: the
    top budget_l prefix tokens (budget_l from the cross-layer reallocation)
    plus the recent window. Per-layer lengths differ.

    Returns the new cache and a list of per-layer kept-token counts.
    """
    n_layers = len(attentions)
    k_lens = [past.layers[L].keys.shape[2] for L in range(n_layers)]

    scores = [_layer_scores(attn, window=window, kernel=kernel, k_len=k_lens[L])
              for L, attn in enumerate(attentions)]

    base = max(budget - window, 0)
    prefix_budgets = _dkv_allocate(scores, base=base, ratio_max=ratio_max,
                                   prefix_cap=prefix_cap)

    new_cache = DynamicCache()
    kept = []
    for L in range(n_layers):
        K = past.layers[L].keys     # [B, n_kv_heads, S, D]
        V = past.layers[L].values
        k_len = k_lens[L]
        n_recent = min(window, k_len)
        b = min(prefix_budgets[L], scores[L].shape[1])

        if b > 0:
            layer_score = scores[L].sum(dim=0)              # head-summed [P]
            top = torch.topk(layer_score, b, largest=True).indices
            recent = torch.arange(k_len - n_recent, k_len, device=K.device)
            idx = torch.cat([top.to(K.device), recent]).sort().values
        else:
            # Window-only layer (allocation 0) or tiny cache (P <= 0).
            idx = torch.arange(k_len - n_recent, k_len, device=K.device)
        kept.append(idx.shape[0])

        Kc = K.index_select(dim=2, index=idx)
        Vc = V.index_select(dim=2, index=idx)
        new_cache.update(Kc, Vc, layer_idx=L)

    return new_cache, kept


def _prefill_and_compress(
    model, ids_t: torch.Tensor, budget: int, window: int, kernel: int,
    device: str, decode_headroom: int = 0,
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
    prefill. After compression the model is switched back to SDPA so the
    decode loop runs the same kernel as the C0 baseline (TPOT comparability).

    decode_headroom: number of decode tokens that will be appended after
    compression. On sliding-window models (phi-3) HF materializes ONE
    sliding-window mask sized from layer 0 once any layer's cache reaches
    config.sliding_window during decode, and applies it to every layer —
    which crashes SDPA when per-layer lengths differ. We therefore cap each
    layer's retained prefix at sliding_window - window - decode_headroom - 1
    so the mask is always skipped (q_len=1, kv_len < sliding_window) and each
    layer attends over its own variable-length cache. The cap only binds when
    a single layer would take > ~3x the mean budget; the water-filling
    reallocates the excess. Models without a sliding window are unaffected.

    Returns (compressed_cache, last_logits[:, -1, :], stats) where stats has
    full_kv_mb / compressed_kv_mb / full_kv_bytes / compressed_kv_bytes /
    avg_kept / kept_min / kept_max / seq_len.
    """
    seq_len = ids_t.shape[1]
    w = min(window, seq_len)

    sliding = getattr(model.config, "sliding_window", None)
    prefix_cap = (max(sliding - window - decode_headroom - 1, 0)
                  if sliding else None)

    if seq_len <= w:
        # Tiny prompt: single eager pass fits trivially.
        model.set_attn_implementation("eager")
        out = model(ids_t, use_cache=True, output_attentions=True)
        full_bytes = kv_cache_bytes(out.past_key_values)
        past, kept = _compress_cache(
            out.past_key_values, out.attentions,
            budget=budget, window=window, kernel=kernel,
            prefix_cap=prefix_cap,
        )
        model.set_attn_implementation("sdpa")
        comp_bytes = kv_cache_bytes(past)
        stats = {"full_kv_mb": full_bytes / (1024 ** 2),
                 "compressed_kv_mb": comp_bytes / (1024 ** 2),
                 "full_kv_bytes": full_bytes,
                 "compressed_kv_bytes": comp_bytes,
                 "avg_kept": sum(kept) / len(kept),
                 "kept_min": min(kept), "kept_max": max(kept),
                 "seq_len": seq_len}
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
    # Physical cache at end of prefill, BEFORE compression: this is the
    # per-prompt FP16 reference (inherits any model-level truncation).
    full_bytes = kv_cache_bytes(out.past_key_values)
    past, kept = _compress_cache(
        out.past_key_values, out.attentions,
        budget=budget, window=window, kernel=kernel,
        prefix_cap=prefix_cap,
    )
    # Decode runs SDPA, same kernel as the C0 baseline.
    model.set_attn_implementation("sdpa")
    comp_bytes = kv_cache_bytes(past)
    stats = {"full_kv_mb": full_bytes / (1024 ** 2),
             "compressed_kv_mb": comp_bytes / (1024 ** 2),
             "full_kv_bytes": full_bytes,
             "compressed_kv_bytes": comp_bytes,
             "avg_kept": sum(kept) / len(kept),
             "kept_min": min(kept), "kept_max": max(kept),
             "seq_len": seq_len}
    return past, out.logits[:, -1, :], stats


def _fp16_bytes_per_token(model) -> int:
    """FP16 KV bytes appended per decoded token:
    2 bytes × 2 (K&V) × n_layers × n_kv_heads × head_dim."""
    cfg = model.config
    n_kv = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or (
        cfg.hidden_size // cfg.num_attention_heads)
    return 2 * 2 * cfg.num_hidden_layers * n_kv * head_dim


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
    print(f"  Prompt: {len(prompt_ids)} tokens  mean budget/layer: {DKV_MAX_CAPACITY[model_key]}")

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
        decode_headroom=SPEED_N_DECODE,
    )
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1000
    print(f"  TTFT (incl. compression): {ttft_ms:.1f} ms")

    full_kv_mb       = stats["full_kv_mb"]
    compressed_kv_mb = stats["compressed_kv_mb"]
    ratio            = full_kv_mb / compressed_kv_mb if compressed_kv_mb > 0 else 1.0
    avg_kept         = stats["avg_kept"]
    print(f"  Full KV: {full_kv_mb:.1f} MB → compressed: {compressed_kv_mb:.1f} MB "
          f"({ratio:.2f}× compression, avg {avg_kept:.0f} tok/layer, "
          f"per-layer {stats['kept_min']}..{stats['kept_max']})")

    # Free anything we don't need before timing decode.
    next_tok = last_logits.argmax(dim=-1, keepdim=True)
    logical_len = input_ids.shape[1]
    del last_logits
    torch.cuda.empty_cache()

    # TPOT: time each decode step (SDPA, same kernel as C0). position_ids must
    # reflect the LOGICAL sequence length (not the compressed cache length) so
    # RoPE for the new query aligns with the original-position-encoded keys we
    # retained.
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
        "kept_min_per_layer":   stats["kept_min"],
        "kept_max_per_layer":   stats["kept_max"],
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
        decode_headroom=max_new,
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

    # Per-prompt byte accounting (§1): decode tokens are kept at fp16 (this
    # config does not compress them; the rebuilt plain cache really grows).
    # The FP16 reference is capped for sliding-window models (Phi-3 FP16
    # serving evicts as it appends at cap, gaining nothing per decode token).
    n_gen = len(generated)
    bpt   = _fp16_bytes_per_token(model)
    stats["n_generated"]  = n_gen
    stats["kv_bytes"]      = stats["compressed_kv_bytes"] + n_gen * bpt
    stats["kv_bytes_fp16"] = stats["full_kv_bytes"] + fp16_decode_growth_bytes(
        stats["full_kv_bytes"], n_gen, bpt, model.config)

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
            pred = postprocess_pred(task, pred)
            full_mbs.append(st["full_kv_mb"])
            comp_mbs.append(st["compressed_kv_mb"])
            kept_list.append(st["avg_kept"])
            seq_lens.append(st["seq_len"])
            s = compute_score(metric, pred, golds,
                              all_classes=sample.get("all_classes"))
            scores.append(s)
            if pp_logger is not None:
                compression = (st["kv_bytes_fp16"] / st["kv_bytes"]
                               if st["kv_bytes"] > 0 else 1.0)
                pp_logger.log(task=task, sample_idx=j, metric=metric, score=s,
                              features=extract_prompt_features(prompt, tokenizer),
                              compression=compression,
                              pred=pred,
                              kv_bytes=st["kv_bytes"],
                              kv_bytes_fp16=st["kv_bytes_fp16"],
                              extra={"kept_min": st["kept_min"],
                                     "kept_max": st["kept_max"],
                                     "n_generated": st["n_generated"]})
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
    p.add_argument("--output", default="runs/p0/C4")
    p.add_argument("--budget", type=int, default=None,
                   help="override mean per-layer budget (default: 512 phi3 / 1024 llama3)")
    p.add_argument("--skip-speed-ppl", action="store_true",
                   help="skip speed + perplexity phases; preserve old fields from existing results.json")
    return p.parse_args()


def main():
    args     = parse_args()
    model_id = MODELS[args.model]
    out_dir  = Path(args.output) / args.model
    assert_not_p0_output_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.budget is not None:
        DKV_MAX_CAPACITY[args.model] = args.budget

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nConfig : C4 (DynamicKV, mean per-layer budget={DKV_MAX_CAPACITY[args.model]}, "
          f"window={DKV_WINDOW_SIZE}, kernel={DKV_KERNEL_SIZE}, ratio_max={DKV_RATIO_MAX})")
    print(f"Model  : {model_id}")
    print(f"Device : {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    print(f"Output : {out_dir}/results.json")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Attention kernels are switched per phase: _prefill_and_compress uses
    # SDPA for the prefix pass, eager for the last-window capture pass (SDPA
    # returns no attention scores on this transformers version), then resets
    # to SDPA so decode runs the same kernel as the C0 baseline.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    results = {
        "config":            "C4",
        "method":            "DynamicKV (cross-layer adaptive budget reallocation, "
                             "shared index set per layer; Zhou et al. 2024)",
        "model":             args.model,
        "model_id":          model_id,
        "max_capacity":      DKV_MAX_CAPACITY[args.model],
        "window_size":       DKV_WINDOW_SIZE,
        "kernel_size":       DKV_KERNEL_SIZE,
        "ratio_max":         DKV_RATIO_MAX,
    }

    if not args.skip_speed_ppl:
        results.update(run_speed(model, tokenizer, args.model, device))
        torch.cuda.empty_cache()

        # PPL doesn't need attention scores; runs on SDPA.
        model.set_attn_implementation("sdpa")
        results.update(run_ppl(model, tokenizer, args.model, device))
        torch.cuda.empty_cache()

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
    assert_not_p0_output_path(out_path)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out_path}")
    print(json.dumps({k: v for k, v in results.items() if k != "longbench"}, indent=2))


if __name__ == "__main__":
    main()
