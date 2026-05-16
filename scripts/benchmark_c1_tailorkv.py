#!/usr/bin/env python3
"""
C1 benchmark — TailorKV: hybrid per-layer KV cache compression.

TailorKV (Yao et al., 2025). arXiv:2505.19586

Algorithm (this script, simplified-but-faithful):
  Layers are split into two classes:
    - Quantization-friendly layers (Q): dense attention, global info.
        → 1-bit KIVI-style quantization: per-channel for K, per-token for V.
    - Sparsity-friendly layers (S): sparse attention, dominant tokens.
        → SnapKV-style retention: keep `n_local` recent tokens plus the
          top-`n_topk` prefix tokens scored by the last-window attention
          aggregated across heads.

  The paper identifies Q layers offline via a "dense preference score"
  (Eq. 6-9). For Llama-3.1-8B the prescribed set is Q={0}; for Yi/Llama-2
  it is Q={0,1}. We default to Q={0} for both phi3 and llama3 — overridable
  with --q-layers.

Simplifications versus the paper:
  - No CPU offloading / asynchronous prefetching. We keep everything on GPU
    and report the EFFECTIVE compressed KV size assuming 1-bit packed
    storage for Q layers and FP16 for S layers.
  - No dynamic per-step Top-K with critical-channel prefetch. We do a single
    SnapKV-style Top-K at end of prefill (the same shape as C4).
  - 1-bit quantization is simulated by round-trip quantize→dequantize so the
    quality impact is real even though the underlying tensors stay in FP16.
  - HuggingFace's DynamicCache requires a UNIFORM per-layer sequence length,
    which the paper sidesteps with a custom Cache. We therefore prune all
    layers to the same S-budget length (n_local + n_topk) and additionally
    quantize the Q layers to 1-bit. Q layers thus get the full hybrid
    treatment (pruned to top tokens AND 1-bit), and S layers get only the
    pruning. This still exercises the paper's "tailored compression per
    layer" idea, just within the GPU-only single-cache invariant.

Measures:
  Speed  : TTFT (ms incl. compression), TPOT (ms), tokens/sec, peak VRAM (MB),
           compressed KV cache size (MB)
  Quality: WikiText-2 perplexity (teacher-forcing, unaffected by prefix-only
           compression), LongBench accuracy (7 tasks, with TailorKV prefill)

Output: runs/C1/{model}/results.json

Usage:
    python scripts/benchmark_c1_tailorkv.py --model phi3
    python scripts/benchmark_c1_tailorkv.py --model llama3
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
# TailorKV hyperparameters (paper LongBench config)
# ---------------------------------------------------------------------------

# Quantization-friendly layers per model. Paper: Llama-3.1-8B uses Q={0}.
# Phi-3 isn't in the paper; using Q={0} as a sensible default (overridable
# via --q-layers).
TKV_Q_LAYERS: dict[str, list[int]] = {
    "phi3":       [0],
    "llama3":     [0],
    "llama31_8b": [0],
    "llama32_3b": [0],
}

TKV_BITS         = 1     # quantization bit width for Q layers
TKV_GROUP_SIZE   = 64    # quantization group size (paper default)
TKV_N_LOCAL      = 64    # recent tokens kept by S layers (paper LongBench)
TKV_N_TOPK       = 128   # top-K prefix tokens kept by S layers (paper LongBench)
TKV_WINDOW_SIZE  = 32    # last-window queries used for SnapKV scoring
TKV_KERNEL_SIZE  = 7     # avgpool kernel for score smoothing


# ---------------------------------------------------------------------------
# 1-bit quantization (KIVI-style) — round-trip simulation
# ---------------------------------------------------------------------------

def _quantize_dequantize(x: torch.Tensor, bits: int, group_size: int,
                         channel_dim: int) -> torch.Tensor:
    """
    Group-wise asymmetric quantize→dequantize along `channel_dim`.

    K: per-channel (channel_dim = head_dim, last axis). Groups of `group_size`
       elements along the *sequence* axis.
    V: per-token (channel_dim = sequence axis). Groups of `group_size` tokens
       share scale/zero-point.

    Returns a tensor with the same shape and dtype as `x` whose values are
    the quantize→dequantize round trip in `bits`-bit precision.
    """
    orig_dtype = x.dtype
    xf = x.float()

    # Move the group-along axis to the last position for easy reshape.
    if channel_dim < 0:
        channel_dim += xf.ndim
    xf = xf.movedim(channel_dim, -1)
    *lead, n = xf.shape

    # Pad along the group axis to a multiple of group_size.
    pad = (-n) % group_size
    if pad:
        xf = F.pad(xf, (0, pad))
    n_pad = xf.shape[-1]
    g = n_pad // group_size

    xf = xf.reshape(*lead, g, group_size)
    x_min = xf.amin(dim=-1, keepdim=True)
    x_max = xf.amax(dim=-1, keepdim=True)
    levels = (1 << bits) - 1
    scale = (x_max - x_min).clamp(min=1e-8) / levels
    q  = ((xf - x_min) / scale).round().clamp(0, levels)
    xq = q * scale + x_min

    xq = xq.reshape(*lead, n_pad)
    if pad:
        xq = xq[..., :n]
    xq = xq.movedim(-1, channel_dim)
    return xq.to(orig_dtype)


def _quantize_layer_kv(K: torch.Tensor, V: torch.Tensor,
                       bits: int, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    K shape [B, H, S, D]: quantize per-channel (group along S, keep D structure).
    V shape [B, H, S, D]: quantize per-token (group along D, keep S structure).
    """
    Kq = _quantize_dequantize(K, bits=bits, group_size=group_size, channel_dim=2)
    Vq = _quantize_dequantize(V, bits=bits, group_size=group_size, channel_dim=3)
    return Kq, Vq


def _quant_layer_bytes(K: torch.Tensor, V: torch.Tensor,
                       bits: int, group_size: int) -> float:
    """
    Effective storage for a quantized layer:
      data:    n_elements × bits / 8
      scale:   one fp16 per group  (2 bytes)
      zero:    one fp16 per group  (2 bytes)
    Same for K and V (each gets its own scale/zero-point arrays).
    """
    total = 0.0
    for T in (K, V):
        n = T.numel()
        n_groups = n // group_size + (1 if n % group_size else 0)
        total += n * bits / 8.0          # quantized data
        total += n_groups * 2 * 2.0      # scale + zero-point in fp16
    return total


# ---------------------------------------------------------------------------
# SnapKV-style Top-K selection (S layers)
# ---------------------------------------------------------------------------

def _layer_token_scores(attn: torch.Tensor, window: int, kernel: int,
                        k_len: int | None = None) -> torch.Tensor:
    """Mean attention from the last `window` queries, smoothed by avgpool."""
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


def _select_indices(scores: torch.Tensor, n_local: int, n_topk: int) -> torch.Tensor:
    """Keep the last `n_local` tokens plus the top `n_topk` from the prefix."""
    seq_len = scores.shape[0]
    budget  = n_local + n_topk
    if seq_len <= budget:
        return torch.arange(seq_len, device=scores.device)

    n_recent = min(n_local, seq_len)
    n_top    = min(n_topk, seq_len - n_recent)
    if n_top <= 0:
        return torch.arange(seq_len - n_recent, seq_len, device=scores.device)

    prefix_scores = scores.clone()
    prefix_scores[-n_recent:] = float("-inf")
    top_idx  = torch.topk(prefix_scores, n_top, largest=True).indices
    recent   = torch.arange(seq_len - n_recent, seq_len, device=scores.device)
    keep     = torch.cat([top_idx, recent])
    return keep.sort().values


# ---------------------------------------------------------------------------
# Hybrid compression — applied after a two-pass prefill
# ---------------------------------------------------------------------------

def _compress_cache(
    past: DynamicCache,
    attentions: tuple,
    q_layers: set[int],
    bits: int,
    group_size: int,
    n_local: int,
    n_topk: int,
    window: int,
    kernel: int,
) -> tuple[DynamicCache, dict]:
    """
    Build a new DynamicCache with uniform per-layer length:
      - Every layer is pruned to (n_local + n_topk) via SnapKV Top-K.
      - Q layers are ADDITIONALLY quantized→dequantized at `bits` bits
        (per-channel for K, per-token for V).
    Returns the new cache and per-layer stats (effective bytes, kept length).
    """
    new_cache = DynamicCache()
    layer_bytes: list[float] = []
    layer_kept:  list[int]   = []
    layer_kind:  list[str]   = []

    n_layers = len(attentions)
    for L in range(n_layers):
        K = past.layers[L].keys     # [B, n_kv_heads, S, D]
        V = past.layers[L].values
        attn = attentions[L]
        k_len = K.shape[2]

        scores = _layer_token_scores(attn, window=window, kernel=kernel, k_len=k_len)
        idx    = _select_indices(scores, n_local=n_local, n_topk=n_topk)
        Kc = K.index_select(dim=2, index=idx)
        Vc = V.index_select(dim=2, index=idx)
        kept_len = idx.shape[0]

        if L in q_layers:
            Kc, Vc = _quantize_layer_kv(Kc, Vc, bits=bits, group_size=group_size)
            new_cache.update(Kc, Vc, layer_idx=L)
            layer_bytes.append(_quant_layer_bytes(Kc, Vc, bits=bits, group_size=group_size))
            layer_kind.append("Q")
        else:
            new_cache.update(Kc, Vc, layer_idx=L)
            layer_bytes.append(
                Kc.numel() * Kc.element_size() + Vc.numel() * Vc.element_size()
            )
            layer_kind.append("S")

        layer_kept.append(kept_len)

    stats = {
        "compressed_kv_mb": sum(layer_bytes) / (1024 ** 2),
        "avg_kept":         sum(layer_kept)  / len(layer_kept),
        "n_q_layers":       sum(1 for k in layer_kind if k == "Q"),
        "n_s_layers":       sum(1 for k in layer_kind if k == "S"),
    }
    return new_cache, stats


def _prefill_and_compress(
    model, ids_t: torch.Tensor, q_layers: set[int], device: str,
) -> tuple[DynamicCache, torch.Tensor, dict]:
    """
    Two-pass prefill:
      Pass 1 (SDPA): build the prefix KV cache cheaply.
      Pass 2 (eager, output_attentions=True): forward only the last
              `TKV_WINDOW_SIZE` tokens with that prefix as past_key_values
              — gives us the SnapKV scoring matrix without the O(N^2) blow-up.

    Then apply hybrid compression (quantize Q layers, prune S layers).

    Returns (compressed_cache, last_logits, stats).
    """
    seq_len = ids_t.shape[1]
    w = min(TKV_WINDOW_SIZE, seq_len)

    if seq_len <= w:
        model.set_attn_implementation("eager")
        with torch.no_grad():
            out = model(ids_t, use_cache=True, output_attentions=True)
        full_mb = kv_cache_mb(out.past_key_values)
        past, st = _compress_cache(
            out.past_key_values, out.attentions,
            q_layers=q_layers, bits=TKV_BITS, group_size=TKV_GROUP_SIZE,
            n_local=TKV_N_LOCAL, n_topk=TKV_N_TOPK,
            window=TKV_WINDOW_SIZE, kernel=TKV_KERNEL_SIZE,
        )
        st["full_kv_mb"] = full_mb
        st["seq_len"]    = seq_len
        return past, out.logits[:, -1, :], st

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
    past, st = _compress_cache(
        out.past_key_values, out.attentions,
        q_layers=q_layers, bits=TKV_BITS, group_size=TKV_GROUP_SIZE,
        n_local=TKV_N_LOCAL, n_topk=TKV_N_TOPK,
        window=TKV_WINDOW_SIZE, kernel=TKV_KERNEL_SIZE,
    )
    st["full_kv_mb"] = full_mb
    st["seq_len"]    = seq_len
    return past, out.logits[:, -1, :], st


# ---------------------------------------------------------------------------
# Phase 1: Speed
# ---------------------------------------------------------------------------

def run_speed(model, tokenizer, model_key: str, q_layers: set[int],
              device: str) -> dict:
    print("\n=== Speed benchmark (TailorKV) ===")
    target_len = LB_MAX_INPUT[model_key]
    all_ids = tokenizer.encode(SPEED_TEXT, add_special_tokens=False)
    if len(all_ids) < target_len:
        raise RuntimeError(
            f"SPEED_TEXT only has {len(all_ids)} tokens; need {target_len}. "
            "Increase the repetition factor in scripts/_common.py."
        )
    prompt_ids = all_ids[:target_len]
    input_ids  = torch.tensor([prompt_ids], device=device)
    print(f"  Prompt: {len(prompt_ids)} tokens   "
          f"Q-layers: {sorted(q_layers)}   "
          f"S-budget: {TKV_N_LOCAL}+({TKV_N_TOPK})  bits: {TKV_BITS}")

    with torch.no_grad():
        _ = model(input_ids[:, :32], use_cache=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    past, last_logits, stats = _prefill_and_compress(
        model, input_ids, q_layers=q_layers, device=device,
    )
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1000
    print(f"  TTFT (incl. compression): {ttft_ms:.1f} ms")

    full_kv_mb = stats["full_kv_mb"]
    comp_kv_mb = stats["compressed_kv_mb"]
    ratio      = full_kv_mb / comp_kv_mb if comp_kv_mb > 0 else 1.0
    print(f"  Full KV: {full_kv_mb:.1f} MB → compressed: {comp_kv_mb:.1f} MB "
          f"({ratio:.2f}× compression)")
    print(f"  Q-layers: {stats['n_q_layers']} (1-bit), "
          f"S-layers: {stats['n_s_layers']} ({TKV_N_LOCAL}+{TKV_N_TOPK} tokens)")

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
        "kv_cache_mb":          round(comp_kv_mb, 2),
        "kv_compression_ratio": round(ratio, 3),
        "speed_prompt_tokens":  len(prompt_ids),
        "speed_n_decode":       SPEED_N_DECODE,
    }


# ---------------------------------------------------------------------------
# Phase 3: LongBench
# ---------------------------------------------------------------------------

def _generate_tkv(
    model, tokenizer, prompt: str, max_new: int, max_input: int,
    device: str, use_chat: bool, q_layers: set[int],
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
        model, ids_t, q_layers=q_layers, device=device,
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


def run_longbench(model, tokenizer, model_key: str, q_layers: set[int],
                  device: str, pp_logger=None) -> dict:
    print("\n=== LongBench (TailorKV) ===")
    max_input = LB_MAX_INPUT[model_key]
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
            pred, st = _generate_tkv(model, tokenizer, prompt, max_new, max_input,
                                     device, use_chat=cfg["chat"], q_layers=q_layers)
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
        avg_full  = sum(full_mbs) / len(full_mbs)
        avg_comp  = sum(comp_mbs) / len(comp_mbs)
        avg_ratio = avg_full / avg_comp if avg_comp > 0 else 1.0
        avg_seq   = sum(seq_lens) / len(seq_lens)
        avg_kept  = sum(kept_list) / len(kept_list)
        print(f"\n  LongBench compression: full {avg_full:.1f} MB → "
              f"compressed {avg_comp:.1f} MB ({avg_ratio:.2f}×, "
              f"avg seq {avg_seq:.0f} → {avg_kept:.0f} avg kept/layer)")
        results["_compression"] = {
            "avg_seq_len":           round(avg_seq, 1),
            "avg_kept_per_layer":    round(avg_kept, 1),
            "avg_kv_cache_mb_full":  round(avg_full, 2),
            "avg_kv_cache_mb":       round(avg_comp, 2),
            "avg_compression_ratio": round(avg_ratio, 3),
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",     required=True, choices=list(MODELS.keys()))
    p.add_argument("--output",    default="runs/C1")
    p.add_argument("--q-layers",  type=str, default=None,
                   help="comma-separated layer indices to quantize "
                        "(default: paper preset per model)")
    p.add_argument("--bits",      type=int, default=None,
                   help="quantization bit width for Q layers (default: 1)")
    p.add_argument("--skip-speed-ppl", action="store_true",
                   help="skip speed + perplexity phases; preserve old fields from existing results.json")
    return p.parse_args()


def main():
    args     = parse_args()
    model_id = MODELS[args.model]
    out_dir  = Path(args.output) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.q_layers is not None:
        TKV_Q_LAYERS[args.model] = [int(x) for x in args.q_layers.split(",") if x.strip()]
    if args.bits is not None:
        global TKV_BITS
        TKV_BITS = args.bits
    q_set = set(TKV_Q_LAYERS[args.model])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nConfig : C1 (TailorKV — hybrid: {TKV_BITS}-bit quant on Q-layers + "
          f"SnapKV Top-K on S-layers)")
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

    results = {
        "config":        "C1",
        "method":        f"TailorKV (hybrid: {TKV_BITS}-bit quant on Q-layers + "
                         f"SnapKV {TKV_N_LOCAL}+{TKV_N_TOPK} on S-layers)",
        "model":         args.model,
        "model_id":      model_id,
        "q_layers":      sorted(q_set),
        "bits":          TKV_BITS,
        "group_size":    TKV_GROUP_SIZE,
        "n_local":       TKV_N_LOCAL,
        "n_topk":        TKV_N_TOPK,
        "window_size":   TKV_WINDOW_SIZE,
        "kernel_size":   TKV_KERNEL_SIZE,
    }

    if not args.skip_speed_ppl:
        results.update(run_speed(model, tokenizer, args.model, q_set, device))
        torch.cuda.empty_cache()

        model.set_attn_implementation("sdpa")
        results.update(run_ppl(model, tokenizer, args.model, device))
        torch.cuda.empty_cache()
        model.set_attn_implementation("eager")

    lb = run_longbench(model, tokenizer, args.model, q_set, device,
                       pp_logger=PerPromptLogger(out_dir / "per_prompt.jsonl",
                                                 config="C1", model=args.model))
    results["longbench"] = lb
    task_scores = [v["score"] for k, v in lb.items() if not k.startswith("_")]
    if task_scores:
        results["longbench_avg"] = round(sum(task_scores) / len(task_scores), 4)
    if "_compression" in lb:
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
