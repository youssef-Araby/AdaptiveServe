#!/usr/bin/env python3
"""
C3 benchmark — KVQuant: per-channel K + per-token V low-bit quantization
with Dense-and-Sparse outlier preservation.

KVQuant (Hooper et al., 2024). arXiv:2401.18079

Algorithm (this script, simplified-but-faithful):
  - K cache:  per-CHANNEL asymmetric uniform quantization. Channels are the
              head_dim axis; groups of `group_size` tokens along the seq
              axis share scale/zero-point. (Paper §3.1: K has large outlier
              channels — per-channel scales handle them better than
              per-token.)
  - V cache:  per-TOKEN  asymmetric uniform quantization. Each token gets
              its own scale/zero-point along head_dim with `group_size`
              groups. (Paper §3.2: V is well-behaved per-token.)
  - Dense-and-Sparse (§3.3): the top `outlier_frac` (1 % by default) of
              elements by magnitude are identified FIRST and excluded from
              the quantization min/max ranges, so the remaining ~99 % are
              quantized on a tighter grid. The outliers themselves are
              preserved in FP16, and the effective-byte count adds index +
              FP16 storage for them.

Simplifications versus the paper:
  - Uniform asymmetric quantization rather than the paper's non-uniform
    NUQ codebook. NUQ gives ≈0.3 PPL improvement at 3-bit; at 4-bit (our
    default) the gap is negligible.
  - We operate on POST-RoPE keys (the cache hands them to us already
    rotated). The paper recommends pre-RoPE quantization for K which buys
    another ~0.1 PPL but requires a custom attention path. Skipped.
  - Quantization is simulated by round-trip quantize→dequantize. The
    underlying tensors stay fp16; only the *effective* compressed bytes
    are reported.
  - Only the prefix is quantized; tokens generated during decode are
    appended to the cache at full FP16. This matches every other prefix-
    only method in the suite (C1, C2, C4) and keeps decode arithmetic
    unchanged. Per-prompt byte accounting charges those decode tokens at
    FP16 on both the effective (kv_bytes) and the FP16-reference
    (kv_bytes_fp16) side.

Measures:
  Speed  : TTFT (ms incl. quantization), TPOT (ms), tokens/sec, peak VRAM,
           compressed KV cache size (MB)
  Quality: WikiText-2 perplexity (teacher-forcing, unaffected by prefix-only
           compression), LongBench accuracy (7 tasks, with quantized prefix)

Output: runs/C3/{model}/results.json

Usage:
    python scripts/benchmark_c3_kvquant.py --model phi3
    python scripts/benchmark_c3_kvquant.py --model llama3
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
    fp16_decode_growth_bytes,
    kv_cache_bytes,
    load_longbench_task,
    postprocess_pred,
    run_ppl,
)

# ---------------------------------------------------------------------------
# KVQuant hyperparameters (paper main config: 4-bit, 1 % outliers, g=128)
# ---------------------------------------------------------------------------

KVQ_BITS         = 4      # quantization bit width for non-outlier elements
KVQ_GROUP_SIZE   = 128    # group size along the per-{channel,token} axis
KVQ_OUTLIER_FRAC = 0.01   # fraction of |x| outliers kept in FP16


# ---------------------------------------------------------------------------
# Group-wise asymmetric quantize → dequantize (round-trip simulation)
# ---------------------------------------------------------------------------

def _quantize_dequantize(x: torch.Tensor, bits: int, group_size: int,
                         channel_dim: int,
                         exclude_mask: torch.Tensor | None = None
                         ) -> torch.Tensor:
    """
    Group-wise asymmetric quantize→dequantize along `channel_dim`.

    Same primitive as C1: moves channel_dim to last, pads to group_size,
    asymmetric min/max → quantize → dequantize.

    `exclude_mask` (optional, bool, same shape as `x`): elements marked True
    (the FP16-preserved outliers) are EXCLUDED from each group's min/max
    range so the remaining elements are quantized on a tighter grid
    (Dense-and-Sparse done right). Padding is excluded from the range too.
    The round-tripped values at excluded positions are meaningless — the
    caller must overwrite them with the FP16 originals.
    """
    orig_dtype = x.dtype
    xf = x.float()
    if channel_dim < 0:
        channel_dim += xf.ndim
    xf = xf.movedim(channel_dim, -1)
    *lead, n = xf.shape

    pad = (-n) % group_size
    if pad:
        xf = F.pad(xf, (0, pad))
    n_pad = xf.shape[-1]
    g = n_pad // group_size

    xf = xf.reshape(*lead, g, group_size)
    if exclude_mask is None:
        x_min = xf.amin(dim=-1, keepdim=True)
        x_max = xf.amax(dim=-1, keepdim=True)
    else:
        excl = exclude_mask.movedim(channel_dim, -1)
        if pad:
            excl = F.pad(excl, (0, pad), value=True)  # pad never widens the range
        excl = excl.reshape(*lead, g, group_size)
        big  = torch.finfo(torch.float32).max
        x_min = xf.masked_fill(excl, big).amin(dim=-1, keepdim=True)
        x_max = xf.masked_fill(excl, -big).amax(dim=-1, keepdim=True)
        # Groups whose real elements are ALL outliers have an empty range;
        # zero it out (their positions are restored to FP16 by the caller).
        empty = x_min > x_max
        x_min = torch.where(empty, torch.zeros_like(x_min), x_min)
        x_max = torch.where(empty, torch.zeros_like(x_max), x_max)
    levels = (1 << bits) - 1
    scale  = (x_max - x_min).clamp(min=1e-8) / levels
    q  = ((xf - x_min) / scale).round().clamp(0, levels)
    xq = q * scale + x_min

    xq = xq.reshape(*lead, n_pad)
    if pad:
        xq = xq[..., :n]
    xq = xq.movedim(-1, channel_dim)
    return xq.to(orig_dtype)


# ---------------------------------------------------------------------------
# Dense-and-Sparse: top-|x| outliers, kept in FP16 and excluded from ranges
# ---------------------------------------------------------------------------

def _outlier_mask(x: torch.Tensor, frac: float
                  ) -> tuple[torch.Tensor | None, int]:
    """
    Boolean mask (same shape as `x`) of the top `frac` |x| elements — the
    Dense-and-Sparse FP16 outlier set. Returns (mask_or_None, n_outliers).
    """
    if frac <= 0:
        return None, 0
    n_total = x.numel()
    n_out   = int(round(n_total * frac))
    if n_out <= 0:
        return None, 0

    flat_abs = x.detach().abs().reshape(-1)
    # topk on n_total elements is fine — fast on GPU.
    idx  = torch.topk(flat_abs, n_out, largest=True, sorted=False).indices
    mask = torch.zeros(n_total, dtype=torch.bool, device=x.device)
    mask[idx] = True
    return mask.reshape(x.shape), n_out


# ---------------------------------------------------------------------------
# Per-layer KVQuant transform + effective-byte accounting
# ---------------------------------------------------------------------------

def _kvquant_layer(K: torch.Tensor, V: torch.Tensor,
                   bits: int, group_size: int, outlier_frac: float
                   ) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    K [B, H, S, D]:   per-channel quantization (groups along S, dim=2)
    V [B, H, S, D]:   per-token   quantization (groups along D, dim=3)
    Outliers (|x|): top `outlier_frac` of K and V each are identified FIRST,
    excluded from the quantization min/max ranges (so the remaining ~99 %
    use a tighter grid), and kept at FP16.

    Returns (K_mixed, V_mixed, effective_bytes).
    """
    mask_k, n_out_k = _outlier_mask(K, outlier_frac)
    mask_v, n_out_v = _outlier_mask(V, outlier_frac)

    Kq = _quantize_dequantize(K, bits=bits, group_size=group_size,
                              channel_dim=2, exclude_mask=mask_k)
    Vq = _quantize_dequantize(V, bits=bits, group_size=group_size,
                              channel_dim=3, exclude_mask=mask_v)

    Km = torch.where(mask_k, K, Kq) if mask_k is not None else Kq
    Vm = torch.where(mask_v, V, Vq) if mask_v is not None else Vq

    # Effective storage:
    #   - quantized data: (n - n_outliers) × bits / 8
    #   - scale + zero (fp16) per group: 4 bytes per group, with groups
    #     formed per vector along the quantization axis (ceil per vector —
    #     matches what _quantize_dequantize physically forms; K groups along
    #     S, V groups along head_dim, which pads e.g. Phi-3's D=96 to one
    #     full group per (head, token) vector)
    #   - outliers:        n_outliers × (fp16 value + int32 index) = 6 B
    bytes_total = 0.0
    for T, n_out, axis_len in ((K, n_out_k, K.shape[2]), (V, n_out_v, V.shape[3])):
        n        = T.numel()
        n_vecs   = n // axis_len
        n_groups = n_vecs * ((axis_len + group_size - 1) // group_size)
        bytes_total += (n - n_out) * bits / 8.0   # quantized data
        bytes_total += n_groups * 4.0             # fp16 scale + fp16 zero
        bytes_total += n_out * 6.0                # fp16 value + int32 index
    return Km, Vm, bytes_total


def _quantize_cache(past: DynamicCache, bits: int, group_size: int,
                    outlier_frac: float) -> tuple[DynamicCache, dict]:
    """In-place rewrite of every layer's K/V; returns new cache and stats."""
    new_cache = DynamicCache()
    total_bytes = 0.0
    n_layers = len(past.layers)
    for L in range(n_layers):
        K = past.layers[L].keys
        V = past.layers[L].values
        Km, Vm, b = _kvquant_layer(K, V, bits=bits, group_size=group_size,
                                    outlier_frac=outlier_frac)
        new_cache.update(Km, Vm, layer_idx=L)
        total_bytes += b

    stats = {
        "compressed_kv_bytes": total_bytes,
        "compressed_kv_mb":    total_bytes / (1024 ** 2),
        "n_layers":            n_layers,
    }
    return new_cache, stats


def _fp16_bytes_per_token(model) -> float:
    """FP16 KV bytes per token: 2 B × 2 (K&V) × n_layers × n_kv_heads × head_dim."""
    cfg      = model.config
    n_kv     = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return 2.0 * 2.0 * cfg.num_hidden_layers * n_kv * head_dim


def _prefill_and_quantize(model, ids_t: torch.Tensor, device: str
                          ) -> tuple[DynamicCache, torch.Tensor, dict]:
    """Single-pass SDPA prefill, then quantize the resulting cache."""
    with torch.no_grad():
        out = model(ids_t, use_cache=True)
    full_bytes = kv_cache_bytes(out.past_key_values)  # measured pre-compression
    past, st = _quantize_cache(out.past_key_values,
                               bits=KVQ_BITS, group_size=KVQ_GROUP_SIZE,
                               outlier_frac=KVQ_OUTLIER_FRAC)
    st["full_kv_bytes"] = float(full_bytes)
    st["full_kv_mb"]    = full_bytes / (1024 ** 2)
    st["seq_len"]       = ids_t.shape[1]
    return past, out.logits[:, -1, :], st


# ---------------------------------------------------------------------------
# Phase 1: Speed
# ---------------------------------------------------------------------------

def run_speed(model, tokenizer, model_key: str, device: str) -> dict:
    print("\n=== Speed benchmark (KVQuant) ===")
    target_len = LB_MAX_INPUT[model_key]
    all_ids = tokenizer.encode(SPEED_TEXT, add_special_tokens=False)
    if len(all_ids) < target_len:
        raise RuntimeError(
            f"SPEED_TEXT only has {len(all_ids)} tokens; need {target_len}."
        )
    prompt_ids = all_ids[:target_len]
    input_ids  = torch.tensor([prompt_ids], device=device)
    print(f"  Prompt: {len(prompt_ids)} tokens   "
          f"bits: {KVQ_BITS}  group: {KVQ_GROUP_SIZE}  "
          f"outliers: {KVQ_OUTLIER_FRAC*100:.1f}%")

    with torch.no_grad():
        _ = model(input_ids[:, :32], use_cache=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    past, last_logits, stats = _prefill_and_quantize(model, input_ids, device)
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1000
    print(f"  TTFT (incl. quantization): {ttft_ms:.1f} ms")

    full_kv_mb = stats["full_kv_mb"]
    comp_kv_mb = stats["compressed_kv_mb"]
    ratio      = full_kv_mb / comp_kv_mb if comp_kv_mb > 0 else 1.0
    print(f"  Full KV: {full_kv_mb:.1f} MB → compressed: {comp_kv_mb:.1f} MB "
          f"({ratio:.2f}× compression, {stats['n_layers']} layers)")

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

def _generate_kvq(model, tokenizer, prompt: str, max_new: int, max_input: int,
                  device: str, use_chat: bool) -> tuple[str, dict]:
    if use_chat:
        prompt = apply_chat_template(tokenizer, prompt)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > max_input:
        half = max_input // 2
        ids  = ids[:half] + ids[-half:]
    ids_t = torch.tensor([ids], device=device)
    logical_len = ids_t.shape[1]

    eos_id = tokenizer.eos_token_id
    past, last_logits, stats = _prefill_and_quantize(model, ids_t, device)
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

    # Every generated token was fed through the model, so each appended one
    # FP16 K/V entry per layer to the cache.
    stats["n_generated"] = len(generated)
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), stats


def run_longbench(model, tokenizer, model_key: str, device: str,
                  pp_logger=None) -> dict:
    print("\n=== LongBench (KVQuant) ===")
    max_input = LB_MAX_INPUT[model_key]
    results   = {}
    full_mbs, comp_mbs, seq_lens = [], [], []
    bpt_fp16  = _fp16_bytes_per_token(model)

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
            pred_raw, st = _generate_kvq(model, tokenizer, prompt, max_new,
                                         max_input, device, use_chat=cfg["chat"])
            # Per-prompt bytes: prefill measured before/after quantization.
            # Decode tokens append at FP16 to the rebuilt plain cache (really
            # grows) and are charged FP16 in kv_bytes; the FP16 reference is
            # capped for sliding-window models (Phi-3 FP16 serving evicts as
            # it appends once at cap, gaining nothing per decode token).
            decode_bytes  = st["n_generated"] * bpt_fp16
            kv_bytes_fp16 = st["full_kv_bytes"] + fp16_decode_growth_bytes(
                st["full_kv_bytes"], st["n_generated"], bpt_fp16, model.config)
            kv_bytes      = st["compressed_kv_bytes"] + decode_bytes
            compression   = kv_bytes_fp16 / kv_bytes if kv_bytes > 0 else 1.0
            full_mbs.append(kv_bytes_fp16 / (1024 ** 2))
            comp_mbs.append(kv_bytes / (1024 ** 2))
            seq_lens.append(st["seq_len"])
            pred = postprocess_pred(task, pred_raw)
            s = compute_score(metric, pred, golds,
                              all_classes=sample.get("all_classes"))
            scores.append(s)
            if pp_logger is not None:
                pp_logger.log(task=task, sample_idx=j, metric=metric, score=s,
                              features=extract_prompt_features(prompt, tokenizer),
                              compression=compression, pred=pred,
                              kv_bytes=kv_bytes, kv_bytes_fp16=kv_bytes_fp16,
                              extra={"n_generated": st["n_generated"]})
            if (j + 1) % 5 == 0:
                print(f"  {task} [{j+1}/{len(samples)}]  {metric}={sum(scores)/len(scores):.4f}")
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"  {task}  FINAL {metric}={avg:.4f}  (n={len(scores)})")
        results[task] = {"metric": metric, "score": round(avg, 4), "n": len(scores)}

    if full_mbs:
        # Run-level aggregate on the same basis as the per-prompt logs:
        # prefill (quantized+metadata vs FP16) + FP16 decode tokens both sides.
        avg_full  = sum(full_mbs) / len(full_mbs)
        avg_comp  = sum(comp_mbs) / len(comp_mbs)
        avg_ratio = avg_full / avg_comp if avg_comp > 0 else 1.0
        avg_seq   = sum(seq_lens) / len(seq_lens)
        print(f"\n  LongBench compression: full {avg_full:.1f} MB → "
              f"compressed {avg_comp:.1f} MB ({avg_ratio:.2f}×, "
              f"avg seq {avg_seq:.0f})")
        results["_compression"] = {
            "avg_seq_len":           round(avg_seq, 1),
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
    p.add_argument("--model",   required=True, choices=list(MODELS.keys()))
    p.add_argument("--output",  default="runs/C3")
    p.add_argument("--bits",    type=int,   default=None,
                   help=f"quantization bit width (default: {KVQ_BITS})")
    p.add_argument("--outliers", type=float, default=None,
                   help=f"FP16 outlier fraction (default: {KVQ_OUTLIER_FRAC})")
    p.add_argument("--skip-speed-ppl", action="store_true",
                   help="skip speed + perplexity phases; preserve old fields from existing results.json")
    return p.parse_args()


def main():
    args     = parse_args()
    model_id = MODELS[args.model]
    out_dir  = Path(args.output) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    global KVQ_BITS, KVQ_OUTLIER_FRAC
    if args.bits is not None:
        KVQ_BITS = args.bits
    if args.outliers is not None:
        KVQ_OUTLIER_FRAC = args.outliers

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nConfig : C3 (KVQuant — {KVQ_BITS}-bit per-channel K + per-token V, "
          f"{KVQ_OUTLIER_FRAC*100:.1f}% FP16 outliers)")
    print(f"Model  : {model_id}")
    print(f"Device : {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    print(f"Output : {out_dir}/results.json")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    results = {
        "config":         "C3",
        "method":         f"KVQuant ({KVQ_BITS}-bit per-channel K + per-token V, "
                          f"{KVQ_OUTLIER_FRAC*100:.1f}% FP16 outliers)",
        "model":          args.model,
        "model_id":       model_id,
        "bits":           KVQ_BITS,
        "group_size":     KVQ_GROUP_SIZE,
        "outlier_frac":   KVQ_OUTLIER_FRAC,
    }

    if not args.skip_speed_ppl:
        results.update(run_speed(model, tokenizer, args.model, device))
        torch.cuda.empty_cache()

        results.update(run_ppl(model, tokenizer, args.model, device))
        torch.cuda.empty_cache()

    lb = run_longbench(model, tokenizer, args.model, device,
                       pp_logger=PerPromptLogger(out_dir / "per_prompt.jsonl",
                                                 config="C3", model=args.model))
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
