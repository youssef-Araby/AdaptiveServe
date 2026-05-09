#!/usr/bin/env python3
"""
C2 benchmark — Full QAQ KV cache quantization (attention-aware, variable-bit).

QAQ: Quality Adaptive Quantization for LLM KV Cache (Dong et al., 2024)
  arXiv: 2403.04643  |  github.com/ClubieDong/QAQ-KVCacheQuantization

This implements the FULL QAQ algorithm (not the fixed-bit ablation):
  - Separate quantization strategies for Key cache vs Value cache
  - Key cache: bits derived from query-norm bound (Section 4.1, Eq. key formula)
      q_norm precomputed as upper-10th-percentile of ||Q||^2 across layers/heads
      σ_t(K) ≤ sqrt(12 / q_norm * log(T^3/(T-1) * σ_max^2 + 1))
      B_t(K) = ceil(log2((K_max - K_min) / (2*σ_t(K)) + 1))
  - Value cache: bits inversely proportional to attention score (Section 4.1, Eq. value)
      σ_t(V) ≤ sqrt(12/T) * σ_max / attention_t
      B_t(V) = ceil(log2((V_max - V_min) / (2*σ_t(V)) + 1))
  - Attention window n=5: use max attention over last 5 steps (avoids over-quantizing
    tokens that will become important)
  - 1% outliers kept at FP16 (mixed precision)
  - Asymmetric (min-max) uniform quantization, head-level granularity
  - Bits clamped to [n_bits_min=2, n_bits_max=16]

Note: TPOT will be slower than C0 due to Python simulation. Real hardware with
packed 2-4 bit storage would achieve significant DRAM-bandwidth speedup.

Measures:
  Speed  : TTFT (ms), TPOT (ms), tokens/sec, peak VRAM (MB), theoretical KV MB
  Quality: WikiText-2 perplexity (teacher-forcing), LongBench (7 tasks, QAQ generation)

Output: runs/C2/{model}/results.json

Usage:
    python scripts/benchmark_c2_qaq.py --model phi3
    python scripts/benchmark_c2_qaq.py --model llama3
"""

from __future__ import annotations

import argparse
import json
import math
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
    compute_score,
    extract_prompt_features,
    load_longbench_task,
    run_ppl,
)

# ---------------------------------------------------------------------------
# Per-model attention-implementation config (QAQ-specific)
# ---------------------------------------------------------------------------
#   attn_impl        : attention implementation for loading
#                      ("eager" = needed for output_attentions; "sdpa" = flash/fast)
#   attn_aware_decode: True  → capture attention per decode step and compute attention-aware
#                              V bits (requires eager + output_attentions, 1 tok = O(seq_len))
#                      False → use K formula for V bits in decode too (SDPA, faster)
#                      Phi-3-mini's eager attention is ~43× slower than SDPA, so we use SDPA.
MODELS_CFG: dict[str, dict] = {
    "llama3": {
        "id":               MODELS["llama3"],
        "attn_impl":        "eager",   # needed for decode output_attentions
        "attn_aware_decode": True,
    },
    "phi3": {
        "id":               MODELS["phi3"],
        "attn_impl":        "sdpa",    # phi3 eager is 43× slower than SDPA
        "attn_aware_decode": False,
    },
}

# QAQ config (full attention-aware, variable-bit)
QAQ_OUTLIER_RATIO      = 0.01   # 1% outliers kept at FP16
QAQ_N_BITS_MIN         = 2      # minimum quantization bits
QAQ_N_BITS_MAX         = 16     # maximum (= FP16, effectively no quantization)
QAQ_LAST_N_ATTENTIONS  = 5      # attention window size (paper: n=5)
QAQ_TARGET_ERROR       = 1.0    # σ_max — target quantization error (paper default)
QAQ_Q_NORM_PERCENTILE  = 90     # use 90th-percentile of ||Q||^2 across heads/layers


# ---------------------------------------------------------------------------
# QAQ quantization core — full attention-aware variable-bit implementation
# ---------------------------------------------------------------------------

def _qaq_outlier_mask(x: torch.Tensor, outlier_ratio: float) -> torch.Tensor:
    """
    Compute per-head-dim outlier mask.
    x: (*, head_dim)   — last dim is quantized over.
    Returns bool mask, True = outlier (keep at FP16).
    """
    head_dim = x.shape[-1]
    if outlier_ratio <= 0.0 or head_dim <= 2:
        return torch.zeros_like(x, dtype=torch.bool)
    lo_k = max(1, int(math.ceil(outlier_ratio / 2 * head_dim)))
    hi_k = max(lo_k + 1, min(head_dim, int(math.floor((1.0 - outlier_ratio / 2) * head_dim))))
    hi_k = min(hi_k, head_dim)
    lower = torch.kthvalue(x, k=lo_k,  dim=-1, keepdim=True).values
    upper = torch.kthvalue(x, k=hi_k, dim=-1, keepdim=True).values
    return (x < lower) | (x > upper)


def _qaq_uniform_quantize_vec(x: torch.Tensor, n_bits: int,
                               mask: torch.Tensor) -> torch.Tensor:
    """
    Asymmetric uniform quantization of x (excluding outliers in mask).
    x, mask: (*, head_dim)
    Returns dequantized tensor at same dtype.
    """
    x_q = x.masked_fill(mask, 0.0)           # ignore outliers for range calc
    # Use masked min/max
    x_min = x.clone()
    x_min[mask] = float('inf')
    x_max = x.clone()
    x_max[mask] = float('-inf')
    x_min = x_min.amin(dim=-1, keepdim=True)
    x_max = x_max.amax(dim=-1, keepdim=True)
    scale = (x_max - x_min).clamp(min=1e-8) / (2 ** n_bits - 1)
    quant   = torch.round((x - x_min) / scale).clamp(0, 2 ** n_bits - 1)
    dequant = quant * scale + x_min
    return torch.where(mask, x, dequant)      # restore outliers


def _qaq_calc_bits_value(
    attn_score: torch.Tensor,   # (batch, n_heads, seq_len)  — attention weight per token
    v_tensor: torch.Tensor,     # (batch, n_heads, seq_len, head_dim)
    outlier_ratio: float,
    target_error: float,
    n_bits_min: int,
    n_bits_max: int,
) -> torch.Tensor:
    """
    Per-token per-head bit allocation for VALUE cache.

    Paper formula (Section 4.1, matches official source quantizer.py):
        σ_t(V) ≤ sqrt(12/T) * σ_max / S_t
        B_t(V) = ceil(log2((V_max - V_min) / (2*σ_t(V)) + 1))

    Returns: (batch, n_heads, seq_len) int64 tensor of bits per token-head.
    """
    T = v_tensor.shape[2]   # seq_len
    # σ_t(V) upper bound per token: inversely proportional to attention score
    # clip attention to avoid div by zero; floor at 1e-6
    s_t = attn_score.clamp(min=1e-6)   # (batch, n_heads, seq_len)
    sigma_v = math.sqrt(12.0 / T) * target_error / s_t  # (batch, n_heads, seq_len)

    # Compute range (V_max - V_min) per head-token, excluding outliers
    x = v_tensor.float()   # (batch, n_heads, seq_len, head_dim)
    head_dim = x.shape[-1]
    if outlier_ratio > 0.0 and head_dim > 2:
        lo_k = max(1, int(math.ceil(outlier_ratio / 2 * head_dim)))
        hi_k = max(lo_k + 1, min(head_dim, int(math.floor((1.0 - outlier_ratio / 2) * head_dim))))
        lower = torch.kthvalue(x, k=lo_k, dim=-1).values   # (batch, n_heads, seq_len)
        upper = torch.kthvalue(x, k=hi_k, dim=-1).values
        range_v = (upper - lower).clamp(min=1e-8)
    else:
        range_v = (x.amax(dim=-1) - x.amin(dim=-1)).clamp(min=1e-8)

    # B = ceil(log2(range / (2*sigma) + 1))    [matches QAQ source: quantizer.py L_calc_quantization_bits]
    n_bits_float = torch.log2(range_v / (2.0 * sigma_v + 1e-30) + 1.0)
    n_bits = torch.ceil(n_bits_float).to(torch.int64)
    return n_bits.clamp(n_bits_min, n_bits_max)


def _qaq_calc_bits_key(
    q_norm: float,              # precomputed: upper-10th-pct of ||Q||^2
    k_tensor: torch.Tensor,     # (batch, n_heads, seq_len, head_dim)
    outlier_ratio: float,
    target_error: float,
    n_bits_min: int,
    n_bits_max: int,
) -> torch.Tensor:
    """
    Per-token per-head bit allocation for KEY cache.

    Paper formula (Section 4.1, matches official source quantizer.py):
        σ_t(K) ≤ sqrt(12 / ||Q||^2 * log(T^3/(T-1) * σ_max^2 + 1))
        B_t(K) = ceil(log2((K_max - K_min) / (2*σ_t(K)) + 1))

    σ_t(K) is the same for all tokens in a given context (depends on seq_len and q_norm).
    Returns: (batch, n_heads, seq_len) int64 tensor.
    """
    T = k_tensor.shape[2]
    sigma_k_sq = (12.0 / q_norm) * math.log(
        max(T**3 / max(T - 1, 1) * target_error**2 + 1, 1 + 1e-30)
    )
    sigma_k = math.sqrt(max(sigma_k_sq, 1e-30))  # scalar, same for all tokens

    x = k_tensor.float()
    head_dim = x.shape[-1]
    if outlier_ratio > 0.0 and head_dim > 2:
        lo_k = max(1, int(math.ceil(outlier_ratio / 2 * head_dim)))
        hi_k = max(lo_k + 1, min(head_dim, int(math.floor((1.0 - outlier_ratio / 2) * head_dim))))
        lower = torch.kthvalue(x, k=lo_k, dim=-1).values
        upper = torch.kthvalue(x, k=hi_k, dim=-1).values
        range_k = (upper - lower).clamp(min=1e-8)
    else:
        range_k = (x.amax(dim=-1) - x.amin(dim=-1)).clamp(min=1e-8)

    n_bits_float = torch.log2(range_k / (2.0 * sigma_k + 1e-30) + 1.0)
    n_bits = torch.ceil(n_bits_float).to(torch.int64)
    return n_bits.clamp(n_bits_min, n_bits_max)


def _qaq_apply_variable_bits(
    tensor: torch.Tensor,   # (batch, n_heads, seq_len, head_dim)
    n_bits_map: torch.Tensor,  # (batch, n_heads, seq_len)  int64
    outlier_ratio: float,
) -> torch.Tensor:
    """
    Apply per-token-head variable-bit quantization.
    Iterates over unique bit levels (typically 2–16) and applies them in bulk.
    """
    x = tensor.float()
    result = x.clone()
    b_min = int(n_bits_map.min().item())
    b_max = int(n_bits_map.max().item())
    for b in range(b_min, b_max + 1):
        idx = (n_bits_map == b)   # (batch, n_heads, seq_len) bool
        if not idx.any():
            continue
        # Gather tokens at this bit level: (N, head_dim)
        selected = x[idx]   # (N, head_dim)
        mask = _qaq_outlier_mask(selected, outlier_ratio)
        quantized = _qaq_uniform_quantize_vec(selected, b, mask)
        result[idx] = quantized
    return result.to(tensor.dtype)


# ---------------------------------------------------------------------------
# High-level QAQ cache functions
# ---------------------------------------------------------------------------

class QAQState:
    """
    Stores state needed for attention-aware quantization across decode steps.

    attn_history: list (length = n_layers) of tensors
                  (batch, n_heads, n_steps, seq_len)  — rolling window of
                  attention weights from recent decode steps.
    """
    def __init__(self, n_layers: int, window: int):
        self.window   = window
        self.n_layers = n_layers
        # attn_history[layer] = (batch, n_heads, seq_len) cumulative-max so far
        # We store the max-over-window directly as we accumulate (memory efficient)
        self.attn_max: list[torch.Tensor | None] = [None] * n_layers
        self.step_attns: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]

    def update(self, new_attns: list[torch.Tensor]):
        """new_attns[i]: (batch, n_heads, 1, seq_len) from one decode step."""
        for i, attn in enumerate(new_attns):
            a = attn.squeeze(2)   # (batch, n_heads, new_seq_len)
            new_len = a.shape[-1]
            # Pad older window entries to current length (new positions get 0 attention)
            padded = []
            for old_a in self.step_attns[i]:
                old_len = old_a.shape[-1]
                if old_len < new_len:
                    pad = torch.zeros(
                        old_a.shape[0], old_a.shape[1], new_len - old_len,
                        dtype=old_a.dtype, device=old_a.device,
                    )
                    old_a = torch.cat([old_a, pad], dim=-1)
                padded.append(old_a)
            padded.append(a)
            if len(padded) > self.window:
                padded.pop(0)
            self.step_attns[i] = padded
            self.attn_max[i] = torch.stack(padded, dim=0).amax(dim=0)

    def get_max_attn(self, layer_idx: int) -> torch.Tensor | None:
        return self.attn_max[layer_idx]


def _precompute_q_norm(model, input_ids: torch.Tensor, device: str) -> float:
    """
    Precompute the upper-10th-percentile of ||Q||^2 across all layers and heads.
    This requires a forward pass with output_attentions=True, then extracting Q
    via a hook on each attention layer.

    Paper Section 4.1: "precompute the distribution of the squared norm of query
    tensors and use the upper 10% quantile of this distribution in the formula."
    """
    q_norms = []
    hooks   = []

    def make_hook(layer_idx):
        def hook(module, args, kwargs, output):
            if not hasattr(module, 'head_dim'):
                return
            hs = args[0] if args else kwargs.get('hidden_states')
            if hs is None:
                return
            with torch.no_grad():
                head_dim = module.head_dim
                if hasattr(module, 'q_proj'):
                    # LLaMA-style: separate Q/K/V projections
                    q = module.q_proj(hs)   # (batch, seq, n_q_heads * head_dim)
                    n_heads = q.shape[-1] // head_dim
                elif hasattr(module, 'qkv_proj'):
                    # Phi-3 style: fused QKV projection
                    # qkv: (batch, seq, (n_q + n_k + n_v) * head_dim)
                    # For MHA: n_q = n_k = n_v = n_heads → Q is first 1/3
                    # For GQA: Q part = num_heads * head_dim
                    qkv = module.qkv_proj(hs)
                    if hasattr(module, 'num_heads'):
                        n_heads = module.num_heads
                    else:
                        n_heads = qkv.shape[-1] // (3 * head_dim)  # MHA assumption
                    q = qkv[:, :, :n_heads * head_dim]
                else:
                    return
                batch, seq, _ = hs.shape
                q = q.view(batch, seq, n_heads, head_dim).transpose(1, 2)
                q_norm_sq = (q.float() ** 2).sum(dim=-1)   # (batch, n_heads, seq)
                q_norms.append(q_norm_sq.flatten().cpu())
        return hook

    # Register hooks on each attention layer
    for idx, layer in enumerate(model.model.layers):
        h = layer.self_attn.register_forward_hook(make_hook(idx), with_kwargs=True)
        hooks.append(h)

    try:
        with torch.no_grad():
            model(input_ids, use_cache=False, output_attentions=False)
    finally:
        for h in hooks:
            h.remove()

    if not q_norms:
        return 300.0   # fallback (paper uses 300 as a grid-search value)
    all_norms = torch.cat(q_norms)
    pct_idx   = int(len(all_norms) * QAQ_Q_NORM_PERCENTILE / 100)
    pct_idx   = max(0, min(pct_idx, len(all_norms) - 1))
    q_norm    = float(all_norms.sort().values[pct_idx].item())
    return max(q_norm, 1.0)   # guard against zero


def _aggregate_attn_to_kv_heads(attn: torch.Tensor, n_kv_heads: int) -> torch.Tensor:
    """
    Aggregate attention from n_query_heads → n_kv_heads by averaging over groups.
    Handles both MHA (n_q == n_kv) and GQA (n_q > n_kv, e.g. LLaMA-3).

    attn: (batch, n_query_heads, *rest)
    Returns: (batch, n_kv_heads, *rest)
    """
    n_q_heads = attn.shape[1]
    if n_q_heads == n_kv_heads:
        return attn
    groups = n_q_heads // n_kv_heads
    # (batch, n_kv_heads, groups, *rest) → mean over groups
    shape = attn.shape
    new_shape = (shape[0], n_kv_heads, groups) + shape[2:]
    return attn.view(new_shape).mean(dim=2)


def _qaq_quantize_full_cache_aware(
    cache,
    q_norm: float,
    prefill_attns: list[torch.Tensor] | None = None,  # optional: (batch, n_q_heads, seq, seq) per layer
) -> tuple:
    """
    Quantize the full prefill KV cache with attention-aware variable-bit allocation.

    - Keys: uniform bit allocation per token-head derived from q_norm and seq_len
    - Values: if prefill_attns provided, bits inversely proportional to last-row attention;
              otherwise K formula is reused for V (avoids O(seq²) memory for long inputs).
    - Last (window-1) tokens kept at max bits (not enough attention history yet)

    Returns: (quantized cache, qaq_state)
    """
    n_layers = len(cache.layers)
    state    = QAQState(n_layers, QAQ_LAST_N_ATTENTIONS)
    _bits_sum = 0.0
    _bits_n   = 0

    for i, layer in enumerate(cache.layers):
        k = layer.keys    # (batch, n_kv_heads, seq_len, head_dim)
        v = layer.values
        seq_len = k.shape[2]

        # ----- KEY bits (uniform per-token, depends only on seq_len and q_norm) -----
        k_bits = _qaq_calc_bits_key(
            q_norm, k, QAQ_OUTLIER_RATIO, QAQ_TARGET_ERROR,
            QAQ_N_BITS_MIN, QAQ_N_BITS_MAX,
        )   # (batch, n_kv_heads, seq_len)
        # Last (window-1) tokens: keep at max bits (not enough history)
        if QAQ_LAST_N_ATTENTIONS > 1:
            k_bits[:, :, -(QAQ_LAST_N_ATTENTIONS - 1):] = QAQ_N_BITS_MAX

        # ----- VALUE bits -----
        if prefill_attns is not None:
            # Attention-aware: use last row of prefill attention
            attn = prefill_attns[i]   # (batch, n_q_heads, seq_len, seq_len)
            last_attn = attn[:, :, -1, :]   # (batch, n_q_heads, seq_len)
            # For GQA: aggregate query heads → KV heads
            n_kv_heads = v.shape[1]
            last_attn_kv = _aggregate_attn_to_kv_heads(last_attn, n_kv_heads)
            v_bits = _qaq_calc_bits_value(
                last_attn_kv, v, QAQ_OUTLIER_RATIO, QAQ_TARGET_ERROR,
                QAQ_N_BITS_MIN, QAQ_N_BITS_MAX,
            )
        else:
            # K-formula approximation for V (used when prefill attention unavailable,
            # e.g. for long inputs where O(seq²) attention would OOM).
            v_bits = k_bits.clone()
            last_attn = None

        if QAQ_LAST_N_ATTENTIONS > 1:
            v_bits[:, :, -(QAQ_LAST_N_ATTENTIONS - 1):] = QAQ_N_BITS_MAX

        # ----- Apply quantization -----
        layer.keys   = _qaq_apply_variable_bits(k, k_bits, QAQ_OUTLIER_RATIO)
        layer.values = _qaq_apply_variable_bits(v, v_bits, QAQ_OUTLIER_RATIO)
        _bits_sum += float(k_bits.float().sum().item()) + float(v_bits.float().sum().item())
        _bits_n   += int(k_bits.numel()) + int(v_bits.numel())

        # Initialise attention history
        if last_attn is not None:
            state.step_attns[i] = [last_attn]
            state.attn_max[i]   = last_attn
        else:
            state.step_attns[i] = []
            state.attn_max[i]   = None

    measured_avg_bits = (_bits_sum / max(_bits_n, 1)) if _bits_n > 0 else float("nan")
    # Account for outlier overhead (outlier_ratio of channels stored at 16 bits)
    measured_avg_bits = (1.0 - QAQ_OUTLIER_RATIO) * measured_avg_bits + QAQ_OUTLIER_RATIO * 16.0
    state.measured_avg_bits = measured_avg_bits
    return cache, state


def _qaq_quantize_new_token_aware(
    cache, new_attns: list[torch.Tensor] | None,
    q_norm: float, state: QAQState,
):
    """
    Quantize the newly appended token's KV entry.

    When new_attns is provided (eager attention-aware mode):
      - K bits: K formula (q_norm, seq_len)
      - V bits: attention-aware (inversely proportional to max-window attention)
    When new_attns is None (K-formula-only mode, used when eager is too slow):
      - K bits: K formula
      - V bits: same K formula (no attention needed)

    new_attns[i]: (batch, n_q_heads, 1, seq_len_new) from one eager decode step.
    """
    # Update attention window (only when attention available)
    if new_attns is not None:
        state.update(new_attns)

    for i, layer in enumerate(cache.layers):
        k = layer.keys    # (batch, n_heads, seq_len, head_dim)
        v = layer.values
        seq_len = k.shape[2]

        # Only quantize the LAST token (index -1), the rest are already quantized
        k_new = k[:, :, -1:, :]   # (batch, n_heads, 1, head_dim)
        v_new = v[:, :, -1:, :]

        # Key bits for new token
        k_bits_new = _qaq_calc_bits_key(
            q_norm, k_new, QAQ_OUTLIER_RATIO, QAQ_TARGET_ERROR,
            QAQ_N_BITS_MIN, QAQ_N_BITS_MAX,
        )   # (batch, n_heads, 1)

        # Value bits for new token
        if new_attns is not None:
            # Attention-aware: use max-window attention for this token position
            max_attn = state.get_max_attn(i)   # (batch, n_q_heads, seq_len) or None
            n_kv_heads = v_new.shape[1]
            if max_attn is not None:
                v_attn_score = _aggregate_attn_to_kv_heads(
                    max_attn[:, :, -1:], n_kv_heads
                )   # (batch, n_kv_heads, 1)
            else:
                v_attn_score = torch.ones(
                    (v_new.shape[0], n_kv_heads, 1), device=v_new.device
                )
            v_bits_new = _qaq_calc_bits_value(
                v_attn_score, v_new, QAQ_OUTLIER_RATIO, QAQ_TARGET_ERROR,
                QAQ_N_BITS_MIN, QAQ_N_BITS_MAX,
            )
        else:
            # K-formula approximation (used when attention weights unavailable)
            v_bits_new = k_bits_new.clone()

        k_quantized = _qaq_apply_variable_bits(k_new, k_bits_new, QAQ_OUTLIER_RATIO)
        v_quantized = _qaq_apply_variable_bits(v_new, v_bits_new, QAQ_OUTLIER_RATIO)

        layer.keys   = torch.cat([k[:, :, :-1, :], k_quantized], dim=2)
        layer.values = torch.cat([v[:, :, :-1, :], v_quantized], dim=2)

    return cache, state


def _average_bits_from_cache(cache) -> float:
    """Estimate average bits per element from current cache (FP16 = 16 bits)."""
    # We can't directly measure stored bits since we dequantize; report theoretical
    # average from outlier ratio and effective quantization.
    return (1.0 - QAQ_OUTLIER_RATIO) * ((QAQ_N_BITS_MIN + QAQ_N_BITS_MAX) / 2) \
           + QAQ_OUTLIER_RATIO * 16.0


# ---------------------------------------------------------------------------
# KV cache size helpers
# ---------------------------------------------------------------------------

def _actual_kv_mb(cache) -> float:
    """Actual VRAM used by the cache tensors (always FP16 in simulation)."""
    total = 0
    for layer in cache.layers:
        k, v = layer.keys, layer.values
        if k is not None:
            total += k.numel() * k.element_size()
        if v is not None:
            total += v.numel() * v.element_size()
    return total / (1024 ** 2)


def _theoretical_kv_mb_variable(cache, avg_bits: float) -> float:
    """Theoretical compressed size given an average bits-per-element."""
    actual_mb = _actual_kv_mb(cache)
    return actual_mb * (avg_bits / 16.0)


# ---------------------------------------------------------------------------
# Phase 1: Speed
# ---------------------------------------------------------------------------

def run_speed(model, tokenizer, model_key: str, device: str,
              attn_aware_decode: bool = True) -> dict:
    print("\n=== Speed benchmark (Full QAQ, attention-aware variable-bit) ===")

    # Match the speed-phase prompt length used by compression configs.
    target_len = LB_MAX_INPUT[model_key]
    all_ids    = tokenizer.encode(SPEED_TEXT, add_special_tokens=False)
    if len(all_ids) < target_len:
        raise RuntimeError(
            f"SPEED_TEXT only has {len(all_ids)} tokens; need {target_len}."
        )
    prompt_ids = all_ids[:target_len]
    input_ids  = torch.tensor([prompt_ids], device=device)
    print(f"  Prompt: {len(prompt_ids)} tokens")
    print(f"  QAQ: bits=[{QAQ_N_BITS_MIN},{QAQ_N_BITS_MAX}], outliers={QAQ_OUTLIER_RATIO*100:.0f}%, "
          f"window={QAQ_LAST_N_ATTENTIONS}, target_error={QAQ_TARGET_ERROR}")

    # --- Precompute q_norm from a short prompt (representative sample) ---
    print("  Precomputing q_norm (upper-10th-pct of ||Q||^2)...")
    q_norm = _precompute_q_norm(model, input_ids[:, :128], device)
    print(f"  q_norm = {q_norm:.1f}")

    # Warm-up
    with torch.no_grad():
        _ = model(input_ids[:, :32], use_cache=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # --- TTFT: prefill + variable-bit quantize ---
    # For attention-aware models: use eager + output_attentions to get per-layer attn
    # For K-formula-only models: use SDPA prefill (already in effect), K formula for V
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        if attn_aware_decode:
            # QAQ only consumes attn[:, :, -1, :] (the last query row). Materializing
            # the full [B, H, N, N] eager attention OOMs at N≈7500 on llama3-8B.
            # Two-pass: SDPA for the prefix (no attentions), then eager for just
            # the last token against that prefix → attention is [B, H, 1, N].
            seq_len = input_ids.shape[1]
            if seq_len > 1:
                model.set_attn_implementation("sdpa")
                prefix = model(input_ids[:, :-1], use_cache=True)
                prefix_past = prefix.past_key_values
                del prefix
                model.set_attn_implementation("eager")
                pos = torch.tensor([[seq_len - 1]], device=device)
                prefill_out = model(input_ids[:, -1:], past_key_values=prefix_past,
                                    use_cache=True, output_attentions=True,
                                    position_ids=pos)
            else:
                prefill_out = model(input_ids, use_cache=True, output_attentions=True)
            prefill_attns = list(prefill_out.attentions)
        else:
            # SDPA (phi3 etc.): no attention weights needed
            prefill_out   = model(input_ids, use_cache=True)
            prefill_attns = None
        past, state = _qaq_quantize_full_cache_aware(
            prefill_out.past_key_values, q_norm, prefill_attns
        )
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1000
    print(f"  TTFT (incl. quantize): {ttft_ms:.1f} ms")

    actual_mb = _actual_kv_mb(past)
    # Estimate average bits from variable-bit allocation
    # (we can't recover exact bits after dequantization, so report range)
    print(f"  KV cache after prefill: {actual_mb:.1f} MB actual (FP16 sim)")
    print(f"  Theoretical: bits range [{QAQ_N_BITS_MIN},{QAQ_N_BITS_MAX}], outliers at 16-bit")

    next_tok = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # --- TPOT: each decode step with attention-aware quantize of new token ---
    decode_times = []
    with torch.no_grad():
        for _ in range(SPEED_N_DECODE):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if attn_aware_decode:
                step = model(next_tok, past_key_values=past, use_cache=True,
                             output_attentions=True)
                new_attns = list(step.attentions)
                past, state = _qaq_quantize_new_token_aware(
                    step.past_key_values, new_attns, q_norm, state
                )
            else:
                # K-formula-only (SDPA): skip per-step decode quantization.
                # Prefill compression dominates; decode tokens are a small fraction.
                step = model(next_tok, past_key_values=past, use_cache=True)
                past = step.past_key_values
            torch.cuda.synchronize()
            decode_times.append((time.perf_counter() - t0) * 1000)
            next_tok = step.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    tpot_ms        = sum(decode_times) / len(decode_times)
    tokens_per_sec = 1000.0 / tpot_ms
    peak_vram_mb   = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Measured avg bits from actual prefill bit allocation (K + V combined)
    avg_bits          = float(getattr(state, "measured_avg_bits", float("nan")))
    theoretical_mb    = _theoretical_kv_mb_variable(past, avg_bits)
    compression_ratio = 16.0 / avg_bits

    print(f"  TPOT (incl. attn+quantize): {tpot_ms:.2f} ms/tok  ({tokens_per_sec:.1f} tok/s)")
    print(f"  Peak VRAM: {peak_vram_mb:.0f} MB")
    print(f"  Measured avg bits: {avg_bits:.2f}  Compression: {compression_ratio:.2f}×")

    return {
        "ttft_ms":              round(ttft_ms, 2),
        "tpot_ms":              round(tpot_ms, 3),
        "tokens_per_sec":       round(tokens_per_sec, 2),
        "peak_vram_mb":         round(peak_vram_mb, 1),
        "kv_cache_mb":          round(theoretical_mb, 2),
        "kv_compression_ratio": round(compression_ratio, 3),
        "speed_prompt_tokens":  len(prompt_ids),
        "speed_n_decode":       SPEED_N_DECODE,
        "qaq_n_bits_min":       QAQ_N_BITS_MIN,
        "qaq_n_bits_max":       QAQ_N_BITS_MAX,
        "qaq_outlier_ratio":    QAQ_OUTLIER_RATIO,
        "qaq_target_error":     QAQ_TARGET_ERROR,
        "qaq_window":           QAQ_LAST_N_ATTENTIONS,
        "qaq_q_norm":           round(q_norm, 2),
        "qaq_avg_bits":         round(avg_bits, 3),
    }


# ---------------------------------------------------------------------------
# Phase 3: LongBench (QAQ generation)
# ---------------------------------------------------------------------------

def _generate_qaq(
    model, tokenizer, prompt: str, max_new: int, max_input: int,
    device: str, use_chat: bool, q_norm: float, attn_aware_decode: bool = True,
) -> str:
    """
    Generate with full QAQ attention-aware variable-bit KV cache quantization.

    Prefill (all modes): SDPA, K formula for both K and V bits (avoids O(seq²) OOM).
    Decode (attn_aware_decode=True):  eager + output_attentions, attention-aware V bits.
    Decode (attn_aware_decode=False): SDPA, K formula for V bits (fast, for phi3, etc.)
    """
    if use_chat:
        prompt = apply_chat_template(tokenizer, prompt)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > max_input:
        half = max_input // 2
        ids  = ids[:half] + ids[-half:]
    ids_t = torch.tensor([ids], device=device)

    generated_ids = []

    with torch.no_grad():
        # Prefill: use SDPA (O(N) memory) to avoid OOM on long inputs.
        # QAQ K bits use the K formula (no attention needed).
        # QAQ V bits also use K formula approximation for prefill;
        # attention-aware V bits kick in from the first decode step onward.
        orig_impl = model.config._attn_implementation
        model.config._attn_implementation = "sdpa"
        try:
            out = model(ids_t, past_key_values=None, use_cache=True)
        finally:
            model.config._attn_implementation = orig_impl
        past, state = _qaq_quantize_full_cache_aware(
            out.past_key_values, q_norm, prefill_attns=None
        )
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if next_tok.item() != tokenizer.eos_token_id:
            generated_ids.append(next_tok.item())

        # Decode: each step gets attention weights for the new token
        for _ in range(max_new - 1):
            if next_tok.item() == tokenizer.eos_token_id:
                break
            if attn_aware_decode:
                # Eager + output_attentions: attention-aware V bits (1 tok = O(seq_len), safe)
                out = model(next_tok, past_key_values=past, use_cache=True,
                            output_attentions=True)
                new_attns = list(out.attentions)
                past, state = _qaq_quantize_new_token_aware(
                    out.past_key_values, new_attns, q_norm, state
                )
            else:
                # SDPA, K-formula-only: skip per-step decode quantization.
                # Decode tokens are a small fraction of total KV; prefill compression dominates.
                out  = model(next_tok, past_key_values=past, use_cache=True)
                past = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if next_tok.item() != tokenizer.eos_token_id:
                generated_ids.append(next_tok.item())

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def run_longbench(
    model, tokenizer, model_key: str, device: str, q_norm: float,
    attn_aware_decode: bool = True, pp_logger=None,
) -> dict:
    print("\n=== LongBench (Full QAQ, attention-aware variable-bit generation) ===")
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
            tmpl   = LB_TEMPLATES[task]
            prompt = tmpl.format(context=sample.get("context", ""), input=sample["input"])
            pred   = _generate_qaq(
                model, tokenizer, prompt, max_new, max_input, device,
                use_chat=cfg["chat"], q_norm=q_norm,
                attn_aware_decode=attn_aware_decode,
            )
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

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  required=True, choices=list(MODELS.keys()))
    p.add_argument("--output", default="runs/C2")
    p.add_argument("--skip-speed-ppl", action="store_true",
                   help="skip speed + perplexity phases; reuse qaq_q_norm from existing results.json")
    return p.parse_args()


def main():
    args      = parse_args()
    mcfg      = MODELS_CFG[args.model]
    model_id  = mcfg["id"]
    attn_impl = mcfg["attn_impl"]
    attn_aware_decode = mcfg["attn_aware_decode"]
    out_dir   = Path(args.output) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    decode_mode = "attention-aware" if attn_aware_decode else "prefill-only (SDPA, K-formula)"
    print(f"\nConfig : C1 (Full QAQ — variable-bit [{QAQ_N_BITS_MIN},{QAQ_N_BITS_MAX}], "
          f"{QAQ_OUTLIER_RATIO*100:.0f}% outliers, window={QAQ_LAST_N_ATTENTIONS}, "
          f"decode={decode_mode})")
    print(f"Model  : {model_id}")
    print(f"Device : {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    print(f"Output : {out_dir}/results.json")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="auto",
        attn_implementation=attn_impl,
    )
    model.eval()

    results = {
        "config":   "C2",
        "method":   f"QAQ full (variable-bit [{QAQ_N_BITS_MIN},{QAQ_N_BITS_MAX}], "

                    f"{QAQ_OUTLIER_RATIO*100:.0f}% outliers, window={QAQ_LAST_N_ATTENTIONS}, "
                    f"decode={decode_mode})",
        "model":    args.model,
        "model_id": model_id,
    }

    if args.skip_speed_ppl:
        old_path = out_dir / "results.json"
        if not old_path.exists():
            raise FileNotFoundError(
                f"--skip-speed-ppl requires existing {old_path} (need qaq_q_norm)")
        old = json.loads(old_path.read_text())
        if "qaq_q_norm" not in old:
            raise KeyError(f"qaq_q_norm not found in {old_path}")
        q_norm = old["qaq_q_norm"]
    else:
        speed_results = run_speed(model, tokenizer, args.model, device,
                                  attn_aware_decode=attn_aware_decode)
        results.update(speed_results)
        q_norm = speed_results["qaq_q_norm"]
        torch.cuda.empty_cache()

        # PPL is teacher-forced and doesn't need attention scores. Switching to
        # SDPA avoids eager attention's O(N^2) softmax tensor on llama3-8B at
        # PPL_MAX_LEN=8192 (which would otherwise OOM / take an hour).
        prev_impl = MODELS_CFG[args.model]["attn_impl"]
        if prev_impl != "sdpa":
            model.set_attn_implementation("sdpa")
        results.update(run_ppl(model, tokenizer, args.model, device))
        torch.cuda.empty_cache()
        if prev_impl != "sdpa":
            model.set_attn_implementation(prev_impl)

    lb = run_longbench(model, tokenizer, args.model, device, q_norm,
                       attn_aware_decode=attn_aware_decode,
                       pp_logger=PerPromptLogger(out_dir / "per_prompt.jsonl",
                                                 config="C2", model=args.model))
    results["longbench"] = lb
    if lb:
        results["longbench_avg"] = round(
            sum(v["score"] for v in lb.values()) / len(lb), 4
        )

    out_path = out_dir / "results.json"
    if out_path.exists():
        old = json.loads(out_path.read_text())
        results = {**old, **results}
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out_path}")
    print(json.dumps({k: v for k, v in results.items() if k != "longbench"}, indent=2))


if __name__ == "__main__":
    main()
