#!/usr/bin/env python3
"""
C0 baseline benchmark — HuggingFace FP16, full KV cache, no compression.

Measures:
  Speed  : TTFT (ms), TPOT (ms), tokens/sec, peak VRAM (MB), KV cache size (MB)
  Quality: WikiText-2 perplexity, LongBench accuracy (7 tasks)

Output: runs/C0/{model}/results.json

Usage:
    python scripts/benchmark_c0.py --model phi3
    python scripts/benchmark_c0.py --model llama3
"""

from __future__ import annotations

import argparse
import json
import math
import string
import time
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS = {
    "phi3":   "microsoft/Phi-3-mini-4k-instruct",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}

# Max input tokens for LongBench (leave room for generation)
LB_MAX_INPUT = {
    "phi3":   3500,
    "llama3": 7500,
}

LB_TASKS = {
    "narrativeqa": {"metric": "f1",       "max_new_tokens": 128, "chat": True,  "first_line": False},
    "qasper":      {"metric": "f1",       "max_new_tokens": 128, "chat": True,  "first_line": False},
    "hotpotqa":    {"metric": "f1",       "max_new_tokens": 32,  "chat": True,  "first_line": True},
    "2wikimqa":    {"metric": "f1",       "max_new_tokens": 32,  "chat": True,  "first_line": True},
    "gov_report":  {"metric": "rouge1",   "max_new_tokens": 512, "chat": True,  "first_line": False},
    "trec":        {"metric": "accuracy", "max_new_tokens": 32,  "chat": False, "first_line": True},
    "triviaqa":    {"metric": "f1",       "max_new_tokens": 32,  "chat": False, "first_line": True},
}

# Task-specific prompt templates (from LongBench / DynamicKV / Ada-KV)
# Uses both {context} (few-shot examples or document) and {input} (question)
LB_TEMPLATES = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question as concisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on "
        "the story as concisely as you can, using a single phrase if possible. Do not provide any "
        "explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely "
        "as you can, using a single phrase or sentence if possible. If the question cannot be "
        "answered based on the information in the article, write \"unanswerable\". If the question "
        "is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any "
        "explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as "
        "concisely as you can, using a single phrase or sentence if possible. If the question cannot "
        "be answered based on the information in the article, write \"unanswerable\". If the question "
        "is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any "
        "explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the "
        "question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the "
        "question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Report:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:"
    ),
    "trec": (
        "Please determine the type of the question below. Here are some examples of questions.\n\n"
        "{context}\n{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer and do not output "
        "any other words. The following are some examples.\n\n{context}\n\n{input}"
    ),
}

LB_CACHE    = Path("~/.cache/longbench/data").expanduser()
LB_N_SAMPLES = 20

# PPL: non-overlapping chunks, all tokens scored (GPTQ/KVQuant standard)
# KVQuant uses model's native context length: 8192 for LLaMA-3, 4096 for Phi-3
PPL_MAX_LEN = {
    "phi3":   4096,
    "llama3": 8192,
}
PPL_N_CHUNKS = None   # None = use full dataset (all complete chunks)

# Speed: fixed-length prompt, generate this many tokens for TPOT
SPEED_N_DECODE = 50

# A ~1024-token prompt about LLM inference (neutral, no task bias)
_SPEED_TEXT = (
    "The development of large language models has transformed natural language "
    "processing. These models, trained on vast corpora of text, demonstrate "
    "remarkable capabilities across a wide range of tasks including translation, "
    "summarization, question answering, and code generation. However, deploying "
    "these models in production environments presents significant challenges. "
    "The primary bottleneck is memory: as context length grows, the key-value "
    "cache required to store intermediate attention states grows linearly, "
    "consuming tens of gigabytes for contexts exceeding one hundred thousand tokens. "
    "Researchers have proposed various techniques to address this challenge, "
    "including quantization of cache elements to lower numerical precision, "
    "eviction of less important tokens based on attention scores, and adaptive "
    "allocation of memory budgets across attention heads and transformer layers. "
    "Each approach involves a quality-efficiency trade-off that must be carefully "
    "characterized for different task types and sequence lengths. "
) * 10


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def _token_f1(pred: str, gold: str) -> float:
    pred_toks = _normalize(pred).split()
    gold_toks = _normalize(gold).split()
    if not pred_toks or not gold_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p = n / len(pred_toks)
    r = n / len(gold_toks)
    return 2 * p * r / (p + r)


def _rouge1(pred: str, ref: str) -> float:
    pred_set = set(_normalize(pred).split())
    ref_set  = set(_normalize(ref).split())
    if not pred_set or not ref_set:
        return 0.0
    common = len(pred_set & ref_set)
    p = common / len(pred_set)
    r = common / len(ref_set)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _accuracy(pred: str, golds: list[str]) -> float:
    pred_norm = _normalize(pred)
    return float(any(
        _normalize(g) in pred_norm or pred_norm in _normalize(g)
        for g in golds
    ))


def compute_score(metric: str, pred: str, golds: list[str]) -> float:
    if metric == "f1":
        return max(_token_f1(pred, g) for g in golds)
    if metric == "rouge1":
        return max(_rouge1(pred, g) for g in golds)
    if metric == "accuracy":
        return _accuracy(pred, golds)
    raise ValueError(f"Unknown metric: {metric}")


# ---------------------------------------------------------------------------
# KV cache size
# ---------------------------------------------------------------------------

def _kv_cache_mb(past) -> float:
    """Return total bytes (as MB) of all key+value tensors in the cache."""
    total = 0
    try:
        for k, v in zip(past.key_cache, past.value_cache):
            if k is not None:
                total += k.numel() * k.element_size()
            if v is not None:
                total += v.numel() * v.element_size()
    except AttributeError:
        # Older transformers: list of (k, v) tuples per layer
        for layer in past:
            k, v = layer[0], layer[1]
            total += k.numel() * k.element_size()
            total += v.numel() * v.element_size()
    return total / (1024 ** 2)


# ---------------------------------------------------------------------------
# Phase 1: Speed
# ---------------------------------------------------------------------------

def run_speed(model, tokenizer, model_key: str, device: str) -> dict:
    print("\n=== Speed benchmark ===")

    # Build a prompt of exactly min(1024, LB_MAX_INPUT) tokens
    target_len = min(1024, LB_MAX_INPUT[model_key])
    all_ids = tokenizer.encode(_SPEED_TEXT, add_special_tokens=False)
    prompt_ids = all_ids[:target_len]
    input_ids  = torch.tensor([prompt_ids], device=device)
    print(f"  Prompt: {len(prompt_ids)} tokens")

    # Warm-up pass (discarded)
    with torch.no_grad():
        _ = model(input_ids[:, :32], use_cache=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # --- TTFT: time the full prefill ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        prefill_out = model(input_ids, use_cache=True)
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1000
    print(f"  TTFT: {ttft_ms:.1f} ms")

    kv_mb = _kv_cache_mb(prefill_out.past_key_values)
    print(f"  KV cache after prefill: {kv_mb:.1f} MB")

    # First generated token
    past         = prefill_out.past_key_values
    next_tok     = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # --- TPOT: time each decode step individually ---
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

    tpot_ms       = sum(decode_times) / len(decode_times)
    tokens_per_sec = 1000.0 / tpot_ms
    peak_vram_mb  = torch.cuda.max_memory_allocated() / (1024 ** 2)

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
# Phase 2: Perplexity
# ---------------------------------------------------------------------------

def run_ppl(model, tokenizer, model_key: str, device: str) -> dict:
    print("\n=== WikiText-2 perplexity ===")

    chunk_len = PPL_MAX_LEN[model_key]
    ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids  = tokenizer.encode(text)
    print(f"  Total tokens: {len(ids):,}")

    n_chunks = len(ids) // chunk_len
    if PPL_N_CHUNKS is not None:
        n_chunks = min(PPL_N_CHUNKS, n_chunks)
    print(f"  Chunks: {n_chunks}  (chunk_size={chunk_len}, score all tokens)")

    total_nll   = 0.0
    total_count = 0

    with torch.no_grad():
        for i in range(n_chunks):
            start   = i * chunk_len
            chunk   = ids[start : start + chunk_len]
            input_t = torch.tensor([chunk], device=device)           # (1, L)

            # Teacher forcing: model sees ground-truth tokens at every position
            out     = model(input_t, use_cache=False)

            # logits[t] predicts token[t+1]; score positions 1..L-1
            logits  = out.logits[:, :-1, :]          # (1, L-1, vocab)
            targets = input_t[:, 1:]                 # (1, L-1)

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            nll       = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
            total_nll   += nll.sum().item()
            total_count += targets.numel()

            if (i + 1) % 5 == 0:
                print(f"  chunk {i+1}/{n_chunks}  PPL={math.exp(total_nll/total_count):.3f}")

    ppl = math.exp(total_nll / total_count)
    print(f"  Final PPL: {ppl:.4f}")
    return {"ppl_wikitext2": round(ppl, 4), "ppl_n_chunks": n_chunks}


# ---------------------------------------------------------------------------
# Phase 3: LongBench
# ---------------------------------------------------------------------------

def _load_task(task: str) -> list[dict]:
    path = LB_CACHE / f"{task}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    step = max(1, len(records) // LB_N_SAMPLES)
    return records[::step][:LB_N_SAMPLES]


def _apply_chat_template(tokenizer, prompt: str) -> str:
    """Wrap prompt in the model's chat template (user turn only, no system prompt)."""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def _generate(model, tokenizer, prompt: str, max_new: int, max_input: int, device: str,
              use_chat: bool = True) -> str:
    if use_chat:
        prompt = _apply_chat_template(tokenizer, prompt)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > max_input:
        # Keep first half + last half (preserves instructions and question)
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
        )
    return tokenizer.decode(out[0, len(ids):], skip_special_tokens=True).strip()


def run_longbench(model, tokenizer, model_key: str, device: str) -> dict:
    print("\n=== LongBench ===")
    max_input = LB_MAX_INPUT[model_key]
    results   = {}

    for task, cfg in LB_TASKS.items():
        try:
            samples = _load_task(task)
        except FileNotFoundError:
            print(f"  SKIP {task}: file not found")
            continue

        metric  = cfg["metric"]
        max_new = cfg["max_new_tokens"]
        scores  = []

        for j, sample in enumerate(samples):
            golds = sample["answers"] if isinstance(sample["answers"], list) else [sample["answers"]]
            tmpl  = LB_TEMPLATES[task]
            prompt = tmpl.format(context=sample.get("context", ""), input=sample["input"])
            pred  = _generate(model, tokenizer, prompt, max_new, max_input, device,
                              use_chat=cfg["chat"])
            if cfg["first_line"]:
                pred = pred.split("\n")[0].strip()
            s     = compute_score(metric, pred, golds)
            scores.append(s)
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
    p.add_argument("--output", default="runs/C0")
    return p.parse_args()


def main():
    args     = parse_args()
    model_id = MODELS[args.model]
    out_dir  = Path(args.output) / args.model
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

    # Phase 1: Speed
    results.update(run_speed(model, tokenizer, args.model, device))
    torch.cuda.empty_cache()

    # Phase 2: Perplexity
    results.update(run_ppl(model, tokenizer, args.model, device))
    torch.cuda.empty_cache()

    # Phase 3: LongBench
    lb = run_longbench(model, tokenizer, args.model, device)
    results["longbench"] = lb
    if lb:
        results["longbench_avg"] = round(
            sum(v["score"] for v in lb.values()) / len(lb), 4
        )

    # Save
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out_path}")
    print(json.dumps({k: v for k, v in results.items() if k != "longbench"}, indent=2))


if __name__ == "__main__":
    main()
