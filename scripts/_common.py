"""
Shared helpers for AdaptiveServe benchmarks (C0, C1, C2, ...).

Only contains code that is genuinely identical across configurations.
Method-specific logic (speed loop, generation, cache handling, model loading)
stays inside each ``benchmark_c*.py`` script.
"""

from __future__ import annotations

import json
import math
import string
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Models & per-model limits
# ---------------------------------------------------------------------------

MODELS: dict[str, str] = {
    "phi3":       "microsoft/Phi-3-mini-4k-instruct",
    "llama3":     "meta-llama/Meta-Llama-3-8B-Instruct",
    # Llama-3.1/3.2 same-recipe distill pair — used for the "real routing" experiment.
    # Llama-3.2-3B is the official compressed/distilled child of Llama-3.1-8B, so
    # the quality gap is purely a function of capacity, not training recipe.
    "llama31_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama32_3b": "meta-llama/Llama-3.2-3B-Instruct",
}

# Max input tokens for LongBench (leave room for generation)
LB_MAX_INPUT: dict[str, int] = {
    "phi3":       3500,
    "llama3":     7500,
    "llama31_8b": 7500,
    "llama32_3b": 7500,
}

# PPL: non-overlapping chunks. KVQuant standard: model's native context length.
# Llama-3.x is set to 8192 (matching llama3) rather than the full 131072
# context to keep log_softmax tractable on a 24GB GPU.
PPL_MAX_LEN: dict[str, int] = {
    "phi3":       4096,
    "llama3":     8192,
    "llama31_8b": 8192,
    "llama32_3b": 8192,
}
PPL_N_CHUNKS: int | None = None  # None = full WikiText-2 test split

# Speed benchmark: decode this many tokens after a 1024-token prefill.
SPEED_N_DECODE = 50

# A neutral ~1024-token prompt about LLM inference. Repeated to cover the
# longest target prefill length.
SPEED_TEXT = (
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
) * 50


# ---------------------------------------------------------------------------
# LongBench tasks
# ---------------------------------------------------------------------------

LB_TASKS: dict[str, dict] = {
    "narrativeqa":     {"metric": "f1",       "max_new_tokens": 128, "chat": True,  "first_line": False},
    "qasper":          {"metric": "f1",       "max_new_tokens": 128, "chat": True,  "first_line": False},
    "multifieldqa_en": {"metric": "f1",       "max_new_tokens": 64,  "chat": True,  "first_line": True},
    "hotpotqa":        {"metric": "f1",       "max_new_tokens": 32,  "chat": True,  "first_line": True},
    "2wikimqa":        {"metric": "f1",       "max_new_tokens": 32,  "chat": True,  "first_line": True},
    "gov_report":      {"metric": "rouge1",   "max_new_tokens": 512, "chat": True,  "first_line": False},
    "multi_news":      {"metric": "rouge1",   "max_new_tokens": 512, "chat": True,  "first_line": False},
    "qmsum":           {"metric": "rouge1",   "max_new_tokens": 512, "chat": True,  "first_line": False},
    "passage_count":   {"metric": "count_em", "max_new_tokens": 32,  "chat": True,  "first_line": True},
    "trec":            {"metric": "accuracy", "max_new_tokens": 32,  "chat": False, "first_line": True},
    "triviaqa":        {"metric": "f1",       "max_new_tokens": 32,  "chat": False, "first_line": True},
}

LB_TEMPLATES: dict[str, str] = {
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
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following "
        "question based on the above text, only give me the answer and do not output any other "
        "words.\n\nQuestion: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news.\n\nNews:\n"
        "{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or instruction. "
        "Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the "
        "query based on the above meeting transcript in one or more sentences.\n\n"
        "Query: {input}\nAnswer:"
    ),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. "
        "Please carefully read these paragraphs and determine how many unique paragraphs there are "
        "after removing duplicates. In other words, how many non-repeating paragraphs are there in "
        "total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing "
        "duplicates. The output format should only contain the number, such as 1, 2, 3, and so on."
        "\n\nThe final answer is: "
    ),
}

LB_CACHE     = Path("~/.cache/longbench/data").expanduser()
LB_N_SAMPLES = 20


def load_longbench_task(task: str) -> list[dict]:
    """Load a LongBench task file (cached locally) and return up to LB_N_SAMPLES records."""
    path = LB_CACHE / f"{task}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    step = max(1, len(records) // LB_N_SAMPLES)
    return records[::step][:LB_N_SAMPLES]


# ---------------------------------------------------------------------------
# MT-Bench (chat / open-ended quality)
# ---------------------------------------------------------------------------
#
# Official 80-prompt evaluation from FastChat (Zheng et al., NeurIPS 2023).
# We use it as a *short-prompt, open-ended* counterpart to LongBench, primarily
# to evaluate model-routing policies (Phi-3 vs LLaMA-3) where KV-cache
# compression is largely irrelevant (prompts are <300 tokens).
#
# Source file pinned to the FastChat commit that introduced MT-Bench.
MTBENCH_CACHE          = Path("~/.cache/mtbench").expanduser()
MTBENCH_QUESTION_FILE  = MTBENCH_CACHE / "question.jsonl"
MTBENCH_QUESTION_URL   = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/mt_bench/question.jsonl"
)
MTBENCH_MAX_NEW_TOKENS = 1024


def load_mtbench(turn: int = 1) -> list[dict]:
    """Return the 80 official MT-Bench prompts for the requested turn (1 or 2).

    Each record: {prompt_id, category, prompt, reference (optional)}.
    Downloads to MTBENCH_QUESTION_FILE on first call.
    """
    if turn not in (1, 2):
        raise ValueError(f"turn must be 1 or 2, got {turn}")

    if not MTBENCH_QUESTION_FILE.exists():
        import urllib.request
        MTBENCH_CACHE.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading MT-Bench questions → {MTBENCH_QUESTION_FILE}")
        urllib.request.urlretrieve(MTBENCH_QUESTION_URL, MTBENCH_QUESTION_FILE)

    out = []
    for line in MTBENCH_QUESTION_FILE.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        turns = rec.get("turns", [])
        if len(turns) < turn:
            continue
        ref = rec.get("reference", None)
        out.append({
            "prompt_id": rec["question_id"],
            "category":  rec.get("category", "unknown"),
            "prompt":    turns[turn - 1],
            "reference": ref[turn - 1] if ref and len(ref) >= turn else None,
        })
    return out


def apply_chat_template(tokenizer, prompt: str) -> str:
    """Wrap prompt in the model's chat template (user turn, with generation prompt)."""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


# ---------------------------------------------------------------------------
# Scoring helpers (LongBench)
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


def _count_em(pred: str, golds: list[str]) -> float:
    """Exact-match on first integer in prediction vs gold integer.
    Used for passage_count."""
    import re as _re
    m = _re.search(r"-?\d+", pred)
    if m is None:
        return 0.0
    try:
        p = int(m.group(0))
    except ValueError:
        return 0.0
    for g in golds:
        gm = _re.search(r"-?\d+", str(g))
        if gm is None:
            continue
        try:
            if int(gm.group(0)) == p:
                return 1.0
        except ValueError:
            continue
    return 0.0


def compute_score(metric: str, pred: str, golds: list[str]) -> float:
    if metric == "f1":
        return max(_token_f1(pred, g) for g in golds)
    if metric == "rouge1":
        return max(_rouge1(pred, g) for g in golds)
    if metric == "accuracy":
        return _accuracy(pred, golds)
    if metric == "count_em":
        return _count_em(pred, golds)
    raise ValueError(f"Unknown metric: {metric}")


# ---------------------------------------------------------------------------
# Perplexity (method-agnostic: only requires model.forward + use_cache=False)
# ---------------------------------------------------------------------------

def run_ppl(model, tokenizer, model_key: str, device: str) -> dict:
    """WikiText-2 perplexity via teacher-forcing on non-overlapping chunks."""
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
            input_t = torch.tensor([chunk], device=device)

            out     = model(input_t, use_cache=False)
            logits  = out.logits[:, :-1, :]
            targets = input_t[:, 1:]

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
# KV cache size (FP16 default — methods that compress define their own)
# ---------------------------------------------------------------------------

def kv_cache_mb(past) -> float:
    """Total bytes (as MB) of every key+value tensor in a HuggingFace cache."""
    total = 0
    try:
        for k, v in zip(past.key_cache, past.value_cache):
            if k is not None:
                total += k.numel() * k.element_size()
            if v is not None:
                total += v.numel() * v.element_size()
    except AttributeError:
        for layer in past:
            k, v = layer[0], layer[1]
            total += k.numel() * k.element_size()
            total += v.numel() * v.element_size()
    return total / (1024 ** 2)


# ---------------------------------------------------------------------------
# Per-prompt feature extraction (for C7 selector)
# ---------------------------------------------------------------------------
#
# All features are prompt-intrinsic — no LLM forward pass and no task label.
# The classifier sees only what arrives at inference time.
#
#   seq_len_tokens     : prompt length in model tokens (dominant signal)
#   seq_len_chars      : prompt length in characters (cheap fallback if no tokenizer)
#   token_entropy      : Shannon entropy of the prompt's own token distribution
#                        (low = repetitive context, high = diverse content)
#   gzip_ratio         : len(gzip(prompt)) / len(prompt)
#                        (information-theoretic redundancy proxy)
#   unique_token_ratio : type/token ratio over model tokens
#   question_position  : char-offset of last "?" / total length, NaN if absent
#                        (proxy for "lost in the middle"-style long-range dependence)
#   newline_density    : count("\n") / len(chars)  (RAG/multi-doc structure proxy)

def _shannon_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def extract_prompt_features(prompt: str, tokenizer=None) -> dict:
    """Cheap (<50 ms), prompt-intrinsic features for the routing classifier.

    Pass the model tokenizer for accurate token-level features; otherwise
    falls back to whitespace splitting.
    """
    import gzip as _gzip

    if not prompt:
        return {
            "seq_len_tokens":     0,
            "seq_len_chars":      0,
            "token_entropy":      0.0,
            "gzip_ratio":         1.0,
            "unique_token_ratio": 0.0,
            "question_position":  None,
            "newline_density":    0.0,
        }

    if tokenizer is not None:
        try:
            ids = tokenizer.encode(prompt, add_special_tokens=False)
        except Exception:
            ids = prompt.split()
    else:
        ids = prompt.split()

    n_tokens   = len(ids)
    counts     = Counter(ids)
    n_unique   = len(counts)

    raw_bytes  = prompt.encode("utf-8", errors="ignore")
    gzip_bytes = len(_gzip.compress(raw_bytes, compresslevel=6))
    gzip_ratio = gzip_bytes / max(len(raw_bytes), 1)

    last_q  = prompt.rfind("?")
    q_pos   = (last_q / len(prompt)) if last_q >= 0 else float("nan")
    nl_dens = prompt.count("\n") / max(len(prompt), 1)

    return {
        "seq_len_tokens":     n_tokens,
        "seq_len_chars":      len(prompt),
        "token_entropy":      round(_shannon_entropy(list(counts.values())), 4),
        "gzip_ratio":         round(gzip_ratio, 4),
        "unique_token_ratio": round(n_unique / max(n_tokens, 1), 4),
        "question_position":  round(q_pos, 4) if q_pos == q_pos else None,
        "newline_density":    round(nl_dens, 4),
    }


# ---------------------------------------------------------------------------
# Per-prompt logging (for oracle / classifier dataset construction)
# ---------------------------------------------------------------------------

class PerPromptLogger:
    """Append-only JSONL writer.  One record per LongBench sample.

    Each record:
      {
        "config":       "C0" / "C1" / ...,
        "model":        "llama3" / "phi3",
        "task":         "narrativeqa" / ...,
        "sample_idx":   int,
        "metric":       "f1" / "rouge1" / "accuracy",
        "score":        float,
        "features":     {...prompt-intrinsic features...},
      }

    Features are extracted once per (task, sample_idx) regardless of config —
    callers may pre-extract and pass them in to avoid redundant work, but the
    logger will compute them lazily if not provided.
    """

    def __init__(self, path: Path | str, config: str, model: str):
        self.path   = Path(path)
        self.config = config
        self.model  = model
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate at start of run so reruns don't accumulate stale rows.
        self.path.write_text("")

    def log(self, *, task: str, sample_idx: int, metric: str, score: float,
            features: dict) -> None:
        rec = {
            "config":     self.config,
            "model":      self.model,
            "task":       task,
            "sample_idx": sample_idx,
            "metric":     metric,
            "score":      round(float(score), 6),
            "features":   features,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
