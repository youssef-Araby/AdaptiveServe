#!/usr/bin/env python3
"""
MT-Bench judging — calls OpenAI to score cached generations on a 1–10 rubric.

Strict separation from the generation phase: no GPU, no model loading.
Reads generations.jsonl files written by benchmark_mtbench.py and produces
judgments.jsonl alongside them.

Cost controls (all enforced):
  - Pre-flight cost preview with [y/N] confirmation (override: --yes).
  - Hard budget cap from --budget USD or env $MTBENCH_BUDGET_USD.
  - Persistent disk cache keyed by sha256(judge_model || prompt || answer).
    Cached calls cost $0 and skip the API entirely.
  - --dry-run prints the cost preview and exits without any API call.

Auth:
  Reads OPENAI_API_KEY strictly from the environment. Fails loudly if unset.
  The key value is never written to disk, logs, or stdout.

Output:
  runs/mtbench/{config}/{model}/judgments.jsonl
  runs/mtbench/judge_cache/{judge_model}.jsonl   (gitignored)
  runs/mtbench/budget.log                        (gitignored)

Usage:
  python scripts/judge_mtbench.py --model phi3 --config C0 --judge gpt-4o-mini
  python scripts/judge_mtbench.py --model phi3 --config C0 --dry-run
  python scripts/judge_mtbench.py --all --judge gpt-4o-mini --budget 5.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from <repo_root>/.env into os.environ.

    Existing env vars take precedence (so `export OPENAI_API_KEY=...` still
    wins over the file). No external dependency.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()

# ---------------------------------------------------------------------------
# Pricing (USD per 1M tokens, as of 2025-Q1 OpenAI public pricing).
# Update if pricing changes; used only for pre-flight estimates and budget cap.
# ---------------------------------------------------------------------------

PRICING_USD_PER_MTOK: dict[str, dict[str, float]] = {
    "gpt-4o-mini":        {"input": 0.15,  "output": 0.60},
    "gpt-4o":             {"input": 2.50,  "output": 10.00},
    "gpt-4o-2024-08-06":  {"input": 2.50,  "output": 10.00},
}

# Rough estimate for cost preview. The judge prompt + cached answer is the
# dominant input; the judge's reasoning + score is the output.
EST_INPUT_TOKENS_PER_CALL  = 1200
EST_OUTPUT_TOKENS_PER_CALL = 350


# ---------------------------------------------------------------------------
# Official MT-Bench single-turn pointwise judge prompt (from FastChat).
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are a helpful assistant."
)

JUDGE_USER_TEMPLATE = (
    "[Instruction]\n"
    "Please act as an impartial judge and evaluate the quality of the response provided "
    "by an AI assistant to the user question displayed below. Your evaluation should "
    "consider factors such as the helpfulness, relevance, accuracy, depth, creativity, "
    "and level of detail of the response. Begin your evaluation by providing a short "
    "explanation. Be as objective as possible. After providing your explanation, you must "
    "rate the response on a scale of 1 to 10 by strictly following this format: "
    "\"[[rating]]\", for example: \"Rating: [[5]]\".\n\n"
    "[Question]\n{question}\n\n"
    "[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]"
)

# Variant used for math/reasoning categories — includes the reference solution.
JUDGE_USER_TEMPLATE_REF = (
    "[Instruction]\n"
    "Please act as an impartial judge and evaluate the quality of the response provided "
    "by an AI assistant to the user question displayed below. Your evaluation should "
    "consider correctness and helpfulness. You will be given a reference answer and the "
    "assistant's answer. Begin your evaluation by comparing the assistant's answer with "
    "the reference answer. Identify and correct any mistakes. Be as objective as possible. "
    "After providing your explanation, you must rate the response on a scale of 1 to 10 by "
    "strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n"
    "[Question]\n{question}\n\n"
    "[The Start of Reference Answer]\n{reference}\n[The End of Reference Answer]\n\n"
    "[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]"
)

RATING_RE = re.compile(r"\[\[\s*(\d+(?:\.\d+)?)\s*\]\]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def estimate_call_cost(judge_model: str) -> float:
    p = PRICING_USD_PER_MTOK[judge_model]
    return (EST_INPUT_TOKENS_PER_CALL  / 1_000_000) * p["input"] + \
           (EST_OUTPUT_TOKENS_PER_CALL / 1_000_000) * p["output"]


def actual_call_cost(judge_model: str, in_tok: int, out_tok: int) -> float:
    p = PRICING_USD_PER_MTOK[judge_model]
    return (in_tok / 1_000_000) * p["input"] + (out_tok / 1_000_000) * p["output"]


def cache_key(judge_model: str, question: str, answer: str, reference: str | None) -> str:
    h = hashlib.sha256()
    h.update(judge_model.encode())
    h.update(b"\x00")
    h.update(question.encode())
    h.update(b"\x00")
    h.update(answer.encode())
    h.update(b"\x00")
    h.update((reference or "").encode())
    return h.hexdigest()


def load_cache(cache_path: Path) -> dict[str, dict]:
    if not cache_path.exists():
        return {}
    out = {}
    for line in cache_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            out[row["key"]] = row
        except (json.JSONDecodeError, KeyError):
            continue
    return out


def append_cache(cache_path: Path, row: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def append_budget_log(log_path: Path, entry: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def parse_rating(text: str) -> float | None:
    m = RATING_RE.search(text)
    if m is None:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    if not (1.0 <= v <= 10.0):
        return None
    return v


# Categories that use the reference-aware judge prompt.
REF_CATEGORIES = {"math", "reasoning", "coding"}


def build_judge_messages(prompt: str, answer: str, category: str,
                         reference: str | None) -> list[dict]:
    use_ref = reference is not None and category in REF_CATEGORIES
    if use_ref:
        user = JUDGE_USER_TEMPLATE_REF.format(
            question=prompt, reference=reference, answer=answer)
    else:
        user = JUDGE_USER_TEMPLATE.format(question=prompt, answer=answer)
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user",   "content": user},
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--all", action="store_true",
                     help="Judge every (config, model) pair under --root.")
    grp.add_argument("--target", action="append", default=[],
                     metavar="CONFIG/MODEL",
                     help="Specific target like 'C0/phi3'. Repeatable.")
    grp.add_argument("--single", action="store_true",
                     help="Use --config and --model.")

    p.add_argument("--config", default=None)
    p.add_argument("--model",  default=None)
    p.add_argument("--root",   default="runs/mtbench")
    p.add_argument("--judge",  default="gpt-4o-mini",
                   choices=sorted(PRICING_USD_PER_MTOK.keys()))
    p.add_argument("--budget", type=float, default=None,
                   help="Hard USD cap. Default: env MTBENCH_BUDGET_USD or 5.0.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print cost preview only; no API calls.")
    p.add_argument("--yes", action="store_true",
                   help="Skip the [y/N] cost confirmation prompt.")
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


def resolve_targets(args) -> list[tuple[str, str]]:
    root = Path(args.root)
    if args.single:
        if not (args.config and args.model):
            sys.exit("--single requires both --config and --model.")
        return [(args.config, args.model)]
    if args.target:
        out = []
        for t in args.target:
            try:
                c, m = t.split("/", 1)
            except ValueError:
                sys.exit(f"Bad --target {t!r}, expected CONFIG/MODEL.")
            out.append((c, m))
        return out
    # --all
    targets = []
    if not root.exists():
        return []
    for cfg_dir in sorted(root.iterdir()):
        if not cfg_dir.is_dir() or cfg_dir.name == "judge_cache":
            continue
        for model_dir in sorted(cfg_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            if (model_dir / "generations.jsonl").exists():
                targets.append((cfg_dir.name, model_dir.name))
    return targets


def main():
    args = parse_args()

    # Budget resolution: CLI > env > default.
    if args.budget is not None:
        budget_cap = args.budget
    else:
        env_b = os.environ.get("MTBENCH_BUDGET_USD")
        budget_cap = float(env_b) if env_b else 5.0

    targets = resolve_targets(args)
    if not targets:
        sys.exit("No targets resolved. Run benchmark_mtbench.py first.")

    root        = Path(args.root)
    cache_path  = root / "judge_cache" / f"{args.judge}.jsonl"
    budget_log  = root / "budget.log"
    cache       = load_cache(cache_path)
    print(f"Judge cache: {len(cache)} cached entries at {cache_path}")

    # ----- Pre-flight cost preview -----
    plan = []  # list of (config, model, gens_path, judg_path, n_total, n_uncached)
    for cfg, model in targets:
        gens_path = root / cfg / model / "generations.jsonl"
        if not gens_path.exists():
            print(f"  SKIP {cfg}/{model}: {gens_path} missing")
            continue
        rows = [json.loads(l) for l in gens_path.read_text().splitlines() if l.strip()]
        n_total    = len(rows)
        n_uncached = sum(
            1 for r in rows
            if cache_key(args.judge, r["prompt"], r["response"],
                         r.get("reference")) not in cache
        )
        judg_path  = root / cfg / model / f"judgments-{args.judge}.jsonl"
        plan.append((cfg, model, gens_path, judg_path, n_total, n_uncached, rows))

    if not plan:
        sys.exit("No targets had generations.jsonl.")

    per_call    = estimate_call_cost(args.judge)
    total_calls = sum(p[5] for p in plan)
    est_total   = total_calls * per_call

    print(f"\nJudge model     : {args.judge}")
    print(f"Per-call est.   : ${per_call:.4f}  "
          f"({EST_INPUT_TOKENS_PER_CALL} in + {EST_OUTPUT_TOKENS_PER_CALL} out tok)")
    print(f"Targets         : {len(plan)}")
    for cfg, model, _, _, n_total, n_uncached, _ in plan:
        cached = n_total - n_uncached
        print(f"  {cfg}/{model:<8} {n_total:>3} prompts  "
              f"({cached:>3} cached, {n_uncached:>3} to call)")
    print(f"Total API calls : {total_calls}")
    print(f"Estimated cost  : ${est_total:.4f}")
    print(f"Budget cap      : ${budget_cap:.2f}")

    if args.dry_run:
        print("\n[--dry-run] exiting without API calls.")
        return

    if est_total > budget_cap:
        sys.exit(f"\nABORT: estimated cost ${est_total:.4f} exceeds budget "
                 f"${budget_cap:.2f}. Raise --budget or reduce targets.")

    if not args.yes:
        try:
            ans = input(f"\nProceed? [y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print("Aborted.")
            return

    # ----- Auth (read strictly from env) -----
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY environment variable is not set.\n"
                 "Set it in your shell (e.g. ~/.bashrc) before running:\n"
                 "  export OPENAI_API_KEY=...\n"
                 "Never commit the key or paste it into the codebase.")

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        sys.exit("ERROR: 'openai' package not installed. Run: pip install openai")

    client = OpenAI()  # picks up OPENAI_API_KEY from env

    # ----- Judge loop with running budget enforcement -----
    spent = 0.0
    for cfg, model, gens_path, judg_path, n_total, _, rows in plan:
        print(f"\n=== Judging {cfg}/{model} ({len(rows)} prompts) ===")
        judg_path.write_text("")  # truncate per-target output
        with judg_path.open("a") as fout:
            for i, r in enumerate(rows, 1):
                key = cache_key(args.judge, r["prompt"], r["response"],
                                r.get("reference"))
                if key in cache:
                    cached = cache[key]
                    rating  = cached["rating"]
                    judge_text = cached["judge_text"]
                    used_cache = True
                else:
                    if spent + per_call > budget_cap:
                        sys.exit(f"\nABORT: running cost ${spent:.4f} + next "
                                 f"call ${per_call:.4f} would exceed budget "
                                 f"${budget_cap:.2f}.")
                    messages = build_judge_messages(
                        r["prompt"], r["response"],
                        r.get("category", "unknown"), r.get("reference"))
                    t0 = time.time()
                    resp = client.chat.completions.create(
                        model=args.judge,
                        messages=messages,
                        temperature=args.temperature,
                        max_tokens=EST_OUTPUT_TOKENS_PER_CALL + 200,
                    )
                    dt = time.time() - t0
                    judge_text = resp.choices[0].message.content or ""
                    in_tok  = resp.usage.prompt_tokens     if resp.usage else EST_INPUT_TOKENS_PER_CALL
                    out_tok = resp.usage.completion_tokens if resp.usage else EST_OUTPUT_TOKENS_PER_CALL
                    cost = actual_call_cost(args.judge, in_tok, out_tok)
                    spent += cost
                    rating = parse_rating(judge_text)
                    cache_row = {
                        "key":         key,
                        "judge_model": args.judge,
                        "rating":      rating,
                        "judge_text":  judge_text,
                        "in_tok":      in_tok,
                        "out_tok":     out_tok,
                        "cost_usd":    round(cost, 6),
                    }
                    append_cache(cache_path, cache_row)
                    cache[key] = cache_row
                    append_budget_log(budget_log, {
                        "ts":          time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "judge":       args.judge,
                        "config":      cfg,
                        "model":       model,
                        "prompt_id":   r["prompt_id"],
                        "in_tok":      in_tok,
                        "out_tok":     out_tok,
                        "cost_usd":    round(cost, 6),
                        "spent_usd":   round(spent, 6),
                        "wall_s":      round(dt, 3),
                    })
                    used_cache = False

                fout.write(json.dumps({
                    "prompt_id":  r["prompt_id"],
                    "category":   r.get("category"),
                    "turn":       r.get("turn"),
                    "model":      model,
                    "config":     cfg,
                    "judge":      args.judge,
                    "rating":     rating,
                    "judge_text": judge_text,
                    "from_cache": used_cache,
                }) + "\n")
                fout.flush()
                tag = "cache" if used_cache else f"${spent:.3f}"
                rating_s = f"{rating:.1f}" if rating is not None else "  ? "
                print(f"  [{i:>3}/{len(rows)}] q{r['prompt_id']:<3} "
                      f"{r.get('category','?'):<14} rating={rating_s}  ({tag})")

        # Per-target summary
        ratings = [
            json.loads(l)["rating"]
            for l in judg_path.read_text().splitlines() if l.strip()
        ]
        valid   = [x for x in ratings if x is not None]
        if valid:
            print(f"  → {cfg}/{model}: mean={sum(valid)/len(valid):.3f}  "
                  f"(parsed {len(valid)}/{len(ratings)})")
        print(f"  Saved → {judg_path}")

    print(f"\nDone. Total spent this run: ${spent:.4f} of ${budget_cap:.2f}.")


if __name__ == "__main__":
    main()
