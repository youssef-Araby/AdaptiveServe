#!/usr/bin/env python3
"""
BEST-Route (pair-wise, n=1) on MT-Bench.

Implements the binary pair-wise router from BEST-Route (Ding et al., ICML 2025,
arXiv:2506.22716, Section 4.2 "Pair-Wise Router"), without the best-of-n
extension. This degenerates to HybridLLM (Ding et al., ICLR 2024). Faithful to
the paper's training and inference recipe up to that point.

Setup mismatch we explicitly account for:
  - Their reference model is GPT-4o (paid). Ours is LLaMA-3-8B (the bigger of
    the two we serve). Phi-3-mini is the small model.
  - Their quality signal is armoRM. Ours is gpt-4o-as-judge (1-10), already
    cached in runs/mtbench/C0/{model}/judgments-gpt-4o.jsonl.
  - 80 prompts only -> Leave-One-Out CV is the only honest choice.

Routing rule (per query q):
    p(q) = Pr[ rating(phi3, q) >= rating(llama3, q) - margin ]
    if p(q) >= t : route to phi3 (cheap, fast)
    else        : route to llama3 (slow, higher quality on average)

The threshold t sweeps the cost-quality Pareto curve. We report it against six
policies (always-small, always-large, random, oracle, category-only baseline,
BEST-Route) at multiple operating points.

Output: runs/mtbench/bestroute/{result.json, pareto.png}

Usage:
    python scripts/benchmark_bestroute.py
    python scripts/benchmark_bestroute.py --margin 0.0 --classifier hgb
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

from _common import MODELS, extract_prompt_features


JUDGE = "gpt-4o"   # ground-truth label source (already cached)
# Defaults overridden by CLI in main(); kept as module-level for backwards-compat.
SMALL = "phi3"
LARGE = "llama3"


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def load_dataset(root: Path):
    """Build the per-prompt routing dataset from cached generations + judgments."""
    rows = {}
    for m in (SMALL, LARGE):
        gens = [json.loads(l) for l in (root / "C0" / m / "generations.jsonl").read_text().splitlines() if l.strip()]
        judg = [json.loads(l) for l in (root / "C0" / m / f"judgments-{JUDGE}.jsonl").read_text().splitlines() if l.strip()]
        gens_by = {r["prompt_id"]: r for r in gens}
        judg_by = {r["prompt_id"]: r for r in judg}
        for pid in gens_by:
            r = rows.setdefault(pid, {"prompt_id": pid})
            g = gens_by[pid]
            j = judg_by[pid]
            r["category"]   = g["category"]
            r["prompt"]     = g["prompt"]
            r[f"rating_{m}"]    = j["rating"]
            r[f"latency_{m}"]   = g["total_time_s"]
            r[f"out_tokens_{m}"]= g["response_tokens"]

    out = []
    for pid in sorted(rows):
        r = rows[pid]
        if r.get(f"rating_{SMALL}") is None or r.get(f"rating_{LARGE}") is None:
            continue
        out.append(r)
    return out


def featurize(rows, tokenizer):
    """Numeric feature matrix + side info."""
    cats = sorted({r["category"] for r in rows})
    cat_idx = {c: i for i, c in enumerate(cats)}

    base_feat_keys = [
        "seq_len_tokens", "seq_len_chars", "token_entropy",
        "gzip_ratio", "unique_token_ratio", "newline_density",
    ]

    X = []
    for r in rows:
        f = extract_prompt_features(r["prompt"], tokenizer)
        feats = [float(f[k]) for k in base_feat_keys]
        # one-hot category (small dim, 8 cats)
        oh = [0.0] * len(cats)
        oh[cat_idx[r["category"]]] = 1.0
        X.append(feats + oh)
    return np.array(X, dtype=np.float32), cats, base_feat_keys


def build_labels(rows, margin: float):
    """y[i]=1 -> small wins (route to phi3); 0 -> large strictly preferred."""
    y = np.array([
        1 if r[f"rating_{SMALL}"] >= r[f"rating_{LARGE}"] - margin else 0
        for r in rows
    ], dtype=np.int64)
    q_small = np.array([r[f"rating_{SMALL}"]  for r in rows], dtype=np.float64)
    q_large = np.array([r[f"rating_{LARGE}"]  for r in rows], dtype=np.float64)
    t_small = np.array([r[f"latency_{SMALL}"] for r in rows], dtype=np.float64)
    t_large = np.array([r[f"latency_{LARGE}"] for r in rows], dtype=np.float64)
    return y, q_small, q_large, t_small, t_large


# ---------------------------------------------------------------------------
# Routing policies
# ---------------------------------------------------------------------------

def evaluate_route(route, q_small, q_large, t_small, t_large):
    """route[i]=1 -> small, 0 -> large. Returns mean quality & total latency."""
    q = np.where(route == 1, q_small, q_large)
    t = np.where(route == 1, t_small, t_large)
    return float(q.mean()), float(t.sum()), int(route.sum())


def policy_always(n, val: int):
    return np.full(n, val, dtype=np.int64)


def policy_random(n, p_small: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    return (rng.random(n) < p_small).astype(np.int64)


def policy_oracle(q_small, q_large, t_small, t_large, margin: float):
    """Per-prompt best: prefer small unless large is strictly better by margin."""
    return (q_small >= q_large - margin).astype(np.int64)


def policy_category_baseline(rows, train_mask, q_small, q_large):
    """Per-category majority winner from the training fold."""
    by_cat_small_wins = defaultdict(int)
    by_cat_total      = defaultdict(int)
    for i, r in enumerate(rows):
        if not train_mask[i]:
            continue
        by_cat_total[r["category"]] += 1
        if q_small[i] >= q_large[i]:
            by_cat_small_wins[r["category"]] += 1
    pick = {}
    for c in by_cat_total:
        pick[c] = 1 if by_cat_small_wins[c] / by_cat_total[c] >= 0.5 else 0
    out = np.zeros(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        out[i] = pick.get(r["category"], 0)  # fallback: large
    return out


# ---------------------------------------------------------------------------
# BEST-Route classifier (LOO cross-validated probabilities)
# ---------------------------------------------------------------------------

def loo_probs(X, y, classifier: str):
    """Leave-one-out P(small wins) for every i."""
    n = len(y)
    probs = np.zeros(n, dtype=np.float64)
    loo = LeaveOneOut()
    for tr, te in loo.split(X):
        Xtr, ytr = X[tr], y[tr]
        # Edge case: a fold may have only one class -> fall back to base rate.
        if len(set(ytr)) < 2:
            probs[te[0]] = float(ytr.mean())
            continue
        if classifier == "logreg":
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr)
            Xte_s = scaler.transform(X[te])
            clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
            clf.fit(Xtr_s, ytr)
            probs[te[0]] = clf.predict_proba(Xte_s)[0, 1]
        elif classifier == "hgb":
            clf = HistGradientBoostingClassifier(
                max_iter=200, max_depth=3, learning_rate=0.05, random_state=0
            )
            clf.fit(Xtr, ytr)
            probs[te[0]] = clf.predict_proba(X[te])[0, 1]
        else:
            raise ValueError(classifier)
    return probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root",       default="runs/mtbench")
    p.add_argument("--margin",     type=float, default=0.0,
                   help="Quality margin: small wins if rating_small >= rating_large - margin.")
    p.add_argument("--classifier", default="logreg", choices=["logreg", "hgb"])
    p.add_argument("--out",        default=None,
                   help="output dir (default: runs/mtbench/bestroute/{small}_vs_{large})")
    p.add_argument("--small",      default="phi3",
                   help="small/cheap model key in MODELS (default phi3)")
    p.add_argument("--large",      default="llama3",
                   help="large/strong model key in MODELS (default llama3)")
    return p.parse_args()


def main():
    args = parse_args()
    # Bind module-level constants from CLI for the rest of the module.
    global SMALL, LARGE
    SMALL = args.small
    LARGE = args.large
    if SMALL not in MODELS or LARGE not in MODELS:
        raise SystemExit(f"unknown model key(s); valid: {list(MODELS)}")
    root = Path(args.root)
    out  = Path(args.out) if args.out else Path("runs/mtbench/bestroute") / f"{SMALL}_vs_{LARGE}"
    out.mkdir(parents=True, exist_ok=True)

    rows = load_dataset(root)
    n = len(rows)
    print(f"\nLoaded {n} prompts with both {SMALL} and {LARGE} ratings + latencies.")

    # Tokenizer for feature extraction (use small model -- features are tokenizer-relative
    # but the choice is only a constant scaling on seq_len_tokens; downstream classifier
    # is invariant to that under z-scoring).
    tok = AutoTokenizer.from_pretrained(MODELS[SMALL])

    X, cats, feat_keys = featurize(rows, tok)
    y, q_s, q_l, t_s, t_l = build_labels(rows, margin=args.margin)

    print(f"  base_rate (small-wins, margin={args.margin}): {y.mean():.3f}")
    print(f"  category counts: {dict(Counter(r['category'] for r in rows))}")
    print(f"  always-small:  Q={q_s.mean():.3f}  T={t_s.sum():.1f}s")
    print(f"  always-large:  Q={q_l.mean():.3f}  T={t_l.sum():.1f}s")

    # Sanity: per-category small-wins share
    print(f"\nPer-category small-win rate (margin={args.margin}):")
    for c in sorted(set(r["category"] for r in rows)):
        idx = [i for i, r in enumerate(rows) if r["category"] == c]
        s = y[idx].mean()
        print(f"  {c:<12} n={len(idx):>2}  small_wins={s:.2f}  "
              f"q_small={q_s[idx].mean():.2f}  q_large={q_l[idx].mean():.2f}")

    # ----- BEST-Route: LOO cross-validated probabilities -----
    print(f"\nTraining BEST-Route ({args.classifier}, LOO CV over n={n})...")
    p_small = loo_probs(X, y, args.classifier)

    # ----- Sweep threshold for Pareto curve -----
    thresholds = np.linspace(0.0, 1.0, 21)
    pareto = []
    for t in thresholds:
        route = (p_small >= t).astype(np.int64)
        Q, T, n_small = evaluate_route(route, q_s, q_l, t_s, t_l)
        pareto.append({"threshold": float(t), "quality": Q, "latency_s": T,
                       "frac_small": n_small / n})

    # ----- Reference policies -----
    ref = {}
    Q, T, ns = evaluate_route(policy_always(n, 1), q_s, q_l, t_s, t_l)
    ref["always_small"] = {"quality": Q, "latency_s": T, "frac_small": 1.0}
    Q, T, ns = evaluate_route(policy_always(n, 0), q_s, q_l, t_s, t_l)
    ref["always_large"] = {"quality": Q, "latency_s": T, "frac_small": 0.0}
    Q, T, ns = evaluate_route(policy_random(n, p_small=0.5), q_s, q_l, t_s, t_l)
    ref["random_50_50"] = {"quality": Q, "latency_s": T, "frac_small": ns / n}
    Q, T, ns = evaluate_route(policy_oracle(q_s, q_l, t_s, t_l, args.margin),
                              q_s, q_l, t_s, t_l)
    ref["oracle"] = {"quality": Q, "latency_s": T, "frac_small": ns / n}

    # Category baseline: LOO version
    cat_route = np.zeros(n, dtype=np.int64)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        r_full = policy_category_baseline(rows, mask, q_s, q_l)
        cat_route[i] = r_full[i]
    Q, T, ns = evaluate_route(cat_route, q_s, q_l, t_s, t_l)
    ref["category_loo"] = {"quality": Q, "latency_s": T, "frac_small": ns / n}

    # ----- Print Pareto table -----
    print(f"\n{'policy':<24}{'Q':>8}{'T(s)':>10}{'frac_small':>12}")
    for k, v in ref.items():
        print(f"{k:<24}{v['quality']:>8.3f}{v['latency_s']:>10.1f}{v['frac_small']:>12.2f}")

    print(f"\n{'BEST-Route threshold sweep':-^54}")
    print(f"{'t':>6}{'Q':>8}{'T(s)':>10}{'frac_small':>12}{'cost_red%':>11}{'q_drop%':>10}")
    base_T = ref["always_large"]["latency_s"]
    base_Q = ref["always_large"]["quality"]
    for p in pareto:
        cost_red = 100.0 * (1 - p["latency_s"] / base_T)
        q_drop   = 100.0 * (base_Q - p["quality"]) / max(base_Q, 1e-9)
        print(f"{p['threshold']:>6.2f}{p['quality']:>8.3f}{p['latency_s']:>10.1f}"
              f"{p['frac_small']:>12.2f}{cost_red:>10.1f}%{q_drop:>9.1f}%")

    # ----- Headline: BEST-Route at iso-quality vs always-large -----
    # Find the operating point that loses the least latency at quality drop <= 1%.
    valid = [p for p in pareto if (base_Q - p["quality"]) / max(base_Q, 1e-9) <= 0.01]
    if valid:
        best = min(valid, key=lambda p: p["latency_s"])
        cost_red = 100.0 * (1 - best["latency_s"] / base_T)
        q_drop   = 100.0 * (base_Q - best["quality"]) / max(base_Q, 1e-9)
        print(f"\n[iso-quality, drop ≤ 1%] BEST-Route t={best['threshold']:.2f}  "
              f"Q={best['quality']:.3f}  T={best['latency_s']:.1f}s  "
              f"cost_red={cost_red:.1f}%  q_drop={q_drop:.1f}%  "
              f"frac_small={best['frac_small']:.2f}")

    # ----- Save results -----
    out_path = out / "result.json"
    out_path.write_text(json.dumps({
        "n_prompts":  n,
        "judge":      JUDGE,
        "small":      SMALL,
        "large":      LARGE,
        "margin":     args.margin,
        "classifier": args.classifier,
        "feature_keys": feat_keys + [f"cat_{c}" for c in cats],
        "base_rate_small_wins": float(y.mean()),
        "reference_policies": ref,
        "best_route_pareto":   pareto,
    }, indent=2))
    print(f"\nSaved → {out_path}")

    # ----- Pareto plot -----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        Ts = [p["latency_s"] for p in pareto]
        Qs = [p["quality"]   for p in pareto]
        ax.plot(Ts, Qs, "-o", color="C0", label="BEST-Route (n=1)")
        for k, v in ref.items():
            color = {"always_small": "C1", "always_large": "C2",
                     "oracle":       "C3", "random_50_50": "gray",
                     "category_loo": "C4"}.get(k, "k")
            ax.scatter([v["latency_s"]], [v["quality"]],
                       marker="*" if k == "oracle" else "s",
                       s=130 if k == "oracle" else 70, color=color,
                       edgecolor="black", zorder=5, label=k)
        ax.set_xlabel("Total wall latency (s)  ←  cheaper")
        ax.set_ylabel("Mean gpt-4o judge rating  →  better")
        ax.set_title(f"BEST-Route pair-wise router on MT-Bench (n={n}, margin={args.margin})")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        plot_path = out / "pareto.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=130)
        print(f"Saved → {plot_path}")
    except Exception as e:
        print(f"  [plot skipped: {e}]")


if __name__ == "__main__":
    main()
