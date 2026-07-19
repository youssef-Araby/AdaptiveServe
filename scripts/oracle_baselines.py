#!/usr/bin/env python3
"""
Oracle / random / length-only baselines vs the learned CV router.

Answers the questions a routing-paper reviewer asks first:
  1. ORACLE iso-quality router  — per prompt, max TRUE compression s.t.
     TRUE q >= tau * TRUE q(C0); fallback argmax true q. Upper bound on
     what any router over this pool can deliver.
  2. ORACLE-quality router      — per prompt argmax TRUE q over C1..C5.
  3. RANDOM router              — uniform over C1..C5 (exact expectation +
     Monte-Carlo check of the harmonic-mean CR).
  4. LENGTH-ONLY learned router — the exact cv_router_220 CV protocol
     restricted to length features; shows how much the non-length
     features earn.
  5. Full-feature CV router reproduction, cross-checked against
     runs/p0/cv_router_220_tau{tau}.json when present.

Reads runs/p0/dataset/{model}.jsonl; writes
runs/p0/oracle_baselines_tau{tau}.json.

Usage:
    python scripts/oracle_baselines.py            # tau=0.99
    python scripts/oracle_baselines.py --tau 0.95
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P0_RUNS = ROOT / "runs" / "p0"
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_c6_classifier import (  # noqa: E402
    CANDIDATES, FEATURE_KEYS, aggregate, featurize, load_dataset,
)
from cv_router_220 import cv_picks, fixed_on_full  # noqa: E402
from _common import assert_not_p0_output_path  # noqa: E402

MODELS = ["phi3", "llama3", "llama32_3b", "llama31_8b"]


def oracle_iso_picks(rows, tau):
    """Max TRUE compression s.t. TRUE q >= tau * TRUE q(C0); fallback argmax true q."""
    picks, n_fallback = [], 0
    for r in rows:
        thresh = tau * r["scores"]["C0"]
        best_c, best_cr = None, -1.0
        for c in CANDIDATES:
            if r["scores"][c] >= thresh and r["compression"][c] > best_cr:
                best_c, best_cr = c, r["compression"][c]
        if best_c is None:
            n_fallback += 1
            best_c = max(CANDIDATES,
                         key=lambda c: (r["scores"][c], r["compression"][c]))
        picks.append(best_c)
    return picks, n_fallback


def oracle_q_picks(rows, tie="cr"):
    """Argmax TRUE q over C1..C5. tie='cr': ties broken toward higher
    compression; tie='order': first max in C1..C5 order."""
    picks = []
    for r in rows:
        if tie == "cr":
            best_c = max(CANDIDATES,
                         key=lambda c: (r["scores"][c], r["compression"][c]))
        else:
            best_c = max(CANDIDATES, key=lambda c: r["scores"][c])
        picks.append(best_c)
    return picks


def random_router(rows, n_mc=2000, seed=0):
    """Uniform over C1..C5.
    Quality: exact expectation = mean over prompts x configs.
    CR: harmonic mean from per-prompt EXPECTED inverse compression
        (expected KV-memory footprint), N / sum_i mean_c(1/CR_ic),
        plus a Monte-Carlo check of the repo-style harmonic mean."""
    qs  = np.array([[r["scores"][c] for c in CANDIDATES] for r in rows])
    crs = np.array([[r["compression"][c] for c in CANDIDATES] for r in rows])
    q_exp  = float(qs.mean())
    cr_exp = float(len(rows) / (1.0 / crs).mean(axis=1).sum())
    rng = np.random.default_rng(seed)
    hm_draws, q_draws = [], []
    for _ in range(n_mc):
        idx = rng.integers(0, len(CANDIDATES), size=len(rows))
        hm_draws.append(len(rows) / (1.0 / crs[np.arange(len(rows)), idx]).sum())
        q_draws.append(qs[np.arange(len(rows)), idx].mean())
    return {"q": q_exp, "cr": cr_exp,
            "cr_mc": float(np.mean(hm_draws)), "cr_mc_sd": float(np.std(hm_draws)),
            "q_mc": float(np.mean(q_draws)), "q_mc_sd": float(np.std(q_draws))}


def agg_qcr(rows, picks):
    a = aggregate(rows, picks)
    return a["longbench_avg"], a["longbench_avg_compression_ratio"], dict(Counter(picks))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tau", type=float, default=0.99)
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tau = args.tau
    out_path = P0_RUNS / f"oracle_baselines_tau{tau}.json"
    assert_not_p0_output_path(out_path)

    cv_json = P0_RUNS / f"cv_router_220_tau{tau}.json"
    cv_ref = json.loads(cv_json.read_text()) if cv_json.exists() else None

    len_idx_both = [FEATURE_KEYS.index("seq_len_tokens"),
                    FEATURE_KEYS.index("seq_len_chars")]
    len_idx_tok = [FEATURE_KEYS.index("seq_len_tokens")]

    out = {}
    for model in MODELS:
        rows = load_dataset(model)
        X = featurize(rows)
        n = len(rows)

        p_oiso, n_fb = oracle_iso_picks(rows, tau)
        q_oiso, cr_oiso, picks_oiso = agg_qcr(rows, p_oiso)

        q_oq_cr, cr_oq_cr, picks_oq = agg_qcr(rows, oracle_q_picks(rows, "cr"))
        q_oq_or, cr_oq_or, _ = agg_qcr(rows, oracle_q_picks(rows, "order"))

        rnd = random_router(rows)

        p_len2 = cv_picks(rows, X[:, len_idx_both], tau, args.n_splits, args.seed)
        q_len2, cr_len2, picks_len2 = agg_qcr(rows, p_len2)
        p_len1 = cv_picks(rows, X[:, len_idx_tok], tau, args.n_splits, args.seed)
        q_len1, cr_len1, _ = agg_qcr(rows, p_len1)

        p_full = cv_picks(rows, X, tau, args.n_splits, args.seed)
        q_full, cr_full, picks_full = agg_qcr(rows, p_full)

        fixed = fixed_on_full(rows)
        best_fixed = max(CANDIDATES, key=lambda c: fixed[c]["q"])

        out[model] = {
            "n": n,
            "tau": tau,
            "oracle_iso": {"q": q_oiso, "cr": cr_oiso, "picks": picks_oiso,
                           "fallback_frac": round(n_fb / n, 4)},
            "oracle_q_tiecr": {"q": q_oq_cr, "cr": cr_oq_cr, "picks": picks_oq},
            "oracle_q_tieorder": {"q": q_oq_or, "cr": cr_oq_or},
            "random": {k: round(v, 4) for k, v in rnd.items()},
            "len_only_2feat": {"q": q_len2, "cr": cr_len2, "picks": picks_len2},
            "len_only_tokens": {"q": q_len1, "cr": cr_len1},
            "full_router_cv": {"q": q_full, "cr": cr_full, "picks": picks_full},
            "best_fixed": {"name": best_fixed, **fixed[best_fixed]},
            "c0": fixed["C0"],
            "fixed_full": fixed,
        }
        if cv_ref is not None and model in cv_ref:
            ref = cv_ref[model]["router_cv"]
            out[model]["cv_json_crosscheck"] = {
                "q": ref["q"], "cr": ref["cr"],
                "matches": bool(abs(q_full - ref["q"]) < 5e-4
                                and abs(cr_full - ref["cr"]) < 5e-3),
            }
        print(f"{model}: oracle_iso={q_oiso:.4f}/{cr_oiso:.2f}x  "
              f"router={q_full:.4f}/{cr_full:.2f}x  "
              f"random={rnd['q']:.4f}/{rnd['cr']:.2f}x  "
              f"len_only={q_len1:.4f}/{cr_len1:.2f}x")

    assert_not_p0_output_path(out_path)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
