#!/usr/bin/env python3
"""
Stratified k-fold cross-validation of the C6 router. Each prompt is routed
exactly once by a model that did NOT train on it; aggregate over all 220
prompts. Output is apples-to-apples comparable to the fixed-method
baselines computed on the full 220-prompt set.

Addresses prof comment #3 honestly: instead of evaluating the router on a
single 66-prompt held-out subset (high variance, biased by seed), give every
prompt a turn as held-out, then aggregate.
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_c6_classifier import (
    CONFIGS, CANDIDATES, load_dataset, featurize, fit_regressors,
    predict_q, route, c0_floor_from_train, train_mean_compression, aggregate,
)

MODELS = ["phi3", "llama3", "llama32_3b", "llama31_8b"]


def cv_picks(rows, X, tau: float, n_splits: int = 10, seed: int = 0,
             use_floor: bool = True):
    """Stratified k-fold over (task) label. Each prompt picked once."""
    labels = np.array([r["task"] for r in rows])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    picks = [None] * len(rows)
    for tr_idx, te_idx in skf.split(np.zeros(len(rows)), labels):
        rows_tr = [rows[i] for i in tr_idx]
        rows_te = [rows[i] for i in te_idx]
        models = fit_regressors(X[tr_idx], rows_tr)
        q_pred = predict_q(models, X[te_idx])
        floor = c0_floor_from_train(rows_tr) if use_floor else 0.0
        cr_tr = train_mean_compression(rows_tr)
        for i, c in zip(te_idx, route(rows_te, q_pred, tau, c0_floor=floor,
                                      cr_rank=cr_tr)):
            picks[i] = c
    return picks


def fixed_on_full(rows):
    out = {}
    for c in CONFIGS:
        qs = [r["scores"][c] for r in rows]
        crs = [r["compression"][c] for r in rows if r["compression"][c] > 0]
        out[c] = {
            "q": round(sum(qs) / len(qs), 4),
            "cr": round(statistics.harmonic_mean(crs), 3),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tau", type=float, default=0.99)
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"\n=== CV router (tau={args.tau}, {args.n_splits}-fold stratified by task) ===\n")

    summary = {}
    for model in MODELS:
        rows = load_dataset(model)
        X = featurize(rows)
        picks = cv_picks(rows, X, args.tau, args.n_splits, args.seed)
        agg = aggregate(rows, picks)
        fixed = fixed_on_full(rows)

        # Pareto comparison
        rq, rcr = agg["longbench_avg"], agg["longbench_avg_compression_ratio"]
        c_pareto = {}
        for c in CONFIGS:
            cq, ccr = fixed[c]["q"], fixed[c]["cr"]
            dom_router = (rq >= cq and rcr >= ccr and (rq > cq or rcr > ccr))
            dom_fixed  = (cq >= rq and ccr >= rcr and (cq > rq or ccr > rcr))
            c_pareto[c] = "router_dominates" if dom_router else (
                          "router_dominated"  if dom_fixed  else "non_dominated")

        summary[model] = {
            "router_cv": {"q": round(rq, 4), "cr": round(rcr, 3),
                          "picks": agg.get("picks", {})},
            "fixed_full": fixed,
            "pareto_vs_router": c_pareto,
        }

        print(f"{model}")
        print(f"  Router (10-fold CV, all 220 prompts): q={rq:.4f}, cr={rcr:.3f}x")
        print(f"  Fixed methods on full 220 (apples-to-apples):")
        for c in CONFIGS:
            f = fixed[c]
            verdict = c_pareto[c]
            print(f"    {c}: q={f['q']:.4f}, cr={f['cr']:.3f}x   "
                  f"vs router: {verdict}")
        gain_c0 = (rq - fixed["C0"]["q"]) / fixed["C0"]["q"] * 100
        print(f"  Router vs FP16 (paired, fair): "
              f"Δq = {rq - fixed['C0']['q']:+.4f} ({gain_c0:+.1f}%), "
              f"compression: {rcr:.2f}x vs 1.00x")
        print()

    out = ROOT / "runs" / f"cv_router_220_tau{args.tau}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out}\n")


if __name__ == "__main__":
    main()
