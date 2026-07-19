#!/usr/bin/env python3
"""
Statistical tests for the CV router vs FP16 and vs the best fixed method.

Produces, per model:
  1. Paired bootstrap (10k resamples) 95% CI of the mean per-prompt quality
     difference router-minus-C0 and router-minus-best-fixed, plus Wilcoxon
     signed-rank p-values (wilcox and pratt zero handling).
  2. Per-task decomposition of router - C0 (which tasks drive the delta,
     and each task's contribution to the overall mean).
  3. Metric-noise characterization: how often each compressed config scores
     above FP16 on the same prompt, pooled and per metric — the headroom a
     per-prompt selector could exploit from scoring noise alone.

Reads runs/p0/dataset/{model}.jsonl (CV picks regenerated in-process with the
exact cv_router_220 protocol); writes runs/p0/significance_tau{tau}.json.

Usage:
    python scripts/significance_tests.py            # tau=0.99
    python scripts/significance_tests.py --tau 0.95
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
P0_RUNS = ROOT / "runs" / "p0"
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_c6_classifier import (  # noqa: E402
    CANDIDATES, CONFIGS, featurize, load_dataset,
)
from cv_router_220 import cv_picks  # noqa: E402
from _common import assert_not_p0_output_path  # noqa: E402

MODELS = ["phi3", "llama3", "llama32_3b", "llama31_8b"]
NBOOT = 10_000


def wilcoxon_safe(d, zero_method):
    nz = d[d != 0]
    if len(nz) == 0:
        return float("nan"), 0
    try:
        res = wilcoxon(d, zero_method=zero_method, method="auto")
        return float(res.pvalue), len(nz)
    except ValueError:
        return float("nan"), len(nz)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tau", type=float, default=0.99)
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_path = P0_RUNS / f"significance_tau{args.tau}.json"
    assert_not_p0_output_path(out_path)

    out = {}
    for model in MODELS:
        rows = load_dataset(model)
        X = featurize(rows)
        picks = cv_picks(rows, X, args.tau, args.n_splits, args.seed)
        assert all(p is not None for p in picks)

        rq = np.array([r["scores"][c] for r, c in zip(rows, picks)], dtype=float)
        q = {c: np.array([r["scores"][c] for r in rows], dtype=float) for c in CONFIGS}
        tasks = [r["task"] for r in rows]
        metric_of = {r["task"]: r.get("metric", "") for r in rows}

        means_fixed = {c: float(q[c].mean()) for c in CONFIGS}
        best_fixed = max(CANDIDATES, key=lambda c: means_fixed[c])

        rng = np.random.default_rng(12345)
        idx = rng.integers(0, len(rows), size=(NBOOT, len(rows)))

        def paired(dvec):
            boot = dvec[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            pw, n_nz = wilcoxon_safe(dvec, "wilcox")
            pp, _ = wilcoxon_safe(dvec, "pratt")
            return {
                "mean_diff": float(dvec.mean()), "ci95": [float(lo), float(hi)],
                "p_boot": float(2 * min((boot <= 0).mean(), (boot >= 0).mean())),
                "p_wilcoxon_wilcox": pw, "p_wilcoxon_pratt": pp,
                "n_nonzero": n_nz,
                "win_tie_loss": [int((dvec > 0).sum()), int((dvec == 0).sum()),
                                 int((dvec < 0).sum())],
            }

        d_c0 = rq - q["C0"]
        d_bf = rq - q[best_fixed]

        per_task = {}
        for t in sorted(set(tasks)):
            m = np.array([tt == t for tt in tasks])
            per_task[t] = {
                "metric": metric_of[t],
                "n": int(m.sum()),
                "router_q": float(rq[m].mean()),
                "c0_q": float(q["C0"][m].mean()),
                "delta": float((rq[m] - q["C0"][m]).mean()),
                "contrib_to_overall": float((rq[m] - q["C0"][m]).sum() / len(rows)),
            }

        noise = {}
        for c in CANDIDATES:
            d = q[c] - q["C0"]
            noise[c] = {
                "pct_higher_than_c0": float((d > 0).mean()),
                "pct_tie": float((d == 0).mean()),
                "pct_lower": float((d < 0).mean()),
                "mean_diff": float(d.mean()),
            }
        dmax = np.max(np.stack([q[c] - q["C0"] for c in CANDIDATES]), axis=0)
        per_metric_beats = defaultdict(lambda: [0, 0])
        for i, t in enumerate(tasks):
            per_metric_beats[metric_of[t]][1] += 1
            if dmax[i] > 0:
                per_metric_beats[metric_of[t]][0] += 1

        out[model] = {
            "tau": args.tau,
            "router_cv_q": round(float(rq.mean()), 4),
            "fixed_means": {c: round(v, 4) for c, v in means_fixed.items()},
            "best_fixed": best_fixed,
            "router_vs_c0": paired(d_c0),
            "router_vs_best_fixed": paired(d_bf),
            "per_task": per_task,
            "noise_per_config": noise,
            "oracle_max_diff_mean": float(dmax.mean()),
            "pct_prompts_some_config_beats_c0": float((dmax > 0).mean()),
            "per_metric_some_config_beats_c0": {
                k: {"beats": v[0], "n": v[1], "pct": v[0] / v[1]}
                for k, v in per_metric_beats.items()},
        }
        rc0 = out[model]["router_vs_c0"]
        print(f"{model}: router={rq.mean():.4f} c0={means_fixed['C0']:.4f} "
              f"diff={rc0['mean_diff']:+.4f} CI[{rc0['ci95'][0]:+.4f},{rc0['ci95'][1]:+.4f}] "
              f"best_fixed={best_fixed}={means_fixed[best_fixed]:.4f}", flush=True)

    assert_not_p0_output_path(out_path)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
