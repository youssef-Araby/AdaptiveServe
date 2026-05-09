#!/usr/bin/env python3
"""
C6 — feature-based per-prompt KV-cache router.

Trains 6 regressors (one per config) on prompt-intrinsic features to predict
the per-prompt LongBench score. At inference, picks the config with the
highest compression whose predicted score is at least ``tau`` * predicted
score of C0 (iso-quality routing).

Reads the joined dataset built by ``build_dataset.py`` (which already contains
each prompt's features and the measured score under every config). Therefore
this script does NOT call the model — the routing decision is a pure CPU step,
making per-request overhead negligible relative to LLM inference.

Two evaluations are reported and saved:
  - LOTO  : leave-one-task-out CV (out-of-distribution by task family)
  - Split : single random 70/30 split (in-distribution per-workload calibration)

Top-level results.json fields mirror benchmark_c{0..5} so plotting tools can
treat C6 as just another config. Aggregate scores at the top level use the
LOTO predictions; the random-split summary is included under ``eval_split``.

Output:
  runs/C6/{model}/results.json
  runs/C6/{model}/per_prompt.jsonl     (one row per prompt with chosen config)

Usage:
  python scripts/benchmark_c6_classifier.py --model llama3 --tau 0.99
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import harmonic_mean, mean

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

REPO     = Path(__file__).resolve().parents[1]
CONFIGS  = ["C0", "C1", "C2", "C3", "C4", "C5"]
FEATURE_KEYS = [
    "seq_len_tokens", "seq_len_chars", "token_entropy",
    "gzip_ratio", "unique_token_ratio", "question_position",
    "newline_density",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset(model: str) -> list[dict]:
    path = REPO / "runs" / "dataset" / f"{model}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing dataset: {path}\n  run scripts/build_dataset.py first")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def featurize(rows: list[dict]) -> np.ndarray:
    X = np.zeros((len(rows), len(FEATURE_KEYS)), dtype=np.float32)
    for i, r in enumerate(rows):
        f = r["features"]
        for j, k in enumerate(FEATURE_KEYS):
            v = f.get(k)
            X[i, j] = -1.0 if v is None else float(v)
    return X


def fit_regressors(X_tr: np.ndarray, rows_tr: list[dict]):
    """One HistGradientBoosting regressor per config, predicting q[c]."""
    models = {}
    for c in CONFIGS:
        y = np.array([r["scores"][c] for r in rows_tr], dtype=np.float32)
        sc = StandardScaler()
        Xs = sc.fit_transform(X_tr)
        reg = HistGradientBoostingRegressor(
            max_iter=300, max_depth=4, learning_rate=0.05, random_state=0
        )
        reg.fit(Xs, y)
        models[c] = (sc, reg)
    return models


def predict_q(models, X: np.ndarray) -> dict[str, np.ndarray]:
    return {c: reg.predict(sc.transform(X)) for c, (sc, reg) in models.items()}


def route(rows: list[dict], q_pred: dict[str, np.ndarray], tau: float) -> list[str]:
    """Pick the config with the highest measured compression whose predicted
    quality is at least ``tau`` * predicted quality of C0."""
    picks: list[str] = []
    for i, r in enumerate(rows):
        thresh = tau * q_pred["C0"][i]
        best_c, best_cr = "C0", r["compression"]["C0"]
        for c in CONFIGS:
            if q_pred[c][i] >= thresh and r["compression"][c] > best_cr:
                best_c, best_cr = c, r["compression"][c]
        picks.append(best_c)
    return picks


def aggregate(rows: list[dict], picks: list[str]) -> dict:
    """Aggregate per-task scores and overall metrics."""
    by_task: dict[str, list[tuple[float, str, float]]] = defaultdict(list)
    metric_of: dict[str, str] = {}
    for r, c in zip(rows, picks):
        # Per-prompt score under chosen config and the metric name from C0 row.
        by_task[r["task"]].append((r["scores"][c], c, r["compression"][c]))
        metric_of.setdefault(r["task"], r.get("metric", ""))

    longbench: dict[str, dict] = {}
    for task, items in by_task.items():
        scores = [s for s, _, _ in items]
        crs    = [cr for _, _, cr in items]
        pick_dist = Counter(c for _, c, _ in items)
        longbench[task] = {
            "score":       round(mean(scores), 4),
            "n":           len(scores),
            "compression": round(harmonic_mean(crs), 3),
            "picks":       dict(pick_dist),
        }

    overall_q  = mean(r["scores"][c]      for r, c in zip(rows, picks))
    overall_cr = harmonic_mean(r["compression"][c] for r, c in zip(rows, picks))
    return {
        "longbench":                       longbench,
        "longbench_avg":                   round(overall_q, 4),
        "longbench_avg_compression_ratio": round(overall_cr, 3),
        "picks":                           dict(Counter(picks)),
    }


def loto_picks(rows: list[dict], X: np.ndarray, tau: float) -> list[str]:
    tasks = sorted({r["task"] for r in rows})
    picks: list[str | None] = [None] * len(rows)
    for held in tasks:
        tr = [i for i, r in enumerate(rows) if r["task"] != held]
        te = [i for i, r in enumerate(rows) if r["task"] == held]
        models = fit_regressors(X[tr], [rows[i] for i in tr])
        q_pred = predict_q(models, X[te])
        rows_te = [rows[i] for i in te]
        for i, c in zip(te, route(rows_te, q_pred, tau)):
            picks[i] = c
    return picks  # type: ignore[return-value]


def split_picks(rows: list[dict], X: np.ndarray, tau: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    n_tr = int(0.7 * len(rows))
    tr, te = idx[:n_tr].tolist(), idx[n_tr:].tolist()
    models = fit_regressors(X[tr], [rows[i] for i in tr])
    q_pred = predict_q(models, X[te])
    rows_te = [rows[i] for i in te]
    picks_te = route(rows_te, q_pred, tau)
    return rows_te, picks_te


def measure_overhead_us(models, X: np.ndarray, n_rep: int = 100) -> float:
    """Mean per-prompt routing latency (microseconds) including all 6 predicts."""
    t0 = time.perf_counter()
    for _ in range(n_rep):
        predict_q(models, X)
    elapsed = time.perf_counter() - t0
    return (elapsed / n_rep / len(X)) * 1e6


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llama3", "phi3"])
    ap.add_argument("--tau", type=float, default=0.99,
                    help="iso-quality threshold (predicted q[c] >= tau * predicted q[C0])")
    args = ap.parse_args()

    rows = load_dataset(args.model)
    X    = featurize(rows)
    print(f"\n=== C6 router  model={args.model}  tau={args.tau}  "
          f"features={len(FEATURE_KEYS)}  n={len(rows)} ===\n")

    # ---- LOTO (OOD by task) -- headline numbers
    picks_loto = loto_picks(rows, X, args.tau)
    agg_loto   = aggregate(rows, picks_loto)
    print(f"LOTO (OOD): q={agg_loto['longbench_avg']:.4f}  "
          f"cr={agg_loto['longbench_avg_compression_ratio']:.2f}x  "
          f"picks={agg_loto['picks']}")

    # ---- Random 70/30 split (in-dist)
    rows_te, picks_te = split_picks(rows, X, args.tau, seed=0)
    agg_split = aggregate(rows_te, picks_te)
    print(f"Split    : q={agg_split['longbench_avg']:.4f}  "
          f"cr={agg_split['longbench_avg_compression_ratio']:.2f}x  "
          f"picks={agg_split['picks']}  (n_test={len(rows_te)})")

    # ---- Routing overhead
    full_models = fit_regressors(X, rows)
    overhead_us = measure_overhead_us(full_models, X)
    print(f"Routing overhead: {overhead_us:.2f} us / prompt "
          f"(features-already-extracted; sklearn predict only)")

    # ---- Save
    results = {
        "config": "C6",
        "method": "per-prompt feature-based router (HistGradientBoosting regression)",
        "model":  args.model,
        "tau":    args.tau,
        "features": FEATURE_KEYS,
        "n_features": len(FEATURE_KEYS),
        "router_overhead_us_per_prompt": round(overhead_us, 2),

        # Top-level fields mirror C0..C5 using the LOTO (honest OOD) headline.
        "longbench":                       agg_loto["longbench"],
        "longbench_avg":                   agg_loto["longbench_avg"],
        "longbench_avg_compression_ratio": agg_loto["longbench_avg_compression_ratio"],
        "picks":                           agg_loto["picks"],

        # In-distribution random-split summary (workload-calibrated regime).
        "eval_split_70_30": {
            "n_test":                          len(rows_te),
            "longbench_avg":                   agg_split["longbench_avg"],
            "longbench_avg_compression_ratio": agg_split["longbench_avg_compression_ratio"],
            "picks":                           agg_split["picks"],
        },
    }

    out_dir = REPO / "runs" / "C6" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    # Also keep a tau-tagged snapshot so a tau sweep doesn't overwrite itself
    # (used by scripts/plot_pareto.py).
    (out_dir / f"results_tau{args.tau}.json").write_text(json.dumps(results, indent=2))

    pp_path = out_dir / "per_prompt.jsonl"
    with pp_path.open("w") as f:
        for r, c in zip(rows, picks_loto):
            f.write(json.dumps({
                "config":      "C6",
                "model":       args.model,
                "task":        r["task"],
                "sample_idx":  r["sample_idx"],
                "pick":        c,
                "score":       r["scores"][c],
                "compression": r["compression"][c],
            }) + "\n")
    print(f"\nwrote {out_dir/'results.json'}")
    print(f"wrote {pp_path}")


if __name__ == "__main__":
    main()
