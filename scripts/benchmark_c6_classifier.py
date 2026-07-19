#!/usr/bin/env python3
"""
C6 — feature-based per-prompt KV-cache router.

Trains 6 regressors (one per config) on prompt-intrinsic features to predict
the per-prompt LongBench score. At inference, picks the config with the
highest compression whose predicted score is at least ``tau`` * predicted
score of C0 (iso-quality routing).

Reads the joined dataset built by ``build_dataset.py`` (which already contains
each prompt's features and the measured score under every config). Therefore
this script does NOT call the model — the routing decision is a pure CPU step.
Its true single-request cost (feature extraction incl. tokenization + scaling +
6 regressor predicts) is measured honestly at millisecond scale, see
``measure_routing_overhead_ms``.

Two evaluations are reported and saved:
  - LOTO  : leave-one-task-out CV (out-of-distribution by task family)
  - Split : single random 70/30 split (in-distribution per-workload calibration)

Top-level fields of results_tau{tau}.json mirror the results.json written by
benchmark_c{0..5} so plotting tools can treat C6 as just another config. Aggregate scores at the top level use the
LOTO predictions; the random-split summary is included under ``eval_split``.

Output (per tau, so runs at different taus never overwrite each other):
  runs/p0/C6/{model}/results_tau{tau}.json
  runs/p0/C6/{model}/per_prompt_tau{tau}.jsonl
    (one row per prompt with chosen config)

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
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from _common import assert_not_p0_output_path

REPO        = Path(__file__).resolve().parents[1]
P0_RUNS     = REPO / "runs" / "p0"
CONFIGS     = ["C0", "C1", "C2", "C3", "C4", "C5"]
CANDIDATES  = ["C1", "C2", "C3", "C4", "C5"]  # configs the router may CHOOSE
REFERENCE   = "C0"                            # quality yardstick only — never chosen
FEATURE_KEYS = [
    "seq_len_tokens", "seq_len_chars", "token_entropy",
    "gzip_ratio", "unique_token_ratio", "question_position",
    "newline_density",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset(model: str) -> list[dict]:
    path = P0_RUNS / "dataset" / f"{model}.jsonl"
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


def viable_candidates(rows_tr: list[dict], tau: float) -> list[str]:
    """Candidates whose training-set mean quality is >= tau * mean C0.

    A config that on average delivers worse than tau * C0 quality cannot be
    an iso-quality choice in expectation. Excluding such configs prevents
    the router from picking high-compression configs whose predicted quality
    crossed the iso-quality bar via prediction noise alone.

    Empirically: at tau=0.99 this drops C1 (TailorKV) on every model in our
    suite, because C1's mean quality is 87-97% of C0's across phi3, llama3,
    llama32_3b, llama31_8b. C1 still PASSES the per-prompt bar on 60-72% of
    prompts — but its training mean fails on average, so the router can't
    rely on a noisy regressor to find the prompts where C1 happens to work.
    """
    if not rows_tr:
        return list(CANDIDATES)
    mean_c0 = sum(r["scores"]["C0"] for r in rows_tr) / len(rows_tr)
    out = []
    for c in CANDIDATES:
        mean_c = sum(r["scores"][c] for r in rows_tr) / len(rows_tr)
        if mean_c >= tau * mean_c0:
            out.append(c)
    return out if out else list(CANDIDATES)   # fail-safe: never empty


def train_mean_compression(rows_tr: list[dict]) -> dict[str, float]:
    """Per-config harmonic-mean measured compression over the TRAINING fold.

    This is the deployment-safe ranking key for ``route()``: at request time
    the router cannot know the prompt's measured compression under each config
    (that is only observable after running the config), so candidates are
    ranked by the per-config mean compression estimated from training data.
    """
    out: dict[str, float] = {}
    for c in CONFIGS:
        crs = [r["compression"][c] for r in rows_tr if r["compression"][c] > 0]
        out[c] = float(harmonic_mean(crs)) if crs else 1.0
    return out


def route(
    rows: list[dict],
    q_pred: dict[str, np.ndarray],
    tau: float,
    c0_floor: float = 0.0,
    candidates: list[str] | None = None,
    cr_rank: dict[str, float] | None = None,
) -> list[str]:
    """Pick the candidate with the highest TRAIN-FOLD MEAN compression
    (``cr_rank``) whose predicted quality is at least
    ``tau`` * max(predicted q[C0], c0_floor).

    ``cr_rank`` must be the per-config mean compression computed on the
    TRAINING fold (see ``train_mean_compression``). Ranking by the routed
    prompt's own measured compression would leak post-hoc information that is
    unavailable pre-run at deployment; measured per-prompt compression is used
    only for CREDITING the achieved compression in evaluation aggregates
    (see ``aggregate``).

    C0 (uncompressed) is the quality YARDSTICK only — it is never a routing
    choice. The router always selects from ``CANDIDATES`` = {C1..C5}.

    If no candidate clears the iso-quality bar, the safest fallback is the
    candidate with the highest predicted quality (NOT C0): the goal is to
    deliver compression, so we pick the best-quality compressed option rather
    than abandoning compression entirely.

    ``c0_floor`` defends against the regressor systematically under-predicting
    C0 quality on hard tasks: when the predicted C0 collapses, the floor (set
    to the training-set empirical mean of actual C0 scores) keeps the
    iso-quality bar honest. Set to 0.0 to disable.
    """
    if cr_rank is None:
        raise ValueError(
            "route() requires cr_rank — the per-config mean compression of the "
            "TRAINING fold (use train_mean_compression(rows_tr)). Per-prompt "
            "measured compression is not available at deployment time."
        )
    cand = candidates if candidates is not None else list(CANDIDATES)
    picks: list[str] = []
    for i, _r in enumerate(rows):
        anchor = max(float(q_pred[REFERENCE][i]), c0_floor)
        thresh = tau * anchor
        # Among candidates that clear the bar, pick the one with the highest
        # train-fold mean compression.
        best_c, best_cr = None, -1.0
        for c in cand:
            if q_pred[c][i] >= thresh and cr_rank[c] > best_cr:
                best_c, best_cr = c, cr_rank[c]
        if best_c is None:
            # Nothing clears the bar: pick the safest candidate (highest
            # predicted quality among viable candidates, still never C0).
            best_c = max(cand, key=lambda c: float(q_pred[c][i]))
        picks.append(best_c)
    return picks


def c0_floor_from_train(rows_tr: list[dict]) -> float:
    """Empirical mean of ACTUAL C0 scores on the training fold."""
    ys = [r["scores"]["C0"] for r in rows_tr]
    return float(mean(ys)) if ys else 0.0


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


def loto_picks(rows: list[dict], X: np.ndarray, tau: float,
               use_floor: bool = True, filter_candidates: bool = False) -> list[str]:
    tasks = sorted({r["task"] for r in rows})
    picks: list[str | None] = [None] * len(rows)
    for held in tasks:
        tr = [i for i, r in enumerate(rows) if r["task"] != held]
        te = [i for i, r in enumerate(rows) if r["task"] == held]
        rows_tr = [rows[i] for i in tr]
        models = fit_regressors(X[tr], rows_tr)
        q_pred = predict_q(models, X[te])
        rows_te = [rows[i] for i in te]
        floor = c0_floor_from_train(rows_tr) if use_floor else 0.0
        cand  = viable_candidates(rows_tr, tau) if filter_candidates else None
        cr_tr = train_mean_compression(rows_tr)
        for i, c in zip(te, route(rows_te, q_pred, tau, c0_floor=floor,
                                  candidates=cand, cr_rank=cr_tr)):
            picks[i] = c
    return picks  # type: ignore[return-value]


def split_picks(rows: list[dict], X: np.ndarray, tau: float, seed: int = 0,
                use_floor: bool = True, filter_candidates: bool = False):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    n_tr = int(0.7 * len(rows))
    tr, te = idx[:n_tr].tolist(), idx[n_tr:].tolist()
    models = fit_regressors(X[tr], [rows[i] for i in tr])
    # Per-config train/test fit quality (R^2 and MAE) on the predicted score.
    fit_metrics = {}
    for c in CONFIGS:
        sc, reg = models[c]
        y_tr = np.array([rows[i]["scores"][c] for i in tr], dtype=np.float32)
        y_te = np.array([rows[i]["scores"][c] for i in te], dtype=np.float32)
        p_tr = reg.predict(sc.transform(X[tr]))
        p_te = reg.predict(sc.transform(X[te]))
        fit_metrics[c] = {
            "r2_train":  round(float(r2_score(y_tr, p_tr)), 4),
            "r2_test":   round(float(r2_score(y_te, p_te)), 4),
            "mae_train": round(float(mean_absolute_error(y_tr, p_tr)), 4),
            "mae_test":  round(float(mean_absolute_error(y_te, p_te)), 4),
        }
    q_pred_tr = predict_q(models, X[tr])
    q_pred_te = predict_q(models, X[te])
    rows_te = [rows[i] for i in te]
    rows_tr = [rows[i] for i in tr]
    floor = c0_floor_from_train(rows_tr) if use_floor else 0.0
    cand  = viable_candidates(rows_tr, tau) if filter_candidates else None
    cr_tr = train_mean_compression(rows_tr)
    picks_te = route(rows_te, q_pred_te, tau, c0_floor=floor, candidates=cand, cr_rank=cr_tr)
    picks_tr = route(rows_tr, q_pred_tr, tau, c0_floor=floor, candidates=cand, cr_rank=cr_tr)
    return rows_te, picks_te, rows_tr, picks_tr, fit_metrics


def measure_routing_overhead_ms(models, model_key: str, n_prompts: int = 50) -> dict:
    """Honest single-request routing overhead, in MILLISECONDS.

    For ~``n_prompts`` real LongBench prompts (reconstructed exactly as the
    benchmarks build them: load_longbench_task + LB_TEMPLATES), times the full
    per-request routing path on a batch of ONE:

        extract_prompt_features(prompt, tokenizer)   # incl. tokenization+gzip
        + StandardScaler.transform                   # per config
        + regressor.predict on the 1-row batch       # 6 configs

    Returns {"routing_overhead_ms_median", "routing_overhead_ms_p95",
    "routing_overhead_n_prompts"}. This replaces the old batched
    microsecond-scale number, which amortized sklearn predict over hundreds of
    prompts and excluded feature extraction entirely — tokenizing a multi-k-token
    prompt dominates the real cost, putting it at millisecond scale.

    CPU-only: loads the tokenizer (AutoTokenizer), never the model.
    """
    # Lazy imports: _common pulls in torch/datasets, which the pure-routing
    # paths (cv_router_220.py, plot_pareto_cv.py) don't need.
    from transformers import AutoTokenizer
    from _common import LB_TASKS, LB_TEMPLATES, MODELS, load_longbench_task, \
        extract_prompt_features

    tokenizer = AutoTokenizer.from_pretrained(MODELS[model_key])

    per_task = max(1, -(-n_prompts // len(LB_TASKS)))  # ceil division
    prompts: list[str] = []
    for task in LB_TASKS:
        try:
            samples = load_longbench_task(task)
        except FileNotFoundError:
            continue
        for sample in samples[:per_task]:
            prompts.append(LB_TEMPLATES[task].format(
                context=sample.get("context", ""), input=sample["input"]))
    prompts = prompts[:n_prompts]
    if not prompts:
        raise RuntimeError("no LongBench prompts available for overhead measurement")

    def _route_one(prompt: str) -> None:
        feats = extract_prompt_features(prompt, tokenizer)
        x = np.array([[(-1.0 if feats.get(k) is None else float(feats[k]))
                       for k in FEATURE_KEYS]], dtype=np.float32)
        for sc, reg in models.values():
            reg.predict(sc.transform(x))

    _route_one(prompts[0])  # warm-up (sklearn/tokenizer first-call costs)
    times_ms = []
    for prompt in prompts:
        t0 = time.perf_counter()
        _route_one(prompt)
        times_ms.append((time.perf_counter() - t0) * 1e3)

    return {
        "routing_overhead_ms_median": round(float(np.median(times_ms)), 3),
        "routing_overhead_ms_p95":    round(float(np.percentile(times_ms, 95)), 3),
        "routing_overhead_n_prompts": len(times_ms),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["llama3", "phi3", "llama31_8b", "llama32_3b"])
    ap.add_argument("--tau", type=float, default=0.99,
                    help="iso-quality threshold (predicted q[c] >= tau * predicted q[C0])")
    ap.add_argument("--no-floor", action="store_true",
                    help="disable the C0 empirical-mean floor on the iso-quality anchor "
                         "(restores legacy behavior where thresh = tau * pred_C0)")
    ap.add_argument("--filter-candidates", action="store_true",
                    help="exclude candidates whose training-set mean quality is below "
                         "tau * mean(C0). Prevents the router from picking a "
                         "high-compression config whose quality crosses the bar only via "
                         "prediction noise. Empirically drops C1 (TailorKV) on all four "
                         "models at tau=0.99.")
    ap.add_argument("--tau-sweep", type=str, default=None,
                    help="comma-separated tau values to evaluate; writes a summary "
                         "to runs/p0/C6/{model}/tau_sweep[_nofloor][_filtered].json "
                         "(suffix encodes --no-floor/--filter-candidates) and skips "
                         "the headline run")
    args = ap.parse_args()
    protected_out_dir = P0_RUNS / "C6" / args.model
    assert_not_p0_output_path(protected_out_dir)
    use_floor       = not args.no_floor
    filter_cands    = args.filter_candidates

    rows = load_dataset(args.model)
    X    = featurize(rows)

    if filter_cands:
        cand_all = viable_candidates(rows, args.tau)
        dropped  = [c for c in CANDIDATES if c not in cand_all]
        print(f"\n[--filter-candidates] viable on full set @tau={args.tau}: "
              f"{cand_all}   dropped: {dropped}")

    # ---- Tau-sweep mode: evaluate many taus on existing dataset, no GPU.
    if args.tau_sweep:
        taus = [float(t) for t in args.tau_sweep.split(",") if t.strip()]
        print(f"\n=== C6 tau-sweep  model={args.model}  use_floor={use_floor}  "
              f"taus={taus}  n={len(rows)} ===\n")
        sweep = []
        for tau in taus:
            picks_loto_t = loto_picks(rows, X, tau, use_floor=use_floor,
                                      filter_candidates=filter_cands)
            agg_loto_t   = aggregate(rows, picks_loto_t)
            rows_te_t, picks_te_t, rows_tr_t, picks_tr_t, _ = split_picks(
                rows, X, tau, seed=0, use_floor=use_floor,
                filter_candidates=filter_cands)
            agg_split_t    = aggregate(rows_te_t, picks_te_t)
            agg_split_tr_t = aggregate(rows_tr_t, picks_tr_t)
            # iso-quality contract violation rate (per-prompt): picked_score < tau * actual_C0
            def viol_rate(rs, ps):
                v = sum(1 for r, c in zip(rs, ps) if r["scores"][c] < tau * r["scores"]["C0"])
                return round(v / max(len(rs), 1), 4)
            entry = {
                "tau": tau,
                "loto": {
                    "q":  agg_loto_t["longbench_avg"],
                    "cr": agg_loto_t["longbench_avg_compression_ratio"],
                    "picks": agg_loto_t["picks"],
                    "violation_rate": viol_rate(rows, picks_loto_t),
                },
                "split_test": {
                    "q":  agg_split_t["longbench_avg"],
                    "cr": agg_split_t["longbench_avg_compression_ratio"],
                    "picks": agg_split_t["picks"],
                    "violation_rate": viol_rate(rows_te_t, picks_te_t),
                },
                "split_train": {
                    "q":  agg_split_tr_t["longbench_avg"],
                    "cr": agg_split_tr_t["longbench_avg_compression_ratio"],
                },
            }
            sweep.append(entry)
            print(f"tau={tau:.2f}  "
                  f"LOTO q={entry['loto']['q']:.3f} cr={entry['loto']['cr']:.2f}x "
                  f"viol={entry['loto']['violation_rate']:.0%}  |  "
                  f"Split-te q={entry['split_test']['q']:.3f} cr={entry['split_test']['cr']:.2f}x "
                  f"viol={entry['split_test']['violation_rate']:.0%}  "
                  f"picks={entry['split_test']['picks']}")
        out_dir = protected_out_dir
        assert_not_p0_output_path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Filename encodes BOTH ablation flags so sweeps never overwrite each other.
        suffix = ("_nofloor" if not use_floor else "") + \
                 ("_filtered" if filter_cands else "")
        sweep_path = out_dir / f"tau_sweep{suffix}.json"
        assert_not_p0_output_path(sweep_path)
        sweep_path.write_text(json.dumps({
            "model": args.model, "use_floor": use_floor,
            "filter_candidates": filter_cands, "sweep": sweep,
        }, indent=2))
        print(f"\nwrote {out_dir/('tau_sweep'+suffix+'.json')}")
        return

    print(f"\n=== C6 router  model={args.model}  tau={args.tau}  use_floor={use_floor}  "
          f"features={len(FEATURE_KEYS)}  n={len(rows)} ===\n")

    # ---- LOTO (OOD by task) -- headline numbers
    picks_loto = loto_picks(rows, X, args.tau, use_floor=use_floor,
                            filter_candidates=filter_cands)
    agg_loto   = aggregate(rows, picks_loto)
    print(f"LOTO (OOD): q={agg_loto['longbench_avg']:.4f}  "
          f"cr={agg_loto['longbench_avg_compression_ratio']:.2f}x  "
          f"picks={agg_loto['picks']}")

    # ---- Random 70/30 split (in-dist)
    rows_te, picks_te, rows_tr, picks_tr, fit_metrics = split_picks(
        rows, X, args.tau, seed=0, use_floor=use_floor,
        filter_candidates=filter_cands)
    agg_split = aggregate(rows_te, picks_te)
    agg_split_tr = aggregate(rows_tr, picks_tr)
    print(f"Split-tr : q={agg_split_tr['longbench_avg']:.4f}  "
          f"cr={agg_split_tr['longbench_avg_compression_ratio']:.2f}x  "
          f"(n_train={len(rows_tr)})")
    print(f"Split-te : q={agg_split['longbench_avg']:.4f}  "
          f"cr={agg_split['longbench_avg_compression_ratio']:.2f}x  "
          f"picks={agg_split['picks']}  (n_test={len(rows_te)})")
    r2_tr_mean = float(np.mean([m["r2_train"] for m in fit_metrics.values()]))
    r2_te_mean = float(np.mean([m["r2_test"]  for m in fit_metrics.values()]))
    mae_tr_mean = float(np.mean([m["mae_train"] for m in fit_metrics.values()]))
    mae_te_mean = float(np.mean([m["mae_test"]  for m in fit_metrics.values()]))
    print(f"Regressor fit (mean over 6 configs): "
          f"R2 train={r2_tr_mean:.3f} test={r2_te_mean:.3f}  "
          f"MAE train={mae_tr_mean:.3f} test={mae_te_mean:.3f}")
    for c in CONFIGS:
        m = fit_metrics[c]
        print(f"  {c}: R2 tr={m['r2_train']:+.3f} te={m['r2_test']:+.3f}  "
              f"MAE tr={m['mae_train']:.3f} te={m['mae_test']:.3f}")

    # ---- Routing overhead (honest single-request path, ms scale)
    full_models = fit_regressors(X, rows)
    overhead = measure_routing_overhead_ms(full_models, args.model)
    print(f"Routing overhead (single request: feature extraction + scaling + "
          f"6 predicts): median {overhead['routing_overhead_ms_median']:.2f} ms, "
          f"p95 {overhead['routing_overhead_ms_p95']:.2f} ms "
          f"(n={overhead['routing_overhead_n_prompts']} prompts)")

    # ---- Save
    results = {
        "config": "C6",
        "method": "per-prompt feature-based router (HistGradientBoosting regression)",
        "model":  args.model,
        "tau":    args.tau,
        "use_floor": use_floor,
        "features": FEATURE_KEYS,
        "n_features": len(FEATURE_KEYS),
        # Honest single-request overhead: feature extraction (incl. tokenizing
        # the prompt) + scaler.transform + 6 regressor predicts on batch of 1.
        **overhead,

        # Top-level fields mirror C0..C5 using the LOTO (honest OOD) headline.
        "longbench":                       agg_loto["longbench"],
        "longbench_avg":                   agg_loto["longbench_avg"],
        "longbench_avg_compression_ratio": agg_loto["longbench_avg_compression_ratio"],
        "picks":                           agg_loto["picks"],

        # In-distribution random-split summary (workload-calibrated regime).
        "eval_split_70_30": {
            "n_train":                         len(rows_tr),
            "n_test":                          len(rows_te),
            "longbench_avg_train":             agg_split_tr["longbench_avg"],
            "longbench_avg_compression_ratio_train": agg_split_tr["longbench_avg_compression_ratio"],
            "longbench_avg":                   agg_split["longbench_avg"],
            "longbench_avg_compression_ratio": agg_split["longbench_avg_compression_ratio"],
            "picks":                           agg_split["picks"],
        },

        # Regressor fit quality per config (HistGradientBoosting, predicting q[c]).
        "regressor_fit": {
            "model":      "HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05, random_state=0)",
            "n_train":    len(rows_tr),
            "n_test":     len(rows_te),
            "per_config": fit_metrics,
            "mean_r2_train":  round(r2_tr_mean, 4),
            "mean_r2_test":   round(r2_te_mean, 4),
            "mean_mae_train": round(mae_tr_mean, 4),
            "mean_mae_test":  round(mae_te_mean, 4),
        },
    }

    results["filter_candidates"] = filter_cands
    suffix = "_filtered" if filter_cands else ""

    out_dir = protected_out_dir
    assert_not_p0_output_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Tau-specific filename ONLY: a plain results.json would silently be
    # overwritten by whichever tau ran last.
    res_path = out_dir / f"results_tau{args.tau}{suffix}.json"
    assert_not_p0_output_path(res_path)
    res_path.write_text(json.dumps(results, indent=2))

    pp_path = out_dir / f"per_prompt_tau{args.tau}{suffix}.jsonl"
    assert_not_p0_output_path(pp_path)
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
    print(f"\nwrote {res_path}")
    print(f"wrote {pp_path}")


if __name__ == "__main__":
    main()
