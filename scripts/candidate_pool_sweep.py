#!/usr/bin/env python3
"""Evaluate C6 candidate pools on the corrected P0 routing datasets.

The router already accepts an explicit candidate list, but the normal C6 and
CV entrypoints evaluate only the full C1-C5 pool. This script exhaustively
evaluates candidate pools against the same 10-fold, task-stratified protocol
used by ``cv_router_220.py``. Regressors and test-fold predictions are cached
once per fold, then reused for every pool and threshold.

Outputs are written beneath a versioned run directory and include dataset
hashes, evaluation settings, all operating points, Pareto summaries, a
machine-readable all-five consistency check, and per-model figures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import harmonic_mean
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from benchmark_c6_classifier import (
    CANDIDATES,
    CONFIGS,
    aggregate,
    c0_floor_from_train,
    featurize,
    fit_regressors,
    load_dataset,
    predict_q,
    route,
    train_mean_compression,
)


ROOT = Path(__file__).resolve().parents[1]
MODELS = ["phi3", "llama3", "llama32_3b", "llama31_8b"]
DEFAULT_TAUS = (0.99, 0.95, 0.90, 0.85, 0.80)
DEFAULT_RUN_ID = "p0_corrected_2026-07-16"
SCHEMA_VERSION = "adaptive-serve-candidate-pool-sweep/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def canonical_pool(configurations: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    selected = set(configurations)
    unknown = selected.difference(CANDIDATES)
    if unknown:
        raise ValueError(f"unknown candidate configurations: {sorted(unknown)}")
    ordered = tuple(configuration for configuration in CANDIDATES if configuration in selected)
    if not ordered:
        raise ValueError("a candidate pool cannot be empty")
    return ordered


def pool_key(pool: tuple[str, ...]) -> str:
    return "+".join(pool)


def parse_taus(value: str) -> tuple[float, ...]:
    taus = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not taus:
        raise ValueError("at least one tau is required")
    if any(tau <= 0.0 or tau > 1.0 for tau in taus):
        raise ValueError("tau values must be in (0, 1]")
    return taus


def parse_explicit_pools(value: str | None) -> list[tuple[str, ...]]:
    if not value:
        return []
    pools = []
    for raw_pool in value.split(";"):
        configurations = [item.strip().upper() for item in raw_pool.split(",") if item.strip()]
        if not configurations:
            continue
        pools.append(canonical_pool(configurations))
    if not pools:
        raise ValueError("--pools did not contain a usable candidate pool")
    return list(dict.fromkeys(pools))


def enumerate_pools(minimum_size: int, maximum_size: int) -> list[tuple[str, ...]]:
    if minimum_size < 1 or maximum_size > len(CANDIDATES) or minimum_size > maximum_size:
        raise ValueError(
            f"pool sizes must satisfy 1 <= min <= max <= {len(CANDIDATES)}"
        )
    return [
        tuple(pool)
        for size in range(minimum_size, maximum_size + 1)
        for pool in combinations(CANDIDATES, size)
    ]


def cache_folds(
    rows: list[dict[str, Any]],
    features: np.ndarray,
    n_splits: int,
    seed: int,
    use_floor: bool,
) -> list[dict[str, Any]]:
    labels = np.array([row["task"] for row in rows])
    smallest_task_count = min(int((labels == label).sum()) for label in set(labels))
    if n_splits > smallest_task_count:
        raise ValueError(
            f"n_splits={n_splits} exceeds the smallest task count ({smallest_task_count})"
        )

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cached_folds: list[dict[str, Any]] = []
    for training_indices, test_indices in splitter.split(np.zeros(len(rows)), labels):
        training_rows = [rows[index] for index in training_indices]
        test_rows = [rows[index] for index in test_indices]
        regressors = fit_regressors(features[training_indices], training_rows)
        cached_folds.append(
            {
                "test_indices": test_indices,
                "test_rows": test_rows,
                "predictions": predict_q(regressors, features[test_indices]),
                "c0_floor": c0_floor_from_train(training_rows) if use_floor else 0.0,
                "compression_rank": train_mean_compression(training_rows),
            }
        )
    return cached_folds


def fixed_on_full(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    fixed = {}
    for configuration in CONFIGS:
        scores = [row["scores"][configuration] for row in rows]
        compression = [
            row["compression"][configuration]
            for row in rows
            if row["compression"][configuration] > 0
        ]
        fixed[configuration] = {
            "q": round(sum(scores) / len(scores), 4),
            "cr": round(harmonic_mean(compression), 3),
        }
    return fixed


def dominates(left: dict[str, float], right: dict[str, float]) -> bool:
    return (
        left["q"] >= right["q"]
        and left["cr"] >= right["cr"]
        and (left["q"] > right["q"] or left["cr"] > right["cr"])
    )


def pareto_against_fixed(
    router: dict[str, float], fixed: dict[str, dict[str, float]]
) -> dict[str, str]:
    verdicts = {}
    for configuration, fixed_point in fixed.items():
        if dominates(router, fixed_point):
            verdicts[configuration] = "router_dominates"
        elif dominates(fixed_point, router):
            verdicts[configuration] = "router_dominated"
        else:
            verdicts[configuration] = "non_dominated"
    return verdicts


def evaluate_operating_point(
    rows: list[dict[str, Any]],
    cached_folds: list[dict[str, Any]],
    pool: tuple[str, ...],
    tau: float,
    fixed: dict[str, dict[str, float]],
) -> dict[str, Any]:
    chosen_configurations: list[str | None] = [None] * len(rows)
    for cached_fold in cached_folds:
        fold_picks = route(
            cached_fold["test_rows"],
            cached_fold["predictions"],
            tau,
            c0_floor=cached_fold["c0_floor"],
            candidates=list(pool),
            cr_rank=cached_fold["compression_rank"],
        )
        for index, configuration in zip(cached_fold["test_indices"], fold_picks):
            chosen_configurations[int(index)] = configuration

    if any(configuration is None for configuration in chosen_configurations):
        raise RuntimeError("cross-validation did not route every prompt exactly once")
    aggregate_result = aggregate(rows, [str(configuration) for configuration in chosen_configurations])
    router_point = {
        "q": aggregate_result["longbench_avg"],
        "cr": aggregate_result["longbench_avg_compression_ratio"],
    }
    return {
        "pool": list(pool),
        "pool_key": pool_key(pool),
        "pool_size": len(pool),
        "tau": tau,
        "router_cv": {**router_point, "picks": aggregate_result["picks"]},
        "pareto_vs_fixed": pareto_against_fixed(router_point, fixed),
    }


def ranking_key(entry: dict[str, Any]) -> tuple[float, float, int, str, float]:
    router = entry["router_cv"]
    return (
        -float(router["q"]),
        -float(router["cr"]),
        int(entry["pool_size"]),
        str(entry["pool_key"]),
        -float(entry["tau"]),
    )


def best_quality(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return min(entries, key=ranking_key)


def pareto_frontier(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier = []
    for entry in entries:
        point = entry["router_cv"]
        if not any(
            other is not entry and dominates(other["router_cv"], point)
            for other in entries
        ):
            frontier.append(entry)
    return sorted(frontier, key=lambda entry: (entry["router_cv"]["cr"], entry["router_cv"]["q"]))


def best_fp16_dominating(
    entries: list[dict[str, Any]], fixed: dict[str, dict[str, float]]
) -> dict[str, Any] | None:
    c0_point = fixed["C0"]
    dominating = [
        entry
        for entry in entries
        if dominates(entry["router_cv"], c0_point)
    ]
    return best_quality(dominating) if dominating else None


def existing_cv_check(
    model: str, entry: dict[str, Any]
) -> dict[str, Any] | None:
    if tuple(entry["pool"]) != tuple(CANDIDATES):
        return None
    tau = float(entry["tau"])
    path = ROOT / "runs" / f"cv_router_220_tau{tau}.json"
    if not path.exists():
        return {
            "tau": tau,
            "path": str(path.relative_to(ROOT)),
            "status": "not_available",
        }
    expected = json.loads(path.read_text())[model]["router_cv"]
    actual = entry["router_cv"]
    matched = (
        float(actual["q"]) == float(expected["q"])
        and float(actual["cr"]) == float(expected["cr"])
        and actual["picks"] == expected["picks"]
    )
    return {
        "tau": tau,
        "path": str(path.relative_to(ROOT)),
        "status": "matched" if matched else "mismatch",
        "expected": expected,
        "actual": actual,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def entry_label(entry: dict[str, Any] | None) -> str:
    if entry is None:
        return "none"
    router = entry["router_cv"]
    return (
        f"{entry['pool_key']} at tau={entry['tau']:.2f} "
        f"(q={router['q']:.4f}, cr={router['cr']:.3f}x)"
    )


def write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Corrected P0 Candidate-Pool Sweep",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Schema: `{summary['schema_version']}`",
        f"- Generated: `{summary['generated_at_utc']}`",
        "- Protocol: 10-fold task-stratified CV; fold regressors and predictions are cached once, then reused for every pool/tau pair.",
        "- This is an evaluation artifact. `best_by_quality` is a transparent ranking, not a deployment recommendation without a chosen quality/compression objective.",
        "- Pool/tau selection uses the same CV aggregate shown here; use a separate calibration workload or nested CV for an unbiased post-selection generalization estimate.",
        "",
        "## Results",
        "",
        "| Model | Evaluated points | Best by quality | Best strictly dominating C0 | Pareto points |",
        "| --- | ---: | --- | --- | ---: |",
    ]
    for model, model_result in summary["models"].items():
        lines.append(
            "| "
            f"{model} | {len(model_result['operating_points'])} | "
            f"{entry_label(model_result['best_by_quality'])} | "
            f"{entry_label(model_result['best_fp16_dominating'])} | "
            f"{len(model_result['pareto_frontier'])} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary.json`: complete all-model result and provenance manifest.",
            "- `<model>.json`: per-model subset of the manifest.",
            "- `figs/candidate_pools_<model>.png`: P0-corrected pool/tau operating points.",
            "",
            "The historical May candidate-pool figures remain separate pre-P0 artifacts and must not be used as corrected P0 evidence.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def plot_model(
    path: Path,
    model: str,
    model_result: dict[str, Any],
    taus: tuple[float, ...],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    entries = model_result["operating_points"]
    fixed = model_result["fixed_full"]
    figure, axis = plt.subplots(figsize=(10.2, 6.8), constrained_layout=True)
    color_map = plt.get_cmap("viridis")
    tau_colors = {
        tau: color_map(index / max(len(taus) - 1, 1))
        for index, tau in enumerate(sorted(taus))
    }
    marker_by_size = {2: "o", 3: "s", 4: "^", 5: "D"}

    for entry in entries:
        router = entry["router_cv"]
        axis.scatter(
            router["cr"],
            router["q"],
            color=tau_colors[float(entry["tau"])],
            marker=marker_by_size[int(entry["pool_size"])],
            s=54,
            alpha=0.72,
            linewidths=0.35,
            edgecolors="#1d1d1d",
            zorder=3,
        )

    frontier = model_result["pareto_frontier"]
    if frontier:
        frontier_x = [entry["router_cv"]["cr"] for entry in frontier]
        frontier_y = [entry["router_cv"]["q"] for entry in frontier]
        axis.plot(frontier_x, frontier_y, color="#0b6e4f", linewidth=1.4, zorder=2)

    for configuration, point in fixed.items():
        axis.scatter(
            point["cr"],
            point["q"],
            color="#2d4059" if configuration == "C0" else "#d95d39",
            marker="P" if configuration == "C0" else "X",
            s=98,
            zorder=5,
        )
        axis.annotate(
            configuration,
            (point["cr"], point["q"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    best_by_quality = model_result["best_by_quality"]
    best_fp16 = model_result["best_fp16_dominating"]
    if best_fp16 == best_by_quality:
        annotations = [("Best quality and vs C0", best_by_quality, (8, -24))]
    else:
        annotations = [
            ("Best quality", best_by_quality, (8, 16)),
            ("Best vs C0", best_fp16, (8, -26)),
        ]
    for title, entry, offset in annotations:
        if entry is None:
            continue
        router = entry["router_cv"]
        axis.scatter(
            router["cr"],
            router["q"],
            facecolors="none",
            edgecolors="#111111",
            linewidths=1.7,
            s=180,
            zorder=6,
        )
        axis.annotate(
            f"{title}: {entry['pool_key']}, tau={entry['tau']:.2f}",
            (router["cr"], router["q"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "#444444", "lw": 0.7},
        )

    tau_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=tau_colors[tau],
               markeredgecolor="#1d1d1d", label=f"tau={tau:.2f}", markersize=7)
        for tau in sorted(taus, reverse=True)
    ]
    size_handles = [
        Line2D([0], [0], marker=marker, color="#555555", label=f"pool size {size}",
               linestyle="None", markersize=7)
        for size, marker in marker_by_size.items()
    ]
    fixed_handle = Line2D([0], [0], marker="X", color="#d95d39", label="fixed C0-C5",
                          linestyle="None", markersize=7)
    axis.legend(
        handles=[*tau_handles, *size_handles, fixed_handle],
        loc="best",
        fontsize=8,
        frameon=True,
        title="Router points",
        title_fontsize=8,
    )
    axis.margins(x=0.04, y=0.09)
    axis.set_title(f"Corrected P0 candidate-pool sweep: {model}", pad=16)
    axis.set_xlabel("Measured KV compression ratio (harmonic mean, higher is better)")
    axis.set_ylabel("LongBench mean score (higher is better)")
    axis.grid(alpha=0.24, linewidth=0.6)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        choices=MODELS,
        help="model to evaluate; repeat to select several (default: all models)",
    )
    parser.add_argument(
        "--taus",
        default=",".join(str(tau) for tau in DEFAULT_TAUS),
        help="comma-separated iso-quality thresholds",
    )
    parser.add_argument(
        "--pools",
        help="semicolon-separated explicit pools, e.g. C1,C2;C2,C3,C5; default enumerates pools",
    )
    parser.add_argument(
        "--min-pool-size",
        type=int,
        default=2,
        help="minimum enumerated pool size when --pools is omitted (default: 2)",
    )
    parser.add_argument(
        "--max-pool-size",
        type=int,
        default=len(CANDIDATES),
        help="maximum enumerated pool size when --pools is omitted (default: 5)",
    )
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-floor",
        action="store_true",
        help="disable C6's train-fold empirical C0 quality floor",
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="versioned output directory (default: runs/candidate_pool_sweeps/<run-id>)",
    )
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--verify-existing-cv",
        action="store_true",
        help="fail if an evaluated all-five pool disagrees with an existing cv_router_220 result",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_models = args.model or MODELS
    taus = parse_taus(args.taus)
    pools = parse_explicit_pools(args.pools) or enumerate_pools(
        args.min_pool_size, args.max_pool_size
    )
    output_dir = args.output_dir or (
        ROOT / "runs" / "candidate_pool_sweeps" / args.run_id
    )
    figure_dir = output_dir / "figs"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_figures:
        figure_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "generated_at_utc": generated_at,
        "source_commit": current_commit(),
        "protocol": {
            "cv": "StratifiedKFold by LongBench task",
            "n_splits": args.n_splits,
            "seed": args.seed,
            "use_c0_floor": not args.no_floor,
            "taus": list(taus),
            "candidate_pools": [list(pool) for pool in pools],
            "n_operating_points_per_model": len(pools) * len(taus),
            "ranking": "highest quality, then compression, then smaller pool",
            "selection_scope": (
                "Pool/tau ranking uses this same cross-validation aggregate; it is not "
                "a nested-CV post-selection generalization estimate."
            ),
        },
        "models": {},
    }
    failed_checks = []
    available_checks = 0

    for model in selected_models:
        rows = load_dataset(model)
        dataset_path = ROOT / "runs" / "dataset" / f"{model}.jsonl"
        features = featurize(rows)
        print(
            f"[{model}] caching {args.n_splits} folds for {len(rows)} prompts; "
            f"{len(pools)} pools x {len(taus)} taus"
        )
        cached_folds = cache_folds(
            rows, features, args.n_splits, args.seed, use_floor=not args.no_floor
        )
        fixed = fixed_on_full(rows)
        entries = [
            evaluate_operating_point(rows, cached_folds, pool, tau, fixed)
            for pool in pools
            for tau in taus
        ]
        consistency_checks = []
        for entry in entries:
            check = existing_cv_check(model, entry)
            if check is not None:
                consistency_checks.append(check)
                if check["status"] != "not_available":
                    available_checks += 1
                if check["status"] == "mismatch":
                    failed_checks.append(f"{model} tau={entry['tau']}")

        model_result = {
            "model": model,
            "dataset": {
                "path": str(dataset_path.relative_to(ROOT)),
                "sha256": sha256_file(dataset_path),
                "n_rows": len(rows),
            },
            "fixed_full": fixed,
            "operating_points": entries,
            "best_by_quality": best_quality(entries),
            "best_fp16_dominating": best_fp16_dominating(entries, fixed),
            "pareto_frontier": pareto_frontier(entries),
            "all_five_consistency": consistency_checks,
        }
        summary["models"][model] = model_result
        write_json(output_dir / f"{model}.json", model_result)
        if not args.no_figures:
            plot_model(figure_dir / f"candidate_pools_{model}.png", model, model_result, taus)

        best = model_result["best_by_quality"]
        print(
            f"[{model}] best quality: {entry_label(best)}; "
            f"best vs C0: {entry_label(model_result['best_fp16_dominating'])}"
        )

    write_json(output_dir / "summary.json", summary)
    write_markdown_summary(output_dir / "README.md", summary)
    print(f"wrote {display_path(output_dir)}")

    if args.verify_existing_cv:
        if not available_checks:
            raise SystemExit(
                "--verify-existing-cv found no all-five tau with a matching cv_router_220 output"
            )
        if failed_checks:
            raise SystemExit(
                "all-five consistency check failed for: " + ", ".join(failed_checks)
            )
        print("all-five consistency checks passed")


if __name__ == "__main__":
    main()