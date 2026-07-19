#!/usr/bin/env python3
"""
Pareto plot using 10-fold stratified cross-validation for the router curve.

For each model, runs the router under stratified k-fold CV at three iso-quality
thresholds, then plots:
  - Fixed methods c0..c5 as scatter points (computed on full 220 prompts)
  - Router (CV) curve in green (each prompt routed by a model that didn't
    train on it; aggregated over all 220)
  - Router (LOTO) curve in red dashed (held-out task; loaded from existing
    C6 results JSON since LOTO already covers all 220)

Saves to runs/p0/figs/pareto_cv_{model}.png.

Usage:
  python scripts/plot_pareto_cv.py --model llama3
  python scripts/plot_pareto_cv.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_c6_classifier import (
    load_dataset, featurize, aggregate,
)
from cv_router_220 import cv_picks, fixed_on_full
from _common import assert_not_p0_output_path

RUNS = ROOT / "runs" / "p0"
CONFIGS = ["C0", "C1", "C2", "C3", "C4", "C5"]
LABELS = {
    "C0": "FP16", "C1": "TailorKV", "C2": "QAQ",
    "C3": "KVQuant", "C4": "DynamicKV", "C5": "Ada-KV",
}
TAUS = [0.99, 0.95, 0.90]
MODELS = ["phi3", "llama3", "llama32_3b", "llama31_8b"]


def model_title(m: str) -> str:
    return {
        "phi3": "Phi-3-mini",
        "llama3": "LLaMA-3-8B",
        "llama31_8b": "LLaMA-3.1-8B",
        "llama32_3b": "LLaMA-3.2-3B",
    }[m]


def load_loto(model: str):
    """LOTO results from existing C6 JSON (each prompt was in held-out task once)."""
    out = []
    for tau in TAUS:
        # Exact filename only — the old fuzzy fallback silently loaded the
        # tau=0.90 file when tau=0.95 was missing and plotted it mislabeled.
        p = RUNS / "C6" / model / f"results_tau{tau}.json"
        if not p.exists():
            print(f"  WARNING: {p} missing — LOTO point for tau={tau} skipped")
            continue
        d = json.loads(p.read_text())
        out.append((tau, d["longbench_avg"], d["longbench_avg_compression_ratio"]))
    return out


def compute_cv_curve(model: str):
    """Compute CV router (q, cr) at each tau."""
    rows = load_dataset(model)
    X = featurize(rows)
    pts = []
    for tau in TAUS:
        picks = cv_picks(rows, X, tau, n_splits=10, seed=0, use_floor=True)
        agg = aggregate(rows, picks)
        pts.append((tau, agg["longbench_avg"], agg["longbench_avg_compression_ratio"]))
        print(f"  {model} CV tau={tau}: q={agg['longbench_avg']:.4f}, "
              f"cr={agg['longbench_avg_compression_ratio']:.3f}x")
    return pts


def plot_one(model: str, out_path: Path):
    assert_not_p0_output_path(out_path)
    print(f"\n=== {model} ===")
    rows = load_dataset(model)
    fixed = fixed_on_full(rows)
    cv = compute_cv_curve(model)
    loto = load_loto(model)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Fixed-method scatter
    for cfg in CONFIGS:
        f = fixed[cfg]
        marker = "s" if cfg == "C0" else "o"
        ax.scatter(f["cr"], f["q"], s=80, marker=marker, zorder=3,
                   label=f"{cfg} {LABELS[cfg]}")
        ax.annotate(cfg, (f["cr"], f["q"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)

    # Router CV curve (in-distribution)
    if cv:
        crs = [c for _, _, c in cv]
        qs = [q for _, q, _ in cv]
        ax.plot(crs, qs, "-", marker="D", color="tab:green",
                label="Router (10-fold CV, in-dist)", zorder=5, linewidth=2)
        for tau, q, cr in cv:
            ax.annotate(f"τ={tau}", (cr, q),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=8, color="tab:green")

    # Router LOTO curve (cross-task OOD)
    if loto:
        crs = [c for _, _, c in loto]
        qs = [q for _, q, _ in loto]
        ax.plot(crs, qs, "--", marker="^", color="tab:red",
                label="Router (LOTO, cross-task)", zorder=4)
        for tau, q, cr in loto:
            ax.annotate(f"τ={tau}", (cr, q),
                        textcoords="offset points", xytext=(6, -12),
                        fontsize=8, color="tab:red")

    # FP16 reference lines
    c0 = fixed["C0"]
    ax.axhline(c0["q"], color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.axhline(0.99 * c0["q"], color="gray", linestyle=":", linewidth=1, alpha=0.4)

    ax.set_xscale("log")
    ax.set_xlabel("KV-cache compression ratio (×, log scale)")
    ax.set_ylabel("LongBench mean quality")
    ax.set_title(f"Quality–compression Pareto frontier — {model_title(model)}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    assert_not_p0_output_path(out_path.parent)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    assert_not_p0_output_path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default=None)
    ap.add_argument("--all", action="store_true", help="Render all 4 models")
    ap.add_argument("--outdir", default=str(RUNS / "figs"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    targets = MODELS if args.all else [args.model]
    if not targets or targets == [None]:
        ap.error("Specify --model X or --all")
    for m in targets:
        plot_one(m, outdir / f"pareto_cv_{m}.png")


if __name__ == "__main__":
    main()
