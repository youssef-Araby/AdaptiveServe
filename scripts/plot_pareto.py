#!/usr/bin/env python3
"""
Pareto plot: quality vs KV-cache compression for all configs.

Reads runs/C{0..5}/{model}/results.json (single-method baselines) and
runs/C6/{model}/results_tau{0.99,0.95,0.90}.json (router at 3 thresholds).

Plots:
  - C0..C5 as scatter points (one per fixed method)
  - C6 LOTO (OOD)         as a curve through the 3 tau points
  - C6 in-dist (random split) as a second curve

Saves to runs/figs/pareto_{model}.png

Usage:
  python scripts/plot_pareto.py --model llama3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO    = Path(__file__).resolve().parents[1]
RUNS    = REPO / "runs"
CONFIGS = ["C0", "C1", "C2", "C3", "C4", "C5"]
LABELS  = {
    "C0": "FP16",      "C1": "TailorKV", "C2": "QAQ",
    "C3": "KVQuant",   "C4": "DynamicKV","C5": "Ada-KV",
}
TAUS    = [0.99, 0.95, 0.90]


def load_single_method(model: str) -> list[tuple[str, float, float]]:
    out = []
    for cfg in CONFIGS:
        path = RUNS / cfg / model / "results.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        q  = d.get("longbench_avg")
        cr = d.get("longbench_avg_compression_ratio",
                   d.get("kv_compression_ratio", 1.0))
        if q is not None:
            out.append((cfg, float(q), float(cr)))
    return out


def load_router(model: str):
    loto, split = [], []
    for tau in TAUS:
        path = RUNS / "C6" / model / f"results_tau{tau}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        loto.append((tau, d["longbench_avg"], d["longbench_avg_compression_ratio"]))
        sp = d.get("eval_split_70_30", {})
        if sp:
            split.append((tau, sp["longbench_avg"], sp["longbench_avg_compression_ratio"]))
    return loto, split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llama3", "phi3"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    singles = load_single_method(args.model)
    loto, split = load_router(args.model)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Single-method baselines
    for cfg, q, cr in singles:
        marker = "s" if cfg == "C0" else "o"
        ax.scatter(cr, q, s=80, marker=marker, zorder=3,
                   label=f"{cfg} {LABELS[cfg]}")
        ax.annotate(cfg, (cr, q), textcoords="offset points",
                    xytext=(6, 4), fontsize=9)

    # Router curves
    if loto:
        crs = [c for _, _, c in loto]
        qs  = [q for _, q, _ in loto]
        ax.plot(crs, qs, "--", marker="^", color="tab:red",
                label="C6 router  (LOTO / OOD)", zorder=4)
        for tau, q, cr in loto:
            ax.annotate(f"τ={tau}", (cr, q), textcoords="offset points",
                        xytext=(6, -10), fontsize=8, color="tab:red")
    if split:
        crs = [c for _, _, c in split]
        qs  = [q for _, q, _ in split]
        ax.plot(crs, qs, "-", marker="D", color="tab:green",
                label="C6 router  (in-distribution)", zorder=5)

    # C0 reference line
    c0 = next(((q, cr) for cfg, q, cr in singles if cfg == "C0"), None)
    if c0:
        ax.axhline(c0[0],          color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax.axhline(0.99 * c0[0],   color="gray", linestyle=":", linewidth=1, alpha=0.4)
        ax.text(ax.get_xlim()[1], c0[0], "  C0 (FP16)", color="gray", fontsize=8,
                va="center")

    ax.set_xscale("log")
    ax.set_xlabel("KV cache compression ratio (×, log scale)")
    ax.set_ylabel("LongBench mean quality")
    ax.set_title(f"AdaptiveServe Pareto frontier — {args.model}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    out = Path(args.out) if args.out else RUNS / "figs" / f"pareto_{args.model}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
