#!/usr/bin/env python3
"""
Recompute fixed-method (c0-c5) baselines on the SAME 66 test prompts that
the router (C6) is evaluated on, using the same seed=0 70/30 split.

Addresses prof comment #3: the original Table V compared router results on
66 held-out prompts against fixed-method averages over all 220 prompts
(including the 154 the router trained on). This script produces the fair
apples-to-apples comparison.

Output: prints a per-model table of c0-c5 quality (arithmetic mean) and
compression (harmonic mean) on the 66 test prompts, suitable for replacing
the comparison columns of paper Table V.
"""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from _common import assert_not_p0_output_path

ROOT = Path(__file__).resolve().parents[1]
P0_RUNS = ROOT / "runs" / "p0"
CONFIGS = ["C0", "C1", "C2", "C3", "C4", "C5"]
MODELS = ["phi3", "llama3", "llama32_3b", "llama31_8b"]


def load_rows(model: str) -> list[dict]:
    path = P0_RUNS / "dataset" / f"{model}.jsonl"
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def test_split(n: int, seed: int = 0, train_frac: float = 0.7) -> list[int]:
    """Reproduce the exact test indices used by benchmark_c6_classifier.split_picks."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_tr = int(train_frac * n)
    return idx[n_tr:].tolist()


def harmonic_mean(xs: list[float]) -> float:
    return statistics.harmonic_mean([x for x in xs if x > 0])


def fixed_on_subset(rows: list[dict], indices: list[int]) -> dict:
    """For each config, compute mean quality and harmonic-mean compression on the
    selected prompt indices."""
    out = {}
    for c in CONFIGS:
        qs = [rows[i]["scores"][c] for i in indices]
        crs = [rows[i]["compression"][c] for i in indices]
        out[c] = {
            "q": round(sum(qs) / len(qs), 4),
            "cr": round(harmonic_mean(crs), 3),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_path = P0_RUNS / "fair_comparison_66.json"
    assert_not_p0_output_path(out_path)

    print(f"\n=== Fair comparison: fixed methods on the same 66 test prompts (seed={args.seed}) ===\n")

    summary = {}
    for model in MODELS:
        rows = load_rows(model)
        te = test_split(len(rows), seed=args.seed)
        print(f"  {model}: {len(rows)} rows, {len(te)} test prompts")
        summary[model] = {
            "n_total": len(rows),
            "n_test": len(te),
            "fixed_on_test": fixed_on_subset(rows, te),
            "fixed_on_full": fixed_on_subset(rows, list(range(len(rows)))),
        }
    print()

    # Side-by-side table per model: (test 66) vs (full 220).
    for model in MODELS:
        s = summary[model]
        print(f"\n{model}")
        print(f"  {'config':<3}  {'q@test66':>9} / {'cr@test66':>9}    "
              f"{'q@full220':>9} / {'cr@full220':>9}    {'Δq':>6}")
        for c in CONFIGS:
            t, f = s["fixed_on_test"][c], s["fixed_on_full"][c]
            dq = t["q"] - f["q"]
            print(f"  {c:<3}  {t['q']:>9.4f} / {t['cr']:>9.3f}x   "
                  f"{f['q']:>9.4f} / {f['cr']:>9.3f}x   {dq:>+6.4f}")

    assert_not_p0_output_path(out_path)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
