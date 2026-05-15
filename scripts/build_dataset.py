"""
Build a per-prompt routing dataset from runs/C{0..5}/{model}/per_prompt.jsonl.

Joins the 6 configs by (task, sample_idx). Output rows:

  {
    "task":       "narrativeqa",
    "sample_idx": 0,
    "features":   {...prompt-intrinsic features (from C0's row)...},
    "scores":     {"C0": 0.44, "C1": 0.30, ... "C5": 0.41},
    "compression": {"C0": 1.0, "C1": 34.6, ...},   # constant per config
    "best_quality":      "C3",        # argmax score (ties: lowest config id)
    "best_quality_score": 0.50,
    "best_iso_quality_99": "C1",      # most compression s.t. score >= 0.99 * C0
    "best_iso_quality_95": "C4",
    "best_iso_quality_90": "C1",
  }

Saved to runs/dataset/{model}.jsonl.

Usage:
  python scripts/build_dataset.py --model llama3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
RUNS    = ROOT / "runs"
CONFIGS = ["C0", "C1", "C2", "C3", "C4", "C5"]
TAUS    = [0.99, 0.95, 0.90]


def _load_per_prompt(model: str) -> dict[str, list[dict]]:
    out = {}
    for cfg in CONFIGS:
        path = RUNS / cfg / model / "per_prompt.jsonl"
        if not path.exists():
            print(f"  warn: missing {path}")
            continue
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        out[cfg] = rows
    return out


def _config_compression(model: str) -> dict[str, float]:
    out = {}
    for cfg in CONFIGS:
        path = RUNS / cfg / model / "results.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        out[cfg] = float(d.get("longbench_avg_compression_ratio",
                               d.get("kv_compression_ratio", 1.0)))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["llama3", "phi3", "llama31_8b", "llama32_3b"])
    p.add_argument("--out",   default=None)
    args = p.parse_args()

    pp = _load_per_prompt(args.model)
    cr = _config_compression(args.model)
    if not pp or "C0" not in pp:
        raise SystemExit(f"no C0 per_prompt log for {args.model} — run benchmarks first")

    # Index by (task, sample_idx)
    idx: dict[tuple[str, int], dict[str, dict]] = {}
    for cfg, rows in pp.items():
        for r in rows:
            key = (r["task"], r["sample_idx"])
            idx.setdefault(key, {})[cfg] = r

    rows_out = []
    skipped  = 0
    for (task, sidx), cfg_rows in sorted(idx.items()):
        if not all(c in cfg_rows for c in CONFIGS):
            skipped += 1
            continue
        scores  = {c: cfg_rows[c]["score"] for c in CONFIGS}
        feats   = dict(cfg_rows["C0"]["features"])   # prompt-intrinsic, same across configs

        # Best by quality (ties → lowest-numbered config = simplest)
        best_q = max(CONFIGS, key=lambda c: (scores[c], -CONFIGS.index(c)))

        # Iso-quality oracles: max compression s.t. score >= tau * C0[task,sidx]
        c0_score = scores["C0"]
        iso = {}
        for tau in TAUS:
            cands = [c for c in CONFIGS if scores[c] >= tau * c0_score]
            if not cands:
                cands = ["C0"]
            iso[f"best_iso_quality_{int(tau*100)}"] = max(cands, key=lambda c: cr.get(c, 1.0))

        rows_out.append({
            "task":               task,
            "sample_idx":         sidx,
            "metric":             cfg_rows["C0"]["metric"],
            "features":           feats,
            "scores":             scores,
            "compression":        cr,
            "best_quality":       best_q,
            "best_quality_score": scores[best_q],
            **iso,
        })

    out_path = Path(args.out) if args.out else RUNS / "dataset" / f"{args.model}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r) for r in rows_out) + "\n")
    print(f"wrote {len(rows_out)} rows → {out_path}  (skipped {skipped} incomplete)")

    # ---- Training-data summary (what the classifier actually sees) ----
    from collections import Counter
    from statistics import mean, harmonic_mean

    if not rows_out:
        return
    feat_keys = sorted(rows_out[0]["features"].keys())
    print(f"\nfeatures per prompt ({len(feat_keys)}): {feat_keys}")
    print(f"compression ratios: " + "  ".join(f"{c}={cr.get(c,1.0):.2f}x" for c in CONFIGS))

    print("\nper-config quality on training data (mean over all prompts):")
    for c in CONFIGS:
        m = mean(r["scores"][c] for r in rows_out)
        print(f"  always-{c}: q={m:.4f}  cr={cr.get(c,1.0):.2f}x")

    print("\nper-prompt oracle (upper bound the classifier targets):")
    for tau in TAUS:
        key = f"best_iso_quality_{int(tau*100)}"
        picks = [r[key] for r in rows_out]
        q = mean(r["scores"][c] for r, c in zip(rows_out, picks))
        cr_hm = harmonic_mean(r["compression"][c] for r, c in zip(rows_out, picks))
        print(f"  oracle@tau={tau:.2f}: q={q:.4f}  cr={cr_hm:.2f}x")

    print("\nlabel distribution (which config wins):")
    for label_key in ["best_quality"] + [f"best_iso_quality_{int(t*100)}" for t in TAUS]:
        dist = Counter(r[label_key] for r in rows_out)
        print(f"  {label_key:<22s}  " + "  ".join(f"{c}={dist.get(c,0)}" for c in CONFIGS))


if __name__ == "__main__":
    main()
