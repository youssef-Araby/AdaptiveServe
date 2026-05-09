"""
Per-prompt oracle analysis: re-runs the oracle from oracle_analysis.py at
prompt granularity using runs/dataset/{model}.jsonl.

Reports:
  - Always-Cx baselines at prompt granularity (mean score, harmonic compression).
  - Quality oracle    : pick best config per prompt.
  - Iso-quality oracle: pick max-compression config s.t. score >= tau * C0[prompt].
  - Iso-comp oracle   : pick max-quality config s.t. compression >= R_min.
  - Per-task breakdown of which config the iso-quality oracle picks.

Usage:
  python scripts/oracle_per_prompt.py --model llama3
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
CONFIGS = ["C0", "C1", "C2", "C3", "C4", "C5"]
LABELS  = {"C0": "FP16", "C1": "TailorKV", "C2": "QAQ",
           "C3": "KVQuant", "C4": "DynamicKV", "C5": "Ada-KV"}
TAUS    = [1.00, 0.99, 0.95, 0.90]
R_MINS  = [1.0, 3.0, 5.0, 10.0, 20.0]


def _harmonic_compression(crs: list[float]) -> float:
    if not crs:
        return float("nan")
    return 1.0 / (sum(1.0 / max(c, 1e-9) for c in crs) / len(crs))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["llama3", "phi3"])
    args = p.parse_args()

    path = ROOT / "runs" / "dataset" / f"{args.model}.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"empty dataset {path} — run build_dataset.py first")

    cr = rows[0]["compression"]
    print(f"\n{'='*72}\n  {args.model.upper()}  (n={len(rows)} prompts)\n{'='*72}")

    # ---- always-Cx baselines ------------------------------------------------
    print("\n  always-Cx baselines (prompt-mean):")
    print(f"    {'config':<12s}  {'mean quality':>13s}  {'compression':>12s}")
    for c in CONFIGS:
        if c not in cr:
            continue
        q = sum(r["scores"][c] for r in rows) / len(rows)
        print(f"    {c} {LABELS[c]:<8s}  {q:13.4f}  {cr[c]:11.2f}×")

    # ---- quality oracle ----------------------------------------------------
    qs, crs = [], []
    picks = Counter()
    for r in rows:
        best = max(r["scores"], key=lambda c: r["scores"][c])
        qs.append(r["scores"][best])
        crs.append(cr[best])
        picks[best] += 1
    print(f"\n  QUALITY oracle (max score per prompt):")
    print(f"    mean quality = {sum(qs)/len(qs):.4f}   "
          f"compression = {_harmonic_compression(crs):.2f}×")
    print(f"    picks: " + "  ".join(f"{c}={picks.get(c,0)}" for c in CONFIGS))

    # ---- iso-quality oracle ------------------------------------------------
    print(f"\n  ISO-QUALITY oracle (max compression s.t. score >= tau * C0):")
    print(f"    {'tau':>6s}  {'mean quality':>13s}  {'compression':>12s}  picks")
    for tau in TAUS:
        qs, crs = [], []
        picks = Counter()
        for r in rows:
            c0 = r["scores"]["C0"]
            cands = [c for c in CONFIGS if r["scores"][c] >= tau * c0]
            if not cands:
                cands = ["C0"]
            best = max(cands, key=lambda c: cr.get(c, 1.0))
            qs.append(r["scores"][best])
            crs.append(cr[best])
            picks[best] += 1
        pick_str = "  ".join(f"{c}={picks.get(c,0)}" for c in CONFIGS)
        print(f"    {tau:6.2f}  {sum(qs)/len(qs):13.4f}  "
              f"{_harmonic_compression(crs):11.2f}×  {pick_str}")

    # ---- iso-compression oracle --------------------------------------------
    print(f"\n  ISO-COMPRESSION oracle (max quality s.t. cr >= R_min):")
    print(f"    {'R_min':>6s}  {'mean quality':>13s}  {'compression':>12s}  picks")
    for r_min in R_MINS:
        elig = [c for c in CONFIGS if cr.get(c, 0) >= r_min]
        if not elig:
            print(f"    {r_min:6.2f}  (no config meets threshold)")
            continue
        qs, crs = [], []
        picks = Counter()
        for r in rows:
            best = max(elig, key=lambda c: r["scores"][c])
            qs.append(r["scores"][best])
            crs.append(cr[best])
            picks[best] += 1
        pick_str = "  ".join(f"{c}={picks.get(c,0)}" for c in CONFIGS)
        print(f"    {r_min:6.2f}  {sum(qs)/len(qs):13.4f}  "
              f"{_harmonic_compression(crs):11.2f}×  {pick_str}")

    # ---- per-task breakdown of iso-quality τ=0.99 oracle picks --------------
    print(f"\n  per-task picks (iso-quality tau=0.99):")
    by_task: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        c0 = r["scores"]["C0"]
        cands = [c for c in CONFIGS if r["scores"][c] >= 0.99 * c0] or ["C0"]
        best = max(cands, key=lambda c: cr.get(c, 1.0))
        by_task[r["task"]][best] += 1
    hdr = "    task            " + "  ".join(f"{c:>4s}" for c in CONFIGS)
    print(hdr)
    for t in sorted(by_task):
        row = f"    {t:<14s}  " + "  ".join(f"{by_task[t].get(c,0):>4d}" for c in CONFIGS)
        print(row)


if __name__ == "__main__":
    main()
