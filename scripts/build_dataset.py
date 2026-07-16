"""
Build a per-prompt routing dataset from runs/C{0..5}/{model}/per_prompt.jsonl.

Joins the 6 configs by (task, sample_idx). Output rows:

  {
    "task":       "narrativeqa",
    "sample_idx": 0,
    "features":   {...prompt-intrinsic features (from C0's row)...},
    "scores":     {"C0": 0.44, "C1": 0.30, ... "C5": 0.41},
    "compression": {"C0": 1.0, "C1": 34.6, ...},   # MEASURED per prompt
    "kv_bytes":      {"C0": 1.2e8, ...},           # effective compressed KV bytes
    "kv_bytes_fp16": {"C0": 1.2e8, ...},           # FP16-reference KV bytes
    "best_quality":      "C3",        # argmax score (ties: lowest config id)
    "best_quality_score": 0.50,
    "best_iso_quality_99": "C1",      # most compression s.t. score >= 0.99 * C0
    "best_iso_quality_95": "C4",
    "best_iso_quality_90": "C1",
  }

Per-prompt compression comes EXCLUSIVELY from the `compression` field each
benchmark logs (computed from measured kv_bytes / kv_bytes_fp16 — see the
unified CR basis in the benchmark scripts). There are no analytic fallbacks:
a per_prompt.jsonl row without kv_bytes means the config was produced by an
old script version and must be re-run.

Saved to runs/dataset/{model}.jsonl.

Usage:
  python scripts/build_dataset.py --model llama3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
RUNS    = ROOT / "runs"
CONFIGS = ["C0", "C1", "C2", "C3", "C4", "C5"]
TAUS    = [0.99, 0.95, 0.90]

# Fields every per_prompt.jsonl row must now carry (measured-CR basis).
REQUIRED_CR_FIELDS = ("kv_bytes", "kv_bytes_fp16", "compression")


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["llama3", "phi3", "llama31_8b", "llama32_3b"])
    p.add_argument("--out",   default=None)
    args = p.parse_args()

    pp = _load_per_prompt(args.model)
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

        # Guard: (task, sample_idx) must be the SAME underlying prompt in every
        # config's log. A config re-run with a different ADAPTIVESERVE_LB_N
        # subsamples different records and would silently join mismatched
        # prompts; seq_len_chars is tokenizer-independent, so it must agree.
        for c in CONFIGS:
            if cfg_rows[c]["features"].get("seq_len_chars") != feats.get("seq_len_chars"):
                sys.exit(
                    f"ERROR: prompt mismatch at (task={task}, sample_idx={sidx}): "
                    f"config {c} logged seq_len_chars="
                    f"{cfg_rows[c]['features'].get('seq_len_chars')} vs C0's "
                    f"{feats.get('seq_len_chars')} — configs were run with "
                    f"different LongBench subsampling (ADAPTIVESERVE_LB_N); re-run."
                )

        # Per-prompt MEASURED compression: every benchmark logs kv_bytes,
        # kv_bytes_fp16 and compression (= kv_bytes_fp16 / kv_bytes) per prompt.
        # No analytic derivations, no run-level fallbacks — fail loudly instead.
        cr_per_prompt: dict[str, float] = {}
        kv_bytes:      dict[str, float] = {}
        kv_bytes_fp16: dict[str, float] = {}
        for c in CONFIGS:
            row = cfg_rows[c]
            missing = [f for f in REQUIRED_CR_FIELDS if row.get(f) is None]
            if missing:
                sys.exit(
                    f"ERROR: runs/{c}/{args.model}/per_prompt.jsonl row "
                    f"(task={task}, sample_idx={sidx}) lacks {'/'.join(missing)}. "
                    f"This log predates the measured-CR accounting — re-run config {c}."
                )
            cr_per_prompt[c] = round(float(row["compression"]), 4)
            kv_bytes[c]      = float(row["kv_bytes"])
            kv_bytes_fp16[c] = float(row["kv_bytes_fp16"])

        # Best by quality (ties → lowest-numbered config = simplest)
        best_q = max(CONFIGS, key=lambda c: (scores[c], -CONFIGS.index(c)))

        # Iso-quality oracles: max compression s.t. score >= tau * C0[task,sidx]
        c0_score = scores["C0"]
        iso = {}
        for tau in TAUS:
            cands = [c for c in CONFIGS if scores[c] >= tau * c0_score]
            if not cands:
                cands = ["C0"]
            iso[f"best_iso_quality_{int(tau*100)}"] = max(cands, key=lambda c: cr_per_prompt.get(c, 1.0))

        rows_out.append({
            "task":               task,
            "sample_idx":         sidx,
            "metric":             cfg_rows["C0"]["metric"],
            "features":           feats,
            "scores":             scores,
            "compression":        cr_per_prompt,
            "kv_bytes":           kv_bytes,
            "kv_bytes_fp16":      kv_bytes_fp16,
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
    cr_hmean = {
        c: harmonic_mean(r["compression"][c] for r in rows_out)
        for c in CONFIGS
    }
    cr_min = {c: min(r["compression"][c] for r in rows_out) for c in CONFIGS}
    cr_max = {c: max(r["compression"][c] for r in rows_out) for c in CONFIGS}
    print(f"per-prompt compression (harmonic mean): "
          + "  ".join(f"{c}={cr_hmean[c]:.2f}x" for c in CONFIGS))
    print(f"per-prompt compression (min .. max):   "
          + "  ".join(f"{c}={cr_min[c]:.2f}..{cr_max[c]:.2f}x" for c in CONFIGS))

    print("\nper-config quality on training data (mean over all prompts):")
    for c in CONFIGS:
        m = mean(r["scores"][c] for r in rows_out)
        print(f"  always-{c}: q={m:.4f}  cr_hmean={cr_hmean[c]:.2f}x")

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
