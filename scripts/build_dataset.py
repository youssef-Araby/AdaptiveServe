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

# Per-model max input length (matches LB_MAX_INPUT in _common.py — the head+tail
# truncation each benchmark applies before feeding the prompt to the model).
LB_MAX_INPUT: dict[str, int] = {
    "phi3":       3500,
    "llama3":     7500,
    "llama31_8b": 7500,
    "llama32_3b": 7500,
}


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


def _load_run_stats(model: str) -> dict[str, dict]:
    """Return the full results.json for each config (for per-prompt compression)."""
    out = {}
    for cfg in CONFIGS:
        path = RUNS / cfg / model / "results.json"
        if not path.exists():
            continue
        out[cfg] = json.loads(path.read_text())
    return out


def per_prompt_compression(
    config: str, seq_len: int, stats: dict, model_max_input: int
) -> float:
    """Compression ratio for ONE prompt under ONE config.

    Derived from each method's known per-prompt semantics rather than the
    run-level average, so the router sees real prompt-length-dependent
    behaviour (e.g. C4 compresses 7x on a 7000-token prompt and 1x on a
    500-token prompt — the run-level average of 6.05x would hide both).

      C0: trivially 1x (no compression).
      C3: ~constant (4-bit + 1% outliers → ~3.41x regardless of prompt). Use
          the run-level number from results.json.
      C4: per-layer length = min(seq_len, max_capacity). Ratio is exactly
          seq_len / min(seq_len, max_capacity).
      C5: same per-layer length as C4 (vote-aggregated to uniform).
      C1: per-layer length pinned to (n_local + n_topk) once seq_len exceeds
          that threshold. Compressed bytes are then ~constant in seq_len, so
          ratio scales LINEARLY in seq_len above the threshold. Below the
          threshold only the 1-bit Q-layer quant contributes (≈3% saving on
          1/32 layers), giving ratio ≈ 1.03 — used as a floor.
      C2: data-dependent (per-token-head variable bits driven by K/V dynamic
          range). Not recoverable from features. Falls back to the run-level
          average. Will be fixed once C2 logs per-prompt avg_bits.
    """
    eff_len = max(1, min(seq_len, model_max_input))

    if config == "C0":
        return 1.0

    if config == "C3":
        return float(stats.get("longbench_avg_compression_ratio",
                               stats.get("kv_compression_ratio", 3.412)))

    if config == "C4":
        budget = int(stats.get("max_capacity", 1024))
        return eff_len / max(min(eff_len, budget), 1)

    if config == "C5":
        budget = int(stats.get("budget_per_head", 1024))
        return eff_len / max(min(eff_len, budget), 1)

    if config == "C1":
        prune_thr = int(stats.get("n_local", 64)) + int(stats.get("n_topk", 128))
        if eff_len <= prune_thr:
            return 1.03  # no pruning; only Q-layer 1-bit quant on 1/n_layers
        avg_seq   = float(stats.get("longbench_avg_seq_len") or eff_len)
        avg_ratio = float(stats.get("longbench_avg_compression_ratio",
                                    stats.get("kv_compression_ratio", 1.0)))
        # Linear-in-seq_len scaling, calibrated by the run-level average.
        return avg_ratio * eff_len / max(avg_seq, 1.0)

    if config == "C2":
        # TODO(#1, follow-up): log per-prompt avg_bits in benchmark_c2_qaq.py
        # and use here. For now: run-level average.
        return float(stats.get("longbench_avg_compression_ratio",
                               stats.get("kv_compression_ratio", 1.0)))

    raise ValueError(f"unknown config: {config}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["llama3", "phi3", "llama31_8b", "llama32_3b"])
    p.add_argument("--out",   default=None)
    args = p.parse_args()

    pp    = _load_per_prompt(args.model)
    stats = _load_run_stats(args.model)
    if not pp or "C0" not in pp:
        raise SystemExit(f"no C0 per_prompt log for {args.model} — run benchmarks first")
    if args.model not in LB_MAX_INPUT:
        raise SystemExit(f"no LB_MAX_INPUT for model {args.model}; update build_dataset.py")
    max_input = LB_MAX_INPUT[args.model]

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

        # Per-prompt compression:
        #   1. If the benchmark logged `compression` directly in per_prompt.jsonl
        #      (preferred — e.g. C2 logs measured avg_bits-derived ratio), use it.
        #   2. Otherwise derive from method semantics + seq_len (see
        #      per_prompt_compression() — exact for C0/C3/C4/C5, calibrated for C1).
        seq_len = int(feats.get("seq_len_tokens", 0))
        cr_per_prompt = {}
        for c in CONFIGS:
            logged = cfg_rows[c].get("compression")
            if logged is not None:
                cr_per_prompt[c] = round(float(logged), 4)
            else:
                cr_per_prompt[c] = round(
                    per_prompt_compression(c, seq_len, stats.get(c, {}), max_input), 4
                )

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
