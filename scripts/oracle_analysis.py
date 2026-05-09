"""
Oracle analysis: per-task best-config selector.

Loads runs/C{0..5}/{model}/results.json and asks:

  - Always-Cx baselines: avg quality and compression of each fixed config.
  - Quality oracle: for each task, pick the config with the highest score.
                    Average over tasks. Compression = harmonic mean of chosen configs.
  - Iso-quality oracle: for each task, pick the config with the highest compression
                        whose score >= tau * C0[task].
  - Iso-compression oracle (per task): pick the config with the highest score whose
                        compression_ratio >= R_min.

Compression ratio per config is taken from longbench_avg_compression_ratio when
available, else kv_compression_ratio. Per-task compression is not measured, so
the oracle compression is the *config's average* compression ratio applied to
that task.

Effective compression across tasks is computed as a harmonic mean (1 / mean(1/cr))
because compression is a memory-saving ratio, and equal-task-weight averaging
of the inverse (= retained-fraction) is the right combiner.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
RUNS    = ROOT / "runs"
CONFIGS = ["C0", "C1", "C2", "C3", "C4", "C5"]
MODELS  = ["llama3", "phi3"]

LABELS = {
    "C0": "FP16",
    "C1": "TailorKV",
    "C2": "QAQ",
    "C3": "KVQuant",
    "C4": "DynamicKV",
    "C5": "Ada-KV",
}

# Quality thresholds for iso-quality oracle (fraction of C0 baseline per task).
TAUS = [1.00, 0.99, 0.95, 0.90]

# Compression thresholds for iso-compression oracle.
R_MINS = [1.0, 3.0, 5.0, 10.0, 20.0]


def _load(model: str) -> dict[str, dict]:
    out = {}
    for cfg in CONFIGS:
        path = RUNS / cfg / model / "results.json"
        if not path.exists():
            continue
        out[cfg] = json.loads(path.read_text())
    return out


def _compression(d: dict) -> float:
    return float(d.get("longbench_avg_compression_ratio",
                       d.get("kv_compression_ratio", 1.0)))


def _per_task_scores(d: dict) -> dict[str, float]:
    lb = d.get("longbench", {})
    return {task: float(v["score"]) for task, v in lb.items()
            if isinstance(v, dict) and "score" in v}


def _avg(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _harmonic_compression(crs: list[float]) -> float:
    """Equal-weight harmonic mean: 1 / mean(1/cr)."""
    if not crs:
        return float("nan")
    return 1.0 / _avg([1.0 / max(c, 1e-9) for c in crs])


def analyze_model(model: str) -> None:
    print(f"\n{'='*72}\n  {model.upper()}\n{'='*72}")
    runs = _load(model)
    if "C0" not in runs:
        print(f"  no C0 baseline for {model}; skipping")
        return

    tasks = sorted(_per_task_scores(runs["C0"]).keys())
    score = {cfg: _per_task_scores(runs[cfg]) for cfg in runs}
    cr    = {cfg: _compression(runs[cfg])     for cfg in runs}

    # ---- per-task table -----------------------------------------------------
    print("\n  per-task scores (rows=task, cols=config):")
    hdr = " task           " + "  ".join(f"{cfg:>7s}" for cfg in CONFIGS if cfg in runs)
    print(hdr)
    for t in tasks:
        row = f"  {t:<14s}"
        for cfg in CONFIGS:
            if cfg in runs:
                row += f"  {score[cfg].get(t, float('nan')):7.4f}"
        print(row)
    row = "  COMPRESSION   "
    for cfg in CONFIGS:
        if cfg in runs:
            row += f"  {cr[cfg]:6.2f}×"
    print(row)

    # ---- baselines (always-Cx) ---------------------------------------------
    print("\n  always-Cx baselines:")
    print(f"    {'config':<12s}  {'avg quality':>12s}  {'compression':>12s}")
    for cfg in CONFIGS:
        if cfg not in runs:
            continue
        q = _avg([score[cfg].get(t, 0.0) for t in tasks])
        print(f"    {cfg} {LABELS[cfg]:<8s}  {q:12.4f}  {cr[cfg]:11.2f}×")

    # ---- quality oracle (max quality per task; ignore compression) ---------
    picks_q, qs_q, crs_q = [], [], []
    for t in tasks:
        best_cfg = max(score, key=lambda c: score[c].get(t, -1))
        picks_q.append((t, best_cfg))
        qs_q.append(score[best_cfg][t])
        crs_q.append(cr[best_cfg])
    print(f"\n  QUALITY oracle  (best score per task, any compression):")
    print(f"    avg quality = {_avg(qs_q):.4f}   compression = {_harmonic_compression(crs_q):.2f}×")
    print(f"    picks: " + ", ".join(f"{t}->{c}" for t, c in picks_q))

    # ---- iso-quality oracle: max compression s.t. score >= tau * C0 -------
    print(f"\n  ISO-QUALITY oracle (max compression s.t. score >= tau * C0):")
    print(f"    {'tau':>6s}  {'avg quality':>12s}  {'compression':>12s}  picks")
    c0_scores = score["C0"]
    for tau in TAUS:
        picks, qs, crs = [], [], []
        for t in tasks:
            floor = tau * c0_scores.get(t, 0.0)
            cands = [c for c in score if score[c].get(t, -1) >= floor]
            if not cands:
                cands = ["C0"]
            best = max(cands, key=lambda c: cr[c])
            picks.append((t, best))
            qs.append(score[best][t])
            crs.append(cr[best])
        pick_str = " ".join(f"{t[:6]}->{c}" for t, c in picks)
        print(f"    {tau:6.2f}  {_avg(qs):12.4f}  {_harmonic_compression(crs):11.2f}×  {pick_str}")

    # ---- iso-compression oracle: max quality s.t. cr >= R_min -------------
    print(f"\n  ISO-COMPRESSION oracle (max quality s.t. cr >= R_min):")
    print(f"    {'R_min':>6s}  {'avg quality':>12s}  {'compression':>12s}  picks")
    for r_min in R_MINS:
        eligible = [c for c in score if cr[c] >= r_min]
        if not eligible:
            print(f"    {r_min:6.2f}  (no config meets threshold)")
            continue
        picks, qs, crs = [], [], []
        for t in tasks:
            best = max(eligible, key=lambda c: score[c].get(t, -1))
            picks.append((t, best))
            qs.append(score[best][t])
            crs.append(cr[best])
        pick_str = " ".join(f"{t[:6]}->{c}" for t, c in picks)
        print(f"    {r_min:6.2f}  {_avg(qs):12.4f}  {_harmonic_compression(crs):11.2f}×  {pick_str}")

    # ---- which configs are NEVER picked by any oracle? --------------------
    used = set()
    for t in tasks:
        used.add(max(score, key=lambda c: score[c].get(t, -1)))
        for tau in TAUS:
            cands = [c for c in score if score[c].get(t, -1) >= tau * c0_scores.get(t, 0.0)] or ["C0"]
            used.add(max(cands, key=lambda c: cr[c]))
        for r_min in R_MINS:
            elig = [c for c in score if cr[c] >= r_min]
            if elig:
                used.add(max(elig, key=lambda c: score[c].get(t, -1)))
    dominated = [c for c in CONFIGS if c in runs and c not in used]
    print(f"\n  configs NEVER chosen by any oracle on any task: {dominated or '(none)'}")


def main() -> None:
    for m in MODELS:
        analyze_model(m)


if __name__ == "__main__":
    main()
