# Experiment Provenance

Last audited: 2026-07-16.

This repository contains several experiments with overlapping filenames. This
document identifies which artifacts may support a current AdaptiveServe-KV claim
and which are retained only for historical context.

## Current AdaptiveServe-KV Evidence

The corrected P0 rerun is the authoritative KV-cache experiment.

| Evidence | Location | Status |
| --- | --- | --- |
| P0 orchestration | `scripts/run_full_pipeline.sh` | Current reproducible workflow |
| P0 execution log | `runs/rerun_p0.log` | Completed 2026-07-16 |
| Raw fixed-method data | `runs/C0` through `runs/C5` | 220 LongBench prompts per model |
| Joined routing data | `runs/dataset/{model}.jsonl` | P0 measured bytes and compression |
| Primary router results | `runs/cv_router_220_tau*.json` | 10-fold task-stratified CV |
| Primary figures | `runs/figs/pareto_cv_*.png` | Current P0 figures |
| Candidate-pool results | `runs/candidate_pool_sweeps/p0_corrected_2026-07-16/` | Versioned 26-pool x 5-tau sweep |
| Human-readable report | `results.md` | Current P0 summary |

Fixed-method scripts ran P0 with `--skip-speed-ppl`. Therefore old speed, PPL,
VRAM, and throughput fields retained in `results.json` are not P0 measurements.
The primary compression measure is the harmonic mean of per-prompt measured
`kv_bytes_fp16 / kv_bytes`, not an older run-level prefill summary.

## Candidate-Pool Interpretation

`scripts/candidate_pool_sweep.py` uses the current C6 router and the P0 JSONL
datasets. It caches fold fits/predictions and writes source hashes plus an
all-five equivalence check. It evaluates calibration candidates; it does not
turn selection over the same CV aggregate into a nested-CV generalization claim.

The May pool figures and the claims in the local Overleaf draft were generated
before P0. Their source evaluator and serialized result matrix are not present
in this repository, so they are historical evidence only.

## Historical Archive

The cleanup archive under `runs/legacy/` is intentionally retained, not deleted.

| Archive | Contents |
| --- | --- |
| `pre_p0_2026-05/c6/` | Unversioned C6, filtered, and old tau-sweep outputs |
| `pre_p0_2026-05/figs/` | May Pareto and candidate-pool figures |
| `pre_p0_2026-05/logs/` | Historical orchestration logs and smoke diagnostics |
| `pre_p0_2026-07-15/` | Snapshot made immediately before P0, including the superseded June CSV/ZIP export |

Current P0 CSV/ZIP exports remain in `runs/dataset/` and include
`manifest.json` with source hashes.

## Independent Tracks

- `runs/mtbench/`, `benchmark_mtbench.py`, `judge_mtbench.py`, and
  `benchmark_bestroute.py` form a separate short-prompt model-routing study.
  They are not evidence for KV-cache C0-C6 results.
- `notebooks/`, `notebooks/results/`, and `docs/wanda_vllm_findings.md` form a
  separate R-Sparse/Wanda pruning and vLLM study. Its results must not be mixed
  with KV-cache measurements.
- `overleaf/` is ignored by Git and contains a pre-P0 paper draft. See
  [paper_draft_status.md](paper_draft_status.md) before using it.

## Claim Rules

1. Cite P0 JSONL, primary CV JSON, P0 sweep manifests, or `results.md` for
   corrected KV-cache statements.
2. Do not cite the legacy candidate-pool figures, old README numbers, or the
   local Overleaf draft as P0 evidence.
3. Treat `pareto_split_*.png`, LOTO outputs, and the 66-prompt comparison as
   supplementary diagnostics; use `pareto_cv_*.png` and `cv_router_220_tau*.json`
   for primary fixed-vs-router comparisons.
4. Keep independent MT-Bench and pruning claims separate from C0-C6 claims.