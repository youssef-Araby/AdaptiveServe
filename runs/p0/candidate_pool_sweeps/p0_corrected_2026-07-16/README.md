# Corrected P0 Candidate-Pool Sweep

- Run ID: `p0_corrected_2026-07-16`
- Schema: `adaptive-serve-candidate-pool-sweep/v1`
- Generated: `2026-07-16T08:09:53+00:00`
- Protocol: 10-fold task-stratified CV; fold regressors and predictions are cached once, then reused for every pool/tau pair.
- This is an evaluation artifact. `best_by_quality` is a transparent ranking, not a deployment recommendation without a chosen quality/compression objective.
- Pool/tau selection uses the same CV aggregate shown here; use a separate calibration workload or nested CV for an unbiased post-selection generalization estimate.

## Results

| Model | Evaluated points | Best by quality | Best strictly dominating C0 | Pareto points |
| --- | ---: | --- | --- | ---: |
| phi3 | 130 | C2+C3 at tau=0.99 (q=0.3184, cr=3.659x) | C2+C3 at tau=0.99 (q=0.3184, cr=3.659x) | 16 |
| llama3 | 130 | C2+C5 at tau=0.85 (q=0.4337, cr=4.936x) | C2+C5 at tau=0.85 (q=0.4337, cr=4.936x) | 18 |
| llama32_3b | 130 | C2+C4+C5 at tau=0.99 (q=0.4069, cr=4.607x) | C2+C4+C5 at tau=0.99 (q=0.4069, cr=4.607x) | 14 |
| llama31_8b | 130 | C2+C3 at tau=0.99 (q=0.4374, cr=3.842x) | C2+C3 at tau=0.99 (q=0.4374, cr=3.842x) | 13 |

## Files

- `summary.json`: complete all-model result and provenance manifest.
- `<model>.json`: per-model subset of the manifest.
- `figs/candidate_pools_<model>.png`: P0-corrected pool/tau operating points.

The historical May candidate-pool figures remain separate pre-P0 artifacts and must not be used as corrected P0 evidence.
