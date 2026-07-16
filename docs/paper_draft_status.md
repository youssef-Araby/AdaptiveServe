# Paper Draft Status

Last assessed: 2026-07-16.

The local, Git-ignored `overleaf/` directory is a **pre-P0 paper draft**. Its
headline results, candidate-pool table, figures, microsecond overhead claim, and
strong Pareto-dominance claims were generated before the corrected P0 rerun.

Do not cite, submit, or treat `overleaf/main.tex` as evidence for current
AdaptiveServe-KV results. The corrected source of truth is:

- [results.md](../results.md)
- [P0 primary CV results](../runs/cv_router_220_tau0.99.json)
- [corrected candidate-pool sweep](../runs/candidate_pool_sweeps/p0_corrected_2026-07-16)
- [experiment provenance](provenance.md)

The historical candidate-pool evaluator and serialized result matrix are absent
from the repository. The replacement
[`candidate_pool_sweep.py`](../scripts/candidate_pool_sweep.py) uses P0 data and
records data hashes plus all-five consistency checks. Its highest-quality points
are calibration candidates selected on the same CV aggregate, not nested-CV
post-selection estimates.

Before the paper is refreshed, replace all pre-P0 numerical claims and figures,
update method descriptions to match the current simplified implementations, and
state the calibration-versus-post-selection limitation for pool/tau selection.