# Legacy AdaptiveServe Artifacts

This directory retains superseded experiment outputs for reproducibility and
forensics. Nothing here is deleted automatically, but these paths are not
current AdaptiveServe-KV evidence.

| Directory | Provenance | Do not use for |
| --- | --- | --- |
| `pre_p0_2026-05/c6/` | Pre-correction router outputs and ablations | P0 C6 values or router claims |
| `pre_p0_2026-05/figs/` | May Pareto and candidate-pool figures | Corrected pool/tau figures or paper tables |
| `pre_p0_2026-05/logs/` | Historical pipeline and smoke logs | Current execution status |
| `pre_p0_2026-07-15/` | Snapshot immediately before P0, including June CSV/ZIP exports | Corrected rerun evidence |
| `orphaned_bytecode_2026-07-17/` | Three source-less Python 3.12 bytecode files preserved during cleanup | Executable source or current experimental evidence |

See the parent [`runs/README.md`](../README.md) for the current run namespace
map.
