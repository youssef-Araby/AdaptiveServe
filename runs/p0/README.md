# Corrected P0 Run Archive

This directory contains the complete corrected P0 AdaptiveServe-KV experiment.
The 131 evidence artifacts were relocated here during the 2026-07-19
repository cleanup. The relocation changed their filesystem locations only;
the generated artifacts themselves were not rewritten.

## Read-only archive rule

Everything beneath `runs/p0/` is immutable evidence. Do not direct benchmark,
dataset, analysis, plotting, or export output into this directory. Active
Python writers enforce this rule through the shared
`assert_not_p0_output_path` guard and fail before creating, truncating, or
rewriting an archive path.

New experiments must use a fresh namespace outside this directory, such as
`runs/longbench16_24k/`. The files here may be read for reproduction checks
and historical comparison, but regenerated outputs belong in a separate run
directory.

## Evaluation scope

P0 evaluated 220 prompts per model: 20 examples from each of 11 LongBench
tasks (NarrativeQA, Qasper, MultiFieldQA-en, HotpotQA, 2WikiMQA, GovReport,
MultiNews, QMSum, PassageCount, TREC, and TriviaQA).

The four evaluated model aliases were:

- `phi3`: `microsoft/Phi-3-mini-4k-instruct`
- `llama3`: `meta-llama/Meta-Llama-3-8B-Instruct`
- `llama31_8b`: `meta-llama/Llama-3.1-8B-Instruct`
- `llama32_3b`: `meta-llama/Llama-3.2-3B-Instruct`

`C0` through `C5` contain the fixed-method measurements, and `C6` contains
the router evaluations:

| ID | Configuration |
| --- | --- |
| `C0` | FP16 full-KV baseline |
| `C1` | TailorKV-inspired hybrid |
| `C2` | QAQ |
| `C3` | KVQuant simulation |
| `C4` | DynamicKV |
| `C5` | Ada-KV-inspired shared-index eviction |
| `C6` | AdaptiveServe-KV router |

## Contents

| Path | Contents |
| --- | --- |
| `C0/` through `C5/` | Per-prompt fixed-method measurements, aggregate results, and P0 completion markers |
| `C6/` | Router results and per-prompt selections at tau 0.99, 0.95, and 0.90 |
| `dataset/` | Joined routing JSONL, CSV exports, provenance manifest, and portable ZIP |
| `cv_router_220_tau*.json` | Primary 10-fold task-stratified router results |
| `fair_comparison_66.json` | Supplementary same-split fixed-method comparison |
| `figs/` | Primary CV and supplementary split Pareto figures |
| `candidate_pool_sweeps/p0_corrected_2026-07-16/` | Corrected candidate-pool sweep, manifests, and figures |
| `rerun_p0.log` | Complete corrected P0 execution record |

The standalone primary CV files exist only for tau 0.99, 0.95, and 0.90.
The candidate-pool sweep also evaluated tau 0.85 and 0.80 internally, but its
`all_five_consistency` entries correctly record the corresponding standalone
`cv_router_220` files as `not_available`; those files did not exist before the
move and were not created during it.

## Relocation map

Generated files may still mention their original locations. Resolve only the
following corrected-P0 paths through this map:

| Original generation-time location | Current location |
| --- | --- |
| `runs/C0/` through `runs/C6/` | `runs/p0/C0/` through `runs/p0/C6/` |
| `runs/dataset/` | `runs/p0/dataset/` |
| `runs/cv_router_220_tau*.json` | `runs/p0/cv_router_220_tau*.json` |
| `runs/fair_comparison_66.json` | `runs/p0/fair_comparison_66.json` |
| `runs/figs/` | `runs/p0/figs/` |
| `runs/candidate_pool_sweeps/p0_corrected_2026-07-16/` | `runs/p0/candidate_pool_sweeps/p0_corrected_2026-07-16/` |
| `runs/rerun_p0.log` | `runs/p0/rerun_p0.log` |

The paths embedded in `dataset/manifest.json`, the copy of that manifest
inside `dataset/AdaptiveServe-KV-Dataset.zip`, the candidate-pool JSON
manifests, and `rerun_p0.log` record where inputs and outputs existed when the
artifacts were generated. They are intentionally preserved as historical
provenance rather than rewritten after relocation. Their content hashes remain
valid because moving a file does not change its bytes.

## Immutable anchor checksums

These SHA-256 values identify key artifacts before and after relocation:

| Artifact | SHA-256 |
| --- | --- |
| `rerun_p0.log` | `bd12a2e63b7eee351b679a6fd34695987c328718e28f1c6a6f9a346391658735` |
| `dataset/manifest.json` | `0f501afad770e757ca28c46cf117f290af21f2d69db8a30d097012c5b6e38bb9` |
| `dataset/AdaptiveServe-KV-Dataset.zip` | `22ad9af915b9619f28c84fba6efd2580680347582736eeb887a44b09932998ea` |
| `candidate_pool_sweeps/p0_corrected_2026-07-16/summary.json` | `bf72253f7aedd55d0480b47a1d36d9762b61c581b1e6ed9ea004628a27d62caa` |

## Other run namespaces

Do not apply the relocation map to neighboring directories:

- `runs/legacy/` contains superseded pre-P0 and forensic artifacts.
- `runs/longbench16_24k/` is reserved for the planned expanded
  16-task evaluation and is not evidence that evaluation has completed.
