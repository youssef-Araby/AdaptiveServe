# AdaptiveServe-KV

## Completed P0 Evidence

The authoritative completed KV-cache experiment is the **corrected P0 rerun**,
finished on 2026-07-16. It evaluates C0-C5 on 220 LongBench prompts per model,
then evaluates C6 with 10-fold task-stratified cross-validation. Start with
[results.md](results.md) and the [P0 archive guide](runs/p0/README.md).

| Artifact | Purpose |
| --- | --- |
| [runs/p0/rerun_p0.log](runs/p0/rerun_p0.log) | Complete corrected P0 execution record |
| [runs/p0/cv_router_220_tau0.99.json](runs/p0/cv_router_220_tau0.99.json) | Primary all-five router CV at $\tau=0.99$ |
| [runs/p0/figs](runs/p0/figs) | Primary `pareto_cv_*` figures and supplementary split figures |
| [runs/p0/candidate_pool_sweeps/p0_corrected_2026-07-16](runs/p0/candidate_pool_sweeps/p0_corrected_2026-07-16) | Corrected pool/tau sweep, figures, hashes, and consistency checks |

The P0 GPU phase used `--skip-speed-ppl`. Latency, throughput, VRAM, and
perplexity fields preserved in fixed-method `results.json` files are not fresh
P0 measurements. C6's full single-request routing overhead is measured in
milliseconds because it includes feature extraction and tokenization.

## Planned Expanded Evaluation

[dataset.md](dataset.md) defines the planned next evaluation: all 16
non-Chinese LongBench tasks, all 3,750 official test examples,
Llama-3.1-8B-Instruct as the primary checkpoint, and a 24,000-token
formatted-input cap. This is a **protocol, not completed evidence**.

Future outputs belong under
[`runs/longbench16_24k/`](runs/longbench16_24k/), separated first by model alias
and then by run ID. They must not overwrite or reuse the P0 completion markers
under `runs/p0/`.

## Corrected Candidate-Pool Sweep

C6 can route over any subset of C1-C5. The P0 sweep evaluates all 26 pools of
size 2-5 at $\tau \in \{0.99, 0.95, 0.90, 0.85, 0.80\}$ using the same 10-fold
task-stratified protocol as the primary router results.

| Model | Highest-quality evaluated pool | $\tau$ | Quality / compression |
| --- | --- | ---: | ---: |
| Phi-3-mini | `{C2,C3}` | 0.99 | 0.3184 / 3.659x |
| LLaMA-3-8B | `{C2,C5}` | 0.85 | 0.4337 / 4.936x |
| LLaMA-3.2-3B | `{C2,C4,C5}` | 0.99 | 0.4069 / 4.607x |
| LLaMA-3.1-8B | `{C2,C3}` | 0.99 | 0.4374 / 3.842x |

Each point strictly dominates its P0 C0 baseline on this CV aggregate. These are
calibration results, not nested-CV estimates after selecting among all 130
operating points.

## Run Policy

`runs/p0/` is a read-only evidence archive. Active scripts must not write into
it or replace its completion markers. New evaluations belong under
`runs/longbench16_24k/<model-alias>/<run-id>/`, with a separate manifest for
each immutable run.

See [runs/README.md](runs/README.md) for the separation between completed P0
evidence, the planned expanded evaluation, and superseded legacy artifacts.
