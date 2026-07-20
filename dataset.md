# Dataset Protocol: Full Non-Chinese LongBench Evaluation

> **Status:** planned expanded evaluation. This document defines the dataset
> and input protocol for the next experiment. It does **not** describe results
> that have already been produced. The existing corrected P0 artifacts remain
> an 11-task, 220-prompt pilot.

## 1. Final decision

| Item | Decision |
| --- | --- |
| Benchmark | LongBench v1 |
| Benchmark panel | All 16 non-Chinese tasks: 14 English tasks and 2 code tasks |
| Categories | All six LongBench categories |
| Split | Complete official test split of every selected task |
| Benchmark examples per model | 3,750, each identified by `(task, _id)` |
| Sampling | None; no 100-per-task subsample |
| Primary checkpoint | `meta-llama/Llama-3.1-8B-Instruct` |
| Confirmation checkpoint | Not yet locked; see Section 8 |
| Fixed configurations | C0 through C5 |
| Generations per model | 3,750 prompts × 6 configurations = 22,500 |
| Final formatted-input cap | 24,000 tokens |
| Output allowance | Official task-specific limit: 32–512 new tokens |
| Overlength handling | Exact middle truncation in token space |
| Decoding | Greedy, with no stochastic sampling |
| Primary quality summary | Equal-weight mean of the six category scores |

The paper must describe this panel as:

> all 16 non-Chinese LongBench tasks spanning all six task categories

It must not call the panel “all LongBench tasks,” because complete LongBench v1
contains 21 tasks, including five Chinese tasks.

## 2. Why LongBench v1

LongBench v1 is selected for five connected reasons:

1. **The workload matches the system question.** KV-cache methods matter most
   when prompts are long. LongBench was designed specifically to test
   long-context understanding rather than ordinary short-context accuracy.
2. **It covers heterogeneous workloads.** Its six categories include
   document QA, multi-document reasoning, summarization, in-context learning,
   synthetic retrieval/counting, and code completion. This diversity is
   important for evaluating whether one fixed cache policy works universally
   or whether per-prompt routing is useful.
3. **It supports direct comparison with the selected literature.** DynamicKV
   and Ada-KV use the same 16-task panel; KVQuant reports the same 16 task
   scores; TailorKV uses a 13-task subset.
4. **It is reproducible at per-example resolution.** The prompts, reference
   answers, prompt templates, task-specific generation limits, and automatic
   scorers are public. This allows all cache configurations to be evaluated on
   exactly the same prompts.
5. **Preliminary memory checks support feasibility on the available
   hardware.** Single-prompt probes support a common 24,000-token input policy
   on the project’s RTX 3090 Ti. Full-panel feasibility remains conditional on
   the clean-process pilots and implementation gates in Section 12.

LongBench-E is not used as the primary dataset because it is a 13-task,
length-balanced variant rather than the complete 21-task benchmark. LongBench
v2 is not used because the selected cache papers evaluated LongBench v1 and
the newer benchmark includes contexts beyond the scope of the current
single-GPU experiment. Synthetic-only benchmarks may be useful diagnostics,
but they cannot replace the six-category task coverage required here.

This choice does not make LongBench a complete measure of production
generalization. Its automatic metrics, English/code scope, and the 24,000-token
experimental cap must be reported as limitations.

Official benchmark source:
[THUDM/LongBench](https://github.com/THUDM/LongBench/tree/main/LongBench).

## 3. Why the panel contains 16 tasks rather than 6 or 21

The number six refers to **task categories**, not datasets:

> 21 total tasks = 14 English tasks + 5 Chinese tasks + 2 code tasks

The selected panel is therefore:

> 14 English tasks + 2 code tasks = 16 non-Chinese tasks

These 16 tasks still cover all six categories. Excluding the five Chinese
tasks aligns the primary experiment with the main LongBench panels used by the
selected KV-cache literature. It also makes the language scope explicit rather
than mixing multilingual evaluation into an English/code primary study.

The five excluded tasks are:

| Category | Excluded Chinese task |
| --- | --- |
| Single-document QA | MultiFieldQA-zh |
| Multi-document QA | DuReader |
| Summarization | VCSUM |
| Few-shot learning | LSHT |
| Synthetic tasks | PassageRetrieval-zh |

The exclusion limits multilingual claims. A later multilingual experiment
would require all five tasks, appropriate multilingual checkpoint validation,
and a separately declared protocol.

## 4. Selected tasks, official sizes, metrics, and output limits

All counts below are complete official test-split sizes. Metric names refer to
the functions in LongBench’s
[official evaluator](https://github.com/THUDM/LongBench/blob/main/LongBench/eval.py).
Generation limits come from
[`dataset2maxlen.json`](https://github.com/THUDM/LongBench/blob/main/LongBench/config/dataset2maxlen.json).

| Category | Task | LongBench task ID | Test examples | Official scorer | Max new tokens | Llama-3.1 formatted inputs >24K |
| --- | --- | --- | ---: | --- | ---: | ---: |
| Single-document QA | NarrativeQA | `narrativeqa` | 200 | QA F1 | 128 | 115 |
| Single-document QA | Qasper | `qasper` | 200 | QA F1 | 128 | 0 |
| Single-document QA | MultiFieldQA-en | `multifieldqa_en` | 150 | QA F1 | 64 | 0 |
| Multi-document QA | HotpotQA | `hotpotqa` | 200 | QA F1 | 32 | 0 |
| Multi-document QA | 2WikiMultihopQA | `2wikimqa` | 200 | QA F1 | 32 | 0 |
| Multi-document QA | MuSiQue | `musique` | 200 | QA F1 | 32 | 0 |
| Summarization | GovReport | `gov_report` | 200 | ROUGE-L F | 512 | 7 |
| Summarization | QMSum | `qmsum` | 200 | ROUGE-L F | 512 | 22 |
| Summarization | MultiNews | `multi_news` | 200 | ROUGE-L F | 512 | 0 |
| Few-shot learning | TREC | `trec` | 200 | `classification_score` | 64 | 0 |
| Few-shot learning | TriviaQA | `triviaqa` | 200 | QA F1 | 32 | 0 |
| Few-shot learning | SAMSum | `samsum` | 200 | ROUGE-L F | 128 | 0 |
| Synthetic tasks | PassageCount | `passage_count` | 200 | `count_score` | 32 | 3 |
| Synthetic tasks | PassageRetrieval-en | `passage_retrieval_en` | 200 | `retrieval_score` | 32 | 0 |
| Code completion | LCC | `lcc` | 500 | `code_sim_score` | 64 | 1 |
| Code completion | RepoBench-P | `repobench-p` | 500 | `code_sim_score` | 64 | 23 |
| **Total** | **16 tasks** |  | **3,750** |  |  | **171** |

Category totals are:

| Category | Examples |
| --- | ---: |
| Single-document QA | 550 |
| Multi-document QA | 600 |
| Summarization | 600 |
| Few-shot learning | 600 |
| Synthetic tasks | 400 |
| Code completion | 1,000 |
| **Total** | **3,750** |

The exact task counts are also published in the
[LongBench dataset card](https://huggingface.co/datasets/zai-org/LongBench#task-statistics).

## 5. Alignment with the selected KV-cache papers

The selected papers do not all use one identical benchmark protocol. The
comparison below records what is supported by their papers and released code;
an undisclosed sample policy is not treated as a complete-test-set result.

| Technique | LongBench scope | Verified prompts per model/configuration | Relationship to this protocol |
| --- | --- | ---: | --- |
| [DynamicKV](https://aclanthology.org/2025.findings-emnlp.426.pdf) | All 16 non-Chinese tasks | 3,750 | Same task panel and complete test splits |
| [Ada-KV](https://arxiv.org/pdf/2407.11550) | All 16 non-Chinese tasks | 3,750 | Same task panel and complete test splits |
| [KVQuant](https://arxiv.org/pdf/2401.18079) | Scores reported for all 16 tasks | Not disclosed | Same reported tasks; sample-count equivalence must not be claimed |
| [TailorKV](https://aclanthology.org/2025.findings-acl.1043.pdf) | 13 non-Chinese tasks | 3,150 | Omits NarrativeQA, MuSiQue, and QMSum |
| [QAQ](https://arxiv.org/pdf/2403.04643) | No LongBench evaluation | Not applicable | Applying QAQ to this panel is an extension |

This table justifies benchmark comparability; it does not claim that every
local implementation is an exact reproduction. In the current project,
C1 is TailorKV-inspired, C3 is a simplified-but-faithful KVQuant simulation,
C4 uses a shared index set per layer rather than the official head-specific
retention, and C5 uses Ada-KV-inspired budget weighting with a shared index
set. Those implementation differences must remain explicit in the methods
section.

## 6. Difference from the current pilot dataset

| Item | Existing corrected P0 pilot | Planned expanded evaluation |
| --- | ---: | ---: |
| Tasks | 11 | 16 |
| Examples per task | 20 | Complete official split |
| Benchmark examples per model | 220 | 3,750 |
| Categories represented | 5 represented; code completion absent | All 6 |
| Fixed configurations | C0–C5 | C0–C5 |
| Fixed-method generations per model | 1,320 | 22,500 |

The five tasks missing from the current pilot are:

1. MuSiQue
2. SAMSum
3. PassageRetrieval-en
4. LCC
5. RepoBench-P

Existing files under `runs/p0/dataset/` remain evidence for the 220-prompt P0
experiment. They must not be relabelled as, merged silently with, or cited as
results from the expanded 3,750-prompt protocol.

## 7. Sample policy and experimental unit

The final experiment uses every record from each selected official test split:

- no per-task cap;
- no random sampling;
- no evenly spaced sampling;
- no replacement;
- no deduplication beyond the benchmark’s released content; and
- no seed-dependent selection.

The original benchmark `_id`, task ID, and original within-split index must be
stored for every record. Development smoke tests may use a small subset, but
their outputs must be stored separately and must not enter final tables.

For one model:

- **3,750 benchmark examples with unique composite `(task, _id)` keys** are
  evaluated;
- each prompt is generated under **C0, C1, C2, C3, C4, and C5**;
- this produces **22,500 fixed-configuration generations**; and
- the downstream routing table contains **3,750 rows**, one row per prompt,
  with the six configurations represented as label columns.

C6 routing does not add another LLM generation configuration. Its primary
evaluation retains the project’s existing 10-fold task-stratified
cross-validation, with fold shuffling seed 0. Every example is held out once,
and the regressors retain `random_state=0`. The primary operating point is
the existing \(\tau=0.99\) iso-quality policy over C1–C5: a candidate is
eligible when its predicted score is at least 0.99 times the larger of the
predicted C0 score and the training-fold mean actual C0 score. Eligible
candidates are ranked by training-fold harmonic-mean compression; if none is
eligible, select the candidate with the highest predicted quality. Compression
ranking, feature scaling, and model fitting must use only the training fold.
Candidate ties use the fixed C1, C2, C3, C4, C5 order. Leave-one-task-out
evaluation may be reported separately, but it is not a substitute for the
primary protocol. Because each stratified fold contains held-out examples from
the same 16 tasks represented in training, C6 measures within-task held-out
prompt generalization; it must not be presented as unseen-task generalization.

The raw-generation artifact therefore has 22,500 records per model, while the
joined routing artifact has 3,750 rows per model with C0–C5 score,
compression, and byte labels stored as columns.

The primary post-hoc maximum-potential baseline mirrors that operating point
with true held-out labels: among C1–C5 configurations whose actual score is at
least \(0.99\) times the example’s actual C0 score, choose the configuration
with the highest accounted per-example compression. Compression ties use the
fixed C1–C5 order. If no candidate qualifies, choose the highest actual-score
candidate, breaking score ties by compression and then fixed order. Report
this as an oracle upper bound, never as a deployable router. Also report the
quality-first oracle (quality, then compression, then fixed order) as a
diagnostic.

To preserve the existing router feature contract, C6’s `seq_len_tokens`
feature remains the selected tokenizer’s length of the official
task-formatted prompt before chat wrapping and truncation. The separately
logged final-input pre-truncation and post-truncation lengths are audit fields;
they are not silently substituted into the seven-feature router. Any feature
change requires a separately named ablation.

The configuration names used in this document are:

| ID | Project configuration |
| --- | --- |
| C0 | FP16 full KV cache reference |
| C1 | TailorKV-inspired hybrid |
| C2 | QAQ attention-aware variable-bit quantization |
| C3 | Simplified-but-faithful KVQuant simulation |
| C4 | DynamicKV cross-layer adaptive retention, with a shared index set per layer |
| C5 | Head-vote top-k eviction with Ada-KV-inspired budget weighting and a shared index set |
| C6 | AdaptiveServe-KV router over C1–C5; no additional LLM generation |

## 8. Model strategy

### Primary model

The primary checkpoint is:

> `meta-llama/Llama-3.1-8B-Instruct`

It is used directly by TailorKV and Ada-KV, belongs to the only model family
represented across all five selected techniques, supports grouped-query
attention, has a native context window larger than this experiment’s cap, and
is already supported by the repository under the alias `llama31_8b`.

QAQ on this checkpoint is an explicit extension because the QAQ paper evaluated
Llama-2 rather than Llama-3.1.

### Confirmation model

The confirmation checkpoint is deliberately **not yet locked**. The intended
design is to complete and validate the full primary-model pipeline first, then
repeat the unchanged dataset, prompt, context, scoring, and configuration
protocol on one independent checkpoint requested by, or agreed with, the
journal.

`mistralai/Mistral-7B-Instruct-v0.2` is the leading candidate because it was
used by DynamicKV and Ada-KV and the Mistral family was evaluated by KVQuant.
It must not be described as the final confirmation model until that choice is
formally made and repository support is implemented.

## 9. Prompt construction, context length, and generation

### Prompt construction

1. Format each example with LongBench’s official task template from
   [`dataset2prompt.json`](https://github.com/THUDM/LongBench/blob/main/LongBench/config/dataset2prompt.json).
2. Follow LongBench’s wrapper-exclusion set: TREC, TriviaQA, SAMSum, LCC, and
   RepoBench-P receive no additional chat wrapper.
3. For the remaining tasks, integrate Llama-3.1 through its
   tokenizer-provided one-user-turn chat template and request the generation
   prompt. This is a checkpoint-specific adaptation of the
   [released LongBench runner](https://github.com/THUDM/LongBench/blob/main/LongBench/pred.py),
   which predates Llama-3.1. Unlike that runner, this protocol applies
   truncation after final formatting. Pass the locked template date string
   `20 Jul 2026` so the rendered system header is reproducible.
4. Tokenize the complete, final model input before applying the length rule.
   Retain the tokenizer's normal special tokens (including Llama's BOS token)
   on the five tasks without a chat wrapper.
5. Save both the pre-truncation and post-truncation token counts.

### Context policy

The uniform cap is:

> **24,000 tokens for the final formatted input**

If the final token sequence has length \(N > 24{,}000\), retain:

> the first 12,000 tokens + the last 12,000 tokens

The implementation must assert that the resulting input contains at most
24,000 tokens. It must not decode and re-tokenize the retained pieces, because
that can change the exact length. Every configuration for a given model must
receive the same retained token sequence for an example. Each model applies
the same 24,000-token algorithm with its own tokenizer; any cross-model
difference in retained text must be logged rather than hidden.

Generation then uses the official task-specific `max_new_tokens` value from
Section 4. The maximum possible request is therefore:

> 24,000 input tokens + 512 generated tokens = 24,512 total tokens

Decoding is greedy (`do_sample=False` or exact argmax), with no beam search or
stochastic sampling. This does not by itself assert deterministic CUDA-kernel
execution.
Use the pinned checkpoint's complete native terminal-token set:
`128001` (`<|end_of_text|>`), `128008` (`<|eom_id|>`), and `128009`
(`<|eot_id|>`). For SAMSum, require at least one generated token and add the
tokenizer’s newline token as a stopping ID, matching the released runner.

### Why 24,000 tokens

A full-panel audit with the Llama-3.1-8B-Instruct tokenizer found:

| Statistic | Result |
| --- | ---: |
| Median final formatted length | 8,867.0 tokens |
| 90th percentile | 17,505.9 tokens |
| 95th percentile | 23,205.1 tokens |
| 99th percentile | 46,741.5 tokens |
| Maximum | 65,404 tokens |
| Inputs at or below 24,000 | 3,579 / 3,750 (95.44%) |
| Inputs requiring truncation | 171 / 3,750 (4.56%) |
| Aggregate input tokens retained | 93.88% |

These counts are specific to the selected Llama-3.1 tokenizer, chat
adaptation, and final-formatting policy. They must be recomputed for a
confirmation model.

At 31,500 tokens, 95 examples would require truncation instead of 171. The
larger cap would therefore leave 76 fewer examples truncated and would retain
more content within the remaining 95 truncated examples.

Preliminary single-prompt hardware probes found that the limiting C3 path
reached 24,530 MiB of reserved memory at 31,500 tokens on a 24,564 MiB RTX
3090 Ti, leaving no reliable physical-VRAM safety margin. At 24,000 tokens, it
reserved 22,348 MiB, leaving approximately 1.9 GiB after driver overhead. C0,
C2, C4, and C5 preliminary probes also passed at 24,000 tokens; C1 passed even
at 31,500 after suppressing unnecessary full-sequence logits. These probes
support the cap decision but are not substitutes for the saved, clean-process
C0–C5 pilots required before the full run.

The 24,000-token value is therefore a hardware-feasible common experimental
budget, not the native limit of Llama-3.1-8B-Instruct. Different caps must not
be used for different cache methods. CPU offload or WDDM memory paging cannot
be counted as a successful in-VRAM timing run.

## 10. Scoring and aggregation

Each generated answer is scored with LongBench’s official task scorer:

- `qa_f1_score` for the six QA datasets plus TriviaQA;
- `rouge_score` for GovReport, QMSum, MultiNews, and SAMSum;
- `classification_score` for TREC;
- `count_score` for PassageCount;
- `retrieval_score` for PassageRetrieval-en; and
- `code_sim_score` for LCC and RepoBench-P.

For examples with multiple references, use the maximum score over references,
as in the official evaluator. Apply official first-line post-processing to
TREC, TriviaQA, and SAMSum. Store per-example scores on the \([0,1]\) scale;
multiply by 100 only for clearly labelled publication tables.

Unequal full-test-set sizes make aggregation order important. The primary
quality summary is computed as follows:

1. average example scores within each task;
2. average task scores within each of the six categories; and
3. average the six category scores with equal category weight.

This is the category-balanced presentation used by the official LongBench
leaderboard and prevents the two 500-example code tasks from dominating the
headline score. Every paper table must also report all 16 task scores and the
six category scores. A prompt-level micro-average may be retained as a
diagnostic, but it must not be labelled “LongBench Avg.”

All configurations must be scored and aggregated from the same prompt IDs in
the same order.

For compression, retain the project’s existing definition:

> per-example compression ratio =
> `kv_bytes_fp16 / kv_bytes`

Here `kv_bytes` is an effective-storage accounting field, not a uniform
measurement of live CUDA allocation. C0, C4, and C5 use physical retained
FP16 tensor bytes. C1, C2, and C3 are round-trip FP16 simulations whose
`kv_bytes` values model packed quantized storage, including the implemented
quantization metadata; their runtime tensors are not physically stored in
that packed representation. Accordingly, the reported ratio estimates KV
storage reduction under the specified representation. It must not be
described as measured peak-VRAM reduction, end-to-end memory reduction, or
latency improvement. Every raw and joined record carries the applicable
accounting descriptor.

The primary compression summary is the harmonic mean of the 3,750 per-example
ratios. Because this request-weighted statistic gives more weight to tasks
with larger official test splits, also report a category-balanced diagnostic:
compute the harmonic mean within each task, average task values within each
category, and then average the six category values equally. Apply the same two
compression summaries to C0–C6 and label them unambiguously.

## 11. Reproducibility and provenance

All expanded-evaluation outputs belong under the model-neutral
`runs/longbench16_24k/` namespace. Each execution must use
`<model-alias>/<run-id>/` beneath that root: the primary model uses
`llama31_8b/<run-id>/`, while the eventual confirmation checkpoint uses its
own model alias and run ID. A run ID identifies one immutable execution; retries
that change generated records must use a new run ID.

The expanded run must produce a machine-readable manifest containing:

- benchmark name, dataset ID, split, and resolved dataset revision;
- the 16 task IDs and exact row counts;
- ordered `_id` values and a SHA-256 hash for each task source file;
- model and tokenizer IDs plus resolved revisions;
- hashes of the prompt-template and output-limit configurations;
- the 24,000-token policy and per-task truncation counts;
- all task-specific generation and post-processing settings;
- the C6 feature definition, 10-fold task-stratified split procedure, fold seed
  0, and regressor `random_state=0`;
- C0–C5 implementation identifiers and hyperparameters;
- source-code commit;
- Python, PyTorch, Transformers, CUDA, and driver versions;
- GPU model and physical VRAM;
- run start/end timestamps and completion status; and
- hashes of every per-prompt result file.

Each per-prompt record must include at least:

- model;
- configuration;
- task;
- benchmark `_id`;
- original within-split index;
- metric;
- reference answers;
- prediction;
- score;
- pre-truncation token count;
- post-truncation token count;
- truncation flag;
- a hash of the final input token-ID sequence;
- task-specific output limit;
- generated-token count; and
- effective and FP16-reference KV byte fields plus the configuration-specific
  physical-versus-modeled accounting descriptor.

The manifest must make interrupted, retried, or failed prompts visible. Raw
records and the joined table must use `(task, benchmark_id)` as the primary
key, with the source index retained as an audit field. A final run is complete
only when all six configurations contain the same 3,750 composite IDs exactly
once and their final-input token hashes agree within each model.

## 12. Implementation gates before the full run

The codebase formerly implemented only the old 11-task/20-example protocol.
Checked items below are now implemented and CPU-tested; unchecked items require
the canonical prepared artifact, saved GPU evidence, or complete generation
records. The full evaluation must not begin until every pre-generation item is
checked:

- [x] Add MuSiQue, SAMSum, PassageRetrieval-en, LCC, and RepoBench-P to the
      task registry and prompt registry.
- [x] Add the official retrieval and code-similarity scorers.
- [x] Add SAMSum to official first-line post-processing and termination logic.
- [x] Remove the 20-example loader default for the final-run mode and make an
      accidentally set `ADAPTIVESERVE_LB_N` fatal.
- [x] Validate the exact 16-task membership and all per-task counts before any
      generation begins; missing files or mismatched counts must fail closed
      rather than be skipped.
- [x] Load all 3,750 official test examples and verify the counts in Section 4.
- [x] Apply the 24,000-token limit to the final formatted input, identically
      across C0–C5.
- [x] Extend per-prompt logging with benchmark ID, source index, references,
      pre/post token counts, truncation state, output counts, and final token-ID
      hash; join configurations on `(task, benchmark_id)`.
- [x] Add the six-category map and implement the task → category → headline
      quality aggregation consistently in C0–C5, C6, cross-validation,
      candidate-pool summaries, plots, and exports.
- [x] Preserve the primary harmonic-mean compression summary and add the
      labelled category-balanced compression diagnostic from Section 10.
- [x] Request only final-position logits in non-perplexity manual prefills;
      full-sequence vocabulary logits are unnecessary and exceed the memory
      budget.
- [x] Explicitly exclude speed and perplexity from this quality/KV-label
      milestone; they require no runs here and may use a separately named
      protocol later.
- [ ] Run a worst-case 24,000-input + 512-output pilot for C0–C5.
- [x] Force all 512 feasibility decode positions even if a terminal token is
      produced, require at least 1.5 GiB of physical-VRAM headroom, and block
      final-mode generation until all six pilots pass.
- [ ] Save the feasibility-pilot script, configuration, raw output, and
      environment metadata as a versioned provenance artifact.
- [ ] Verify that every configuration produces the same ordered composite-ID
      set and the same final-input token hash within a model.
- [ ] Generate and validate the provenance manifest before aggregating results.

## 13. Draft paper description — use only after the run is complete

The paragraph below is a template. It must not be inserted into the paper as a
completed-method statement until every gate in Section 12 has passed and the
3,750-example result set has been validated.

> We evaluate AdaptiveServe-KV on LongBench v1 because it directly targets
> long-context understanding, is used by the selected KV-cache literature, and
> provides public prompts, task-specific generation limits, and automatic
> metrics across heterogeneous workloads. Our primary panel contains all 16
> non-Chinese LongBench tasks—14 English tasks and two code-completion
> tasks—spanning single-document QA, multi-document QA, summarization,
> few-shot learning, synthetic tasks, and code completion. We use every example
> from the official test splits, yielding 3,750 benchmark examples with unique
> `(task, _id)` keys. Each example is evaluated under the FP16 baseline and
> five cache configurations, for 22,500 generations per model. We cap the final
> formatted input at 24,000 tokens and apply exact middle truncation only when
> necessary; 95.44% of the primary-model inputs remain untruncated. This cap
> was selected after clean-process feasibility pilots confirmed a safe in-VRAM
> margin on our 24 GB RTX 3090 Ti, whereas the tested 31,500-token alternative
> did not. We report official per-task scores and aggregate them through
> equal-weight category means.

## 14. Decisions intentionally left open

Only one material design choice remains open:

- the exact confirmation-model checkpoint. Mistral-7B-Instruct-v0.2 is the
  current candidate, but the final choice will be made after the primary
  pipeline succeeds and journal guidance is known.

The dataset panel, complete-test-set policy, primary checkpoint, 24,000-token
formatted-input cap, middle-truncation rule, task-specific output limits, and
six-configuration generation matrix are locked by this protocol.
