# AdaptiveServe-KV: Oracle Potential and Current C6 Router

This report first establishes how each fixed compression technique performs
alone, then presents the finalized post-hoc oracle potential and the current
held-out C6 router result. The earlier 220-prompt P0 results are intentionally
excluded.

The evidence comes from the completed Llama-3.1-8B-Instruct primary run over
all 3,750 official examples from the 16 non-Chinese LongBench tasks, covering
all six task categories. Each fixed configuration, C0 through C5, completed
3,750 generations, producing 22,500 generation records with no failed attempts
or retried keys. The joined analysis and C6 evaluation each contain 3,750
examples.

| Run field | Final value |
| --- | --- |
| Model | `meta-llama/Llama-3.1-8B-Instruct` |
| Model revision | `0e9e39f249a16976918f6564b8830bc894c89659` |
| Source commit used for the run | `17e27ba7bd53d00bb933b22d8c5f57485a75a2ad` |
| Run ID | `primary-20260720-17e27ba-c3fix-expseg` |
| Final status | Complete; all finalization checks passed |
| Final manifest SHA-256 | `3d84c92931c9179063f53a89308a67e145730249f9bb2d1c7cbcfad0726c1d37` |

## Fixed Compression Techniques in Isolation

Each row below applies one configuration to all 3,750 examples. These are
standalone results, not router selections. Quality is the equal-weight mean
across the six LongBench categories, and compression is the harmonic mean of
the per-example effective KV-storage ratios.

| Configuration | Technique | What it does | Quality | Change from C0 | Harmonic KV ratio |
| --- | --- | --- | ---: | ---: | ---: |
| C0 | Full FP16 KV cache | Keeps the complete native FP16 cache; uncompressed reference | 0.5015 | — | 1.0000x |
| C1 | TailorKV-inspired hybrid | Combines aggressive token retention with low-bit quantization | 0.4426 | -0.0589 | 18.2930x |
| C2 | QAQ | Assigns attention-aware variable-bit precision and preserves outliers | 0.4984 | -0.0031 | 4.6157x |
| C3 | KVQuant simulation | Applies 4-bit groupwise K/V quantization while preserving outliers | 0.5023 | +0.0008 | 3.2449x |
| C4 | DynamicKV | Adapts the retained-token budget across layers using a shared index set per layer | 0.4897 | -0.0118 | 4.8725x |
| C5 | Ada-KV-inspired eviction | Uses attention-head voting and budget weighting to retain a shared top-k token set | 0.4922 | -0.0093 | 4.8735x |

C1 supplies the most aggressive standalone compression, but with the largest
quality loss. C2 provides 4.6157x compression while remaining only 0.0031 below
C0 in category-balanced quality. C3 has the highest standalone quality,
although its +0.0008 difference from C0 is descriptive and has not been
established as statistically significant. C4 and C5 provide similar
compression, with C5 retaining slightly more aggregate quality in this run.

The accounting basis is important: C1-C3 report modeled packed bytes, including
quantization metadata, whereas C4-C5 report physically retained FP16 tensor
bytes. The implementation and measurement boundaries are detailed later in
this report.

## Router Potential and Current State

Quality is reported primarily as the equal-weight mean across the six LongBench
categories. The prompt-level micro mean is included as a diagnostic. Compression
is the harmonic mean of the per-example effective KV-storage ratios; this is the
primary compression aggregation.

| Evaluation | Information used for selection | Category-balanced quality | Quality change from C0 | Prompt-micro quality | Harmonic KV ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| C0 reference | Full FP16 KV cache | 0.5015 | — | 0.5093 | 1.0000x |
| Primary oracle | Actual per-example quality and compression | 0.5228 | +0.0213 | 0.5338 | 9.4226x |
| Quality-first oracle | Actual per-example quality and compression | 0.5350 | +0.0335 | 0.5477 | 7.9405x |
| Current C6 router | Prompt-only features under held-out cross-validation | 0.4921 | -0.0094 | 0.4986 | 4.8913x |

C0 is an uncompressed reference, not a router candidate. Both oracle rows are
post-hoc upper bounds that use information unavailable at inference time. C6 is
the only learned-router result in this report.

## Primary Oracle: Iso-Quality Maximum Compression

The primary oracle is
`iso_quality_tau_0.99_max_compression`. For each example, it:

1. treats C1 through C5 as the candidate pool;
2. keeps candidates whose actual quality is at least 99% of the actual C0
   quality;
3. selects the eligible candidate with the highest actual per-example
   compression; and
4. if no candidate is eligible, selects the candidate with the highest actual
   quality, then compression.

Ties use the fixed order C1, C2, C3, C4, C5.

| Primary-oracle outcome | Count | Share |
| --- | ---: | ---: |
| At least one eligible candidate | 3,580 | 95.47% |
| No eligible candidate; fallback used | 170 | 4.53% |
| Threshold violations among non-fallback selections | 0 | 0.00% |

The selected configurations were:

| Configuration | Selections | Share |
| --- | ---: | ---: |
| C1 | 2,379 | 63.44% |
| C2 | 535 | 14.27% |
| C3 | 318 | 8.48% |
| C4 | 402 | 10.72% |
| C5 | 116 | 3.09% |

The primary oracle reaches 9.4226x harmonic compression while increasing
category-balanced quality by 0.0213 and prompt-micro quality by 0.0244 relative
to C0. This shows that the evaluated C1-C5 candidate set contains substantial
per-example routing potential. It does not show that a deployable router can
identify those choices without access to the realized outputs and labels.

When an example's actual C0 score is zero, the 99% threshold is also zero, so
all non-negative candidate scores are eligible. The 170 fallback examples are
the cases in which no candidate met the actual per-example threshold.

## Quality-First Oracle Diagnostic

The required quality-first diagnostic is
`max_quality_then_compression_then_config`. It selects the candidate with the
highest actual per-example quality, then uses actual compression and the fixed
configuration order to break ties.

| Configuration | Selections | Share |
| --- | ---: | ---: |
| C1 | 2,088 | 55.68% |
| C2 | 617 | 16.45% |
| C3 | 433 | 11.55% |
| C4 | 444 | 11.84% |
| C5 | 168 | 4.48% |

This diagnostic reaches 0.5350 category-balanced quality and 7.9405x harmonic
compression. Relative to the primary oracle, prioritizing realized quality
adds 0.0122 quality but gives up 1.4821x harmonic compression. It is included
to expose the quality-compression trade-off, not as the primary objective.

## Current C6 Held-Out Router

C6 was evaluated with 10-fold task-stratified cross-validation using seed 0.
Each fold contains 3,375 training examples and 375 test examples, and every
example is held out exactly once. This is held-out-prompt evaluation within
each task; it is not an unseen-task or external-dataset test.

The router uses only seven pre-generation prompt features:
`seq_len_tokens`, `seq_len_chars`, `token_entropy`, `gzip_ratio`,
`unique_token_ratio`, `question_position`, and `newline_density`. A fresh
standardization transform and one quality regressor per configuration are fit
inside each training fold.

For each held-out example, C6 admits a candidate when its predicted quality is
at least 99% of the larger of predicted C0 quality and the training-fold mean
actual C0 quality. Eligible candidates are ranked by training-fold harmonic
compression; if none is eligible, C6 selects the candidate with the highest
predicted quality.

| Configuration | C6 selections | Share |
| --- | ---: | ---: |
| C1 | 735 | 19.60% |
| C2 | 780 | 20.80% |
| C3 | 811 | 21.63% |
| C4 | 479 | 12.77% |
| C5 | 945 | 25.20% |

C6 achieves 4.8913x harmonic compression. Its category-balanced quality is
0.4921, which is 0.0094 below C0; its prompt-micro quality is 0.4986, which is
0.0107 below C0.

| Actual held-out quality check | Count | Rate |
| --- | ---: | ---: |
| C6 quality below actual C0 quality | 779 | 20.77% |
| C6 quality below 99% of actual C0 quality | 755 | 20.13% |

Mean held-out quality-prediction R-squared ranges from 0.298 to 0.310 across
C0-C5. The observed constraint-violation rate and modest predictive fit show
that the current prompt-only regressors do not yet enforce the intended
per-example iso-quality target reliably.

## Oracle-to-C6 Gap

| Comparison | Primary oracle | Current C6 | Gap |
| --- | ---: | ---: | ---: |
| Category-balanced quality | 0.5228 | 0.4921 | 0.0306 |
| Harmonic KV ratio | 9.4226x | 4.8913x | 4.5314x |

C6's harmonic ratio is 51.91% of the primary oracle's raw ratio. Measured as
compression improvement above the 1.0x C0 reference, C6 realizes 46.20% of the
oracle improvement. At the same time, C6 falls below C0 quality, whereas the
oracle remains above it.

Therefore, these results establish a meaningful routing opportunity but do not
yet establish a successful quality-preserving learned router. The gap suggests
that quality prediction and constraint reliability are the immediate
bottlenecks for C6, rather than an absence of useful per-example variation
among C1-C5.

## Interpretation Boundaries

- The oracles use actual output quality and actual compression after generation.
  They are empirical upper bounds over C1-C5, not deployable policies or
  theoretical global optima.
- Selecting configurations using realized quality introduces post-hoc selection
  advantage. Oracle quality above C0 must not be interpreted as an unbiased
  improvement from a production router.
- The primary quality statistic is the equal-weight mean across six task
  categories. Prompt-micro quality is diagnostic and gives more weight to tasks
  with more examples.
- The primary compression statistic is the harmonic mean across examples.
  Category-balanced compression values in the artifacts are diagnostic and are
  not used for the headline claims.
- C0, C4, and C5 use retained physical FP16 tensor bytes. C1, C2, and C3 use
  modeled packed bytes including quantization metadata, while their runtime
  execution uses round-trip FP16 quantization simulation.
- The reported ratios describe effective KV storage. They are not measurements
  of peak VRAM, latency, throughput, energy, or perplexity.
- C1-C5 are paper-inspired experimental implementations, not claims of exact
  reproduction of the original systems.
- These results cover the primary Llama-3.1-8B-Instruct model only. A second
  confirmation model and statistical uncertainty analysis remain future work.

## Final Artifacts

- [Final run manifest](runs/longbench16_24k/llama31_8b/primary-20260720-17e27ba-c3fix-expseg/manifest.json)
- [Joined analysis and oracle summary](runs/longbench16_24k/llama31_8b/primary-20260720-17e27ba-c3fix-expseg/analysis/summary.json)
- [Joined per-example records](runs/longbench16_24k/llama31_8b/primary-20260720-17e27ba-c3fix-expseg/analysis/joined.jsonl)
- [C6 summary](runs/longbench16_24k/llama31_8b/primary-20260720-17e27ba-c3fix-expseg/analysis/c6_summary.json)
- [C6 held-out per-example records](runs/longbench16_24k/llama31_8b/primary-20260720-17e27ba-c3fix-expseg/analysis/c6_per_prompt.jsonl)
