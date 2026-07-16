# AdaptiveServe-KV Routing Dataset

A dataset of 880 (model, prompt, configuration) labelled rows from 220 LongBench
prompts evaluated under six KV-cache configurations on four open-weight LLMs.

## Files

- `adaptiveserve_kv_dataset.csv` — master file, 880 rows (4 models × 220 prompts)
- `per_model/{phi3,llama3,llama31_8b,llama32_3b}.csv` — split by model, 220 rows each
- Source JSONL files are kept under the same directory for code reproducibility

## Columns

### Identifiers
| column | meaning |
| --- | --- |
| `model` | one of `phi3`, `llama3`, `llama31_8b`, `llama32_3b` |
| `task` | LongBench task ID (e.g., `narrativeqa`, `gov_report`, `triviaqa`) |
| `sample_idx` | 0..19 — index within the task's 20-prompt sample |
| `metric` | LongBench scoring metric for this task (`f1`, `rouge1`, `accuracy`, `em`) |

### Surface features (7, prompt-only, microsecond cost)
| column | meaning |
| --- | --- |
| `seq_len_tokens` | tokenizer length of the prompt |
| `seq_len_chars` | character count |
| `token_entropy` | Shannon entropy of token frequencies |
| `gzip_ratio` | gzip(prompt) bytes / raw bytes |
| `unique_token_ratio` | type/token ratio |
| `question_position` | offset of last `?` divided by prompt length |
| `newline_density` | newline count divided by prompt length |

### Per-configuration labels (12 columns total: 6 scores + 6 compression ratios)
For each configuration `c0..c5` (FP16, TailorKV, QAQ, KVQuant, DynamicKV, Ada-KV):

| column | meaning |
| --- | --- |
| `score_{c0..c5}` | LongBench task score under this configuration, in [0, 1] |
| `cr_{c0..c5}`    | KV-cache compression ratio relative to FP16, computed per prompt |

### Oracle / utility columns (computed from the labels)
| column | meaning |
| --- | --- |
| `best_quality_config` | configuration achieving the highest quality on this prompt |
| `best_quality_score` | that configuration's quality score |
| `best_iso_tau_0.99` | configuration achieving highest CR subject to score ≥ 0.99 · score_c0 |
| `best_iso_tau_0.95` | same with τ = 0.95 |
| `best_iso_tau_0.90` | same with τ = 0.90 |

## Source

220 prompts per model are drawn from LongBench (Bai et al., arXiv:2308.14508),
20 prompts from each of 11 tasks: NarrativeQA, Qasper, MultiFieldQA-en, HotpotQA,
2WikiMQA, GovReport, MultiNews, QMSum, PassageCount, TREC, TriviaQA. Each
configuration's output and per-prompt compression were measured on an NVIDIA
RTX 3090 Ti workstation; full reproduction instructions are in the project
README at the repository root.
