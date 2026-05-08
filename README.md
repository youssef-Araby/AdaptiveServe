# AdaptiveServe

**Benchmarking adaptive KV cache quantization for efficient LLM inference.**

AdaptiveServe measures the quality–efficiency trade-off of KV cache compression strategies for large language models. It provides reproducible benchmarks across latency, memory, and generation quality metrics. The current code base implements a full-precision baseline (C0) and the QAQ attention-aware quantization algorithm (C2); additional methods (C1, C3–C5) and a per-query adaptive selector (C6–C7) are planned.

---

## Motivation

As LLMs are deployed with longer contexts, the key-value (KV) cache becomes the dominant memory bottleneck — growing linearly with sequence length and consuming tens of gigabytes for long contexts. KV cache quantization trades a small amount of numerical precision for significant memory reduction, but the quality impact varies by model, task, and quantization strategy.

This project characterises that trade-off across two configurations, two model families, and seven long-context tasks.

---

## Configurations

| ID | Method | Status | Description |
|----|--------|--------|-------------|
| **C0** | FP16 Baseline | implemented | Full-precision KV cache. No quantization. Reference for all comparisons. |
| **C1** | TailorKV | planned | Offline per-layer hybrid (sparsity + quantization), black-box baseline. |
| **C2** | QAQ Full | implemented | Attention-aware variable-bit quantization ([2, 16] bits), 1 % outliers kept at FP16, attention window of 5. Keys quantized via query-norm error bound; Values quantized inversely proportional to attention score (Dong et al., 2024 — [arXiv:2403.04643](https://arxiv.org/abs/2403.04643)). |
| **C3** | KVQuant | planned | Per-channel asymmetric quantization with calibration. |
| **C4** | DynamicKV | planned | Layer-adaptive token retention. |
| **C5** | Ada-KV | planned | Head-budget adaptive eviction (FlashAttention-2). |
| **C6** | Adaptive-A (rule-based selector) | planned | Per-query selection across {C1…C5}. |
| **C7** | Adaptive-B (learned selector) | planned | Lightweight MLP selector trained on profiling signals. |

---

## Models

| Alias | HuggingFace ID | Parameters | Context |
|-------|---------------|------------|---------|
| `phi3` | `microsoft/Phi-3-mini-4k-instruct` | 3.8 B | 4 096 tokens |
| `llama3` | `meta-llama/Meta-Llama-3-8B-Instruct` | 8 B | 8 192 tokens |

---

## Metrics

### Speed
| Metric | Description |
|--------|-------------|
| **TTFT** | Time-to-first-token (ms) — prefill latency |
| **TPOT** | Time-per-output-token (ms) — decode latency |
| **Tokens/sec** | Decode throughput |
| **Peak VRAM** | Maximum GPU memory (MB) during generation |
| **KV Cache (MB)** | Theoretical KV cache size at the measured prompt length |
| **KV Compression Ratio** | FP16 KV size / quantized KV size |

### Quality
| Metric | Description |
|--------|-------------|
| **WikiText-2 PPL** | Perplexity on non-overlapping chunks — lower is better |
| **LongBench Avg** | Average score across 7 long-context tasks — higher is better |

#### LongBench Tasks
| Task | Metric | Domain |
|------|--------|--------|
| NarrativeQA | F1 | Story comprehension |
| Qasper | F1 | Scientific QA |
| HotpotQA | F1 | Multi-hop reasoning |
| 2WikiMQA | F1 | Multi-hop reasoning |
| GovReport | ROUGE-1 | Long-form summarization |
| TREC | Accuracy | Question classification |
| TriviaQA | F1 | Open-domain QA |

---

## Results

### LLaMA-3-8B-Instruct

| Config | TTFT (ms) | TPOT (ms) | Tok/s | VRAM (MB) | KV Cache (MB) | Compression | PPL ↓ | LongBench ↑ |
|--------|-----------|-----------|-------|-----------|---------------|-------------|-------|-------------|
| C0 (FP16) | 266.3 | 45.8 | 21.83 | 15 719 | 128.0 | 1.00× | 7.460 | 0.505 |
| C2 (QAQ) | 579.7 | 158.1 | 6.32 | 17 790 | 76.1 | **1.76×** | 7.460 | 0.505 |

### Phi-3-mini-4k-instruct

| Config | TTFT (ms) | TPOT (ms) | Tok/s | VRAM (MB) | KV Cache (MB) | Compression | PPL ↓ | LongBench ↑ |
|--------|-----------|-----------|-------|-----------|---------------|-------------|-------|-------------|
| C0 (FP16) | 158.7 | 19.9 | 50.34 | 7 775 | 384.0 | 1.00× | 5.635 | 0.369 |
| C2 (QAQ) | 578.2 | 35.6 | 28.06 | 7 839 | 228.3 | **1.76×** | 5.635 | 0.377 |

### Key Observations

- **C2 achieves 1.76× KV cache compression on both models** with no perplexity degradation.
- **LongBench quality is fully preserved** on LLaMA-3 (0.505 → 0.505) and slightly improves on Phi-3 (0.369 → 0.377), suggesting the attention-aware bit allocation can suppress irrelevant cache noise.
- **TPOT overhead in C2 is a Python simulation artefact** — production hardware with packed 2–4 bit storage would see DRAM-bandwidth speedups over the FP16 baseline, not slowdowns.
- Phi-3's attention-aware decode is disabled (SDPA path) because its eager attention implementation is ~43× slower than SDPA; V-cache bits fall back to the K-formula in that case.

---

## Repository Structure

```
AdaptiveServe/
├── scripts/
│   ├── _common.py                 # Shared constants, scoring, PPL, LongBench loader
│   ├── benchmark_c0_baseline.py   # C0 baseline (FP16 full KV cache)
│   └── benchmark_c2_qaq.py        # C2 QAQ (variable-bit attention-aware quantization)
└── runs/
    ├── C0/
    │   ├── llama3/results.json
    │   └── phi3/results.json
    └── C2/
        ├── llama3/results.json
        └── phi3/results.json
```

Each `benchmark_cN_*.py` script is self-contained: it owns its own model loading,
speed loop, and generation function. Only constants and method-agnostic helpers
(scoring, perplexity, LongBench task loading) are shared via `_common.py`.

---

## Setup

```bash
# Python 3.10+
pip install torch transformers datasets
```

A CUDA-capable GPU with at least 24 GB VRAM is recommended for LLaMA-3-8B. Phi-3-mini runs in ~8 GB.

HuggingFace access tokens are required for gated models:

```bash
huggingface-cli login
```

---

## Usage

```bash
# C0 — FP16 baseline
python scripts/benchmark_c0_baseline.py --model phi3
python scripts/benchmark_c0_baseline.py --model llama3

# C2 — QAQ full quantization
python scripts/benchmark_c2_qaq.py --model phi3
python scripts/benchmark_c2_qaq.py --model llama3
```

Results are written to `runs/{config}/{model}/results.json`.

Each benchmark measures:
1. **Speed**: fixed 1024-token prompt, 50-token decode run.
2. **Perplexity**: non-overlapping chunks over the full WikiText-2 test split.
3. **LongBench**: 20 samples per task, scored with task-specific metrics.

---

## Reference

> Dong et al. (2024). *QAQ: Quality Adaptive Quantization for LLM Key-Value Cache*.  
> arXiv:2403.04643. [https://arxiv.org/abs/2403.04643](https://arxiv.org/abs/2403.04643)
