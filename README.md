# AdaptiveServe

**Benchmarking adaptive KV cache quantization for efficient LLM inference.**

AdaptiveServe measures the quality–efficiency trade-off of KV cache compression strategies for large language models. It provides reproducible benchmarks across latency, memory, and generation quality metrics. The current code base implements a full-precision baseline (C0), TailorKV hybrid quantization+sparsity (C1), QAQ attention-aware quantization (C2), KVQuant per-channel/per-token 4-bit quantization with outliers (C3), and DynamicKV per-layer token retention (C4); an additional method (C5) and a per-query adaptive selector (C6–C7) are planned.

---

## Motivation

As LLMs are deployed with longer contexts, the key-value (KV) cache becomes the dominant memory bottleneck — growing linearly with sequence length and consuming tens of gigabytes for long contexts. KV cache quantization trades a small amount of numerical precision for significant memory reduction, but the quality impact varies by model, task, and quantization strategy.

This project characterises that trade-off across two configurations, two model families, and seven long-context tasks.

---

## Configurations

| ID | Method | Status | Description |
|----|--------|--------|-------------|
| **C0** | FP16 Baseline | implemented | Full-precision KV cache. No quantization. Reference for all comparisons. |
| **C1** | TailorKV | implemented | Hybrid per-layer compression: dense "quantization-friendly" layers (Q={0}) get 1-bit KIVI-style quantization (per-channel K, per-token V); sparse "sparsity-friendly" layers get SnapKV-style retention (64 recent + 128 top-attention prefix tokens). Yao et al. 2025 — [arXiv:2505.19586](https://arxiv.org/abs/2505.19586). |
| **C2** | QAQ Full | implemented | Attention-aware variable-bit quantization ([2, 16] bits), 1 % outliers kept at FP16, attention window of 5. Keys quantized via query-norm error bound; Values quantized inversely proportional to attention score (Dong et al., 2024 — [arXiv:2403.04643](https://arxiv.org/abs/2403.04643)). |
| **C3** | KVQuant | implemented | Per-channel K + per-token V uniform asymmetric quantization at 4 bits, with 1 % FP16 magnitude outliers (Dense-and-Sparse). All layers, full prefix length. Single-pass SDPA prefill + post-hoc cache quantization. Hooper et al., 2024 — [arXiv:2401.18079](https://arxiv.org/abs/2401.18079). |
| **C4** | DynamicKV | implemented | Per-layer attention-driven token retention. Each layer keeps the top-K tokens by aggregated attention score (with sliding window of recent tokens always preserved); per-layer budget is uniform (Zhou et al., 2024 — [arXiv:2407.11550](https://arxiv.org/abs/2407.11550)). |
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

Speed phase uses a fixed prompt of 3 500 tokens (Phi-3) or 7 500 tokens (LLaMA-3) — both near the model context limit so reported compression reflects realistic long-context use.

### LLaMA-3-8B-Instruct  (speed prompt = 7 500 tokens)

| Config | TTFT (ms) | TPOT (ms) | Tok/s | VRAM (MB) | KV Cache (MB) | Compression | PPL ↓ | LongBench ↑ |
|--------|-----------|-----------|-------|-----------|---------------|-------------|-------|-------------|
| C0 (FP16) | 1 815.9 | 38.3 | 26.14 | 18 169 | 937.5 | 1.00× | 7.460 | 0.505 |
| C1 (TailorKV) | 1 956.5 | 53.5 | 18.70 | 18 151 | 23.3 | **40.20×** speed / **34.60×** LongBench | 7.460 | 0.451 |
| C2 (QAQ) | 2 374.7 | 163.5 | 6.12 | 18 168 | 535.0 | **1.76×** | 7.460 | 0.505 |
| C3 (KVQuant) | 1 966.3 | 49.7 | 20.14 | 19 184 | 274.8 | **3.41×** | 7.460 | 0.510 |
| C4 (DynamicKV) | 1 932.1 | 49.4 | 20.26 | 18 151 | 128.0 | **7.32×** speed / **6.30×** LongBench | 7.460 | 0.499 |

### Phi-3-mini-4k-instruct  (speed prompt = 3 500 tokens)

| Config | TTFT (ms) | TPOT (ms) | Tok/s | VRAM (MB) | KV Cache (MB) | Compression | PPL ↓ | LongBench ↑ |
|--------|-----------|-----------|-------|-----------|---------------|-------------|-------|-------------|
| C0 (FP16) | 618.0 | 68.2 | 14.66 | 8 928 | 767.3 | 1.00× | 5.635 | 0.369 |
| C1 (TailorKV) | 759.9 | 28.6 | 34.93 | 8 910 | 70.0 | **10.97×** | 5.635 | 0.360 |
| C2 (QAQ) | 1 039.9 | 42.7 | 23.39 | 9 015 | 434.9 | **1.76×** | 5.635 | 0.377 |
| C3 (KVQuant) | 781.5 | 36.3 | 27.59 | 9 735 | 224.9 | **3.41×** | 5.635 | 0.371 |
| C4 (DynamicKV) | 772.5 | 74.9 | 13.35 | 8 910 | 192.0 | **4.00×** | 5.635 | 0.358 |

### Key Observations

- **Perplexity is identical across configs** within each model — KV compression methods that operate on the prefix only do not affect teacher-forced next-token prediction.
- **C1 reaches the highest compression** (35–40× on LLaMA-3, 11× on Phi-3) by combining per-layer 1-bit quantization for the dense layer 0 with aggressive SnapKV-style pruning (192 tokens) on the rest. LongBench drops by 10.7 % on LLaMA-3 and only 2.4 % on Phi-3 — the trade-off skews more aggressive than C2 or C4.
- **C2 achieves 1.76× KV cache compression with no LongBench quality loss** on LLaMA-3 (0.505 → 0.505) and a slight gain on Phi-3 (0.369 → 0.377), suggesting the attention-aware bit allocation can suppress irrelevant cache noise.
- **C3 reaches 3.41× compression at 4 bits with 1 % FP16 outliers** with no measurable quality loss: LongBench is preserved (LLaMA-3 0.505 → 0.510, Phi-3 0.369 → 0.371) and PPL matches the FP16 baseline. The effective bit budget (≈ 4.7 bits/element including scale/zero metadata and the dense-and-sparse outlier list) explains the 16 / 4.7 ≈ 3.41× ratio.
- **C4 reaches 4–7× compression** with a small LongBench drop (-1.2 % LLaMA-3, -3.0 % Phi-3). The LLaMA-3 compression ratio is higher because LongBench prompts (avg ≈ 6 455 tokens) far exceed the per-layer budget of 1 024.
- **TPOT overhead in C2 is a Python simulation artefact** — production hardware with packed 2–4 bit storage would see DRAM-bandwidth speedups over the FP16 baseline, not slowdowns.
- Phi-3's attention-aware decode is disabled (SDPA path) because its eager attention implementation is ~43× slower than SDPA; V-cache bits fall back to the K-formula in that case.

---

## Repository Structure

```
AdaptiveServe/
├── scripts/
│   ├── _common.py                  # Shared constants, scoring, PPL, LongBench loader
│   ├── benchmark_c0_baseline.py    # C0 baseline (FP16 full KV cache)
│   ├── benchmark_c1_tailorkv.py    # C1 TailorKV (hybrid 1-bit quant + SnapKV pruning)
│   ├── benchmark_c2_qaq.py         # C2 QAQ (variable-bit attention-aware quantization)
│   ├── benchmark_c3_kvquant.py     # C3 KVQuant (4-bit per-channel K + per-token V + 1 % outliers)
│   └── benchmark_c4_dynamickv.py   # C4 DynamicKV (per-layer attention-driven retention)
└── runs/
    ├── C0/{llama3,phi3}/results.json
    ├── C1/{llama3,phi3}/results.json
    ├── C2/{llama3,phi3}/results.json
    ├── C3/{llama3,phi3}/results.json
    └── C4/{llama3,phi3}/results.json
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

# C1 — TailorKV hybrid quantization + sparsity
python scripts/benchmark_c1_tailorkv.py --model phi3
python scripts/benchmark_c1_tailorkv.py --model llama3

# C2 — QAQ full quantization
python scripts/benchmark_c2_qaq.py --model phi3
python scripts/benchmark_c2_qaq.py --model llama3

# C3 — KVQuant 4-bit per-channel K + per-token V with 1 % FP16 outliers
python scripts/benchmark_c3_kvquant.py --model phi3
python scripts/benchmark_c3_kvquant.py --model llama3

# C4 — DynamicKV per-layer token retention
python scripts/benchmark_c4_dynamickv.py --model phi3
python scripts/benchmark_c4_dynamickv.py --model llama3
```

Results are written to `runs/{config}/{model}/results.json`.

Each benchmark measures:
1. **Speed**: fixed long prompt (3 500 tokens for Phi-3, 7 500 for LLaMA-3), 50-token decode run.
2. **Perplexity**: non-overlapping chunks over the full WikiText-2 test split.
3. **LongBench**: 20 samples per task, scored with task-specific metrics.

---

## References

> Dong et al. (2024). *QAQ: Quality Adaptive Quantization for LLM Key-Value Cache*.  
> arXiv:2403.04643. [https://arxiv.org/abs/2403.04643](https://arxiv.org/abs/2403.04643)

> Hooper et al. (2024). *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization*.  
> arXiv:2401.18079. [https://arxiv.org/abs/2401.18079](https://arxiv.org/abs/2401.18079)

> Zhou et al. (2024). *DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs*.  
> arXiv:2407.11550. [https://arxiv.org/abs/2407.11550](https://arxiv.org/abs/2407.11550)

> Yao et al. (2025). *TailorKV: A Hybrid Framework for Long-Context Inference via Tailored KV Cache Optimization*.  
> arXiv:2505.19586. [https://arxiv.org/abs/2505.19586](https://arxiv.org/abs/2505.19586)
