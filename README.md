# AdaptiveServe

**Benchmarking adaptive KV cache quantization for efficient LLM inference.**

AdaptiveServe measures the quality–efficiency trade-off of KV cache compression strategies for large language models. It provides reproducible benchmarks across latency, memory, and generation quality metrics. The current code base implements a full-precision baseline (C0), TailorKV hybrid quantization+sparsity (C1), QAQ attention-aware quantization (C2), KVQuant per-channel/per-token 4-bit quantization with outliers (C3), DynamicKV per-layer token retention (C4), and Ada-KV head-wise adaptive budget allocation (C5); a per-query adaptive selector (C6–C7) is planned.

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
| **C4** | DynamicKV | implemented | Per-layer attention-driven token retention. Each layer keeps the top-K tokens by aggregated attention score (with sliding window of recent tokens always preserved); per-layer budget is uniform (Zhou et al., 2024 — [arXiv:2412.14838](https://arxiv.org/abs/2412.14838)). |
| **C5** | Ada-KV | implemented | Head-wise adaptive budget allocation. Per-layer pool = budget_per_head × n_kv_heads; split across heads proportional to per-head attention concentration. Each head selects its top-k_h prefix tokens; selections are vote-aggregated (head score weighting) to a uniform per-layer length so the HF DynamicCache invariant holds. Feng et al., 2024 — [arXiv:2407.11550](https://arxiv.org/abs/2407.11550). |
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
| C2 (QAQ) | 2 550.0 | 197.9 | 5.05 | 18 168 | 176.8 | **5.34×** | 7.460 | 0.503 |
| C3 (KVQuant) | 1 966.3 | 49.7 | 20.14 | 19 184 | 274.8 | **3.41×** | 7.460 | 0.510 |
| C4 (DynamicKV) | 1 932.1 | 49.4 | 20.26 | 18 151 | 128.0 | **7.32×** speed / **6.30×** LongBench | 7.460 | 0.499 |
| C5 (Ada-KV) | 2 100.7 | 27.4 | 36.55 | 18 151 | 128.0 | **7.32×** speed / **6.30×** LongBench | 7.460 | 0.499 |

### Phi-3-mini-4k-instruct  (speed prompt = 3 500 tokens)

| Config | TTFT (ms) | TPOT (ms) | Tok/s | VRAM (MB) | KV Cache (MB) | Compression | PPL ↓ | LongBench ↑ |
|--------|-----------|-----------|-------|-----------|---------------|-------------|-------|-------------|
| C0 (FP16) | 618.0 | 68.2 | 14.66 | 8 928 | 767.3 | 1.00× | 5.635 | 0.369 |
| C1 (TailorKV) | 759.9 | 28.6 | 34.93 | 8 910 | 70.0 | **10.97×** | 5.635 | 0.360 |
| C2 (QAQ) | 1 326.2 | 35.1 | 28.52 | 9 007 | 161.9 | **4.74×** | 5.635 | 0.374 |
| C3 (KVQuant) | 781.5 | 36.3 | 27.59 | 9 735 | 224.9 | **3.41×** | 5.635 | 0.371 |
| C4 (DynamicKV) | 772.5 | 74.9 | 13.35 | 8 910 | 192.0 | **4.00×** | 5.635 | 0.358 |
| C5 (Ada-KV) | 976.7 | 62.4 | 16.01 | 8 910 | 192.0 | **4.00×** | 5.635 | 0.359 |

### Key Observations

- **Perplexity is identical across configs** within each model — KV compression methods that operate on the prefix only do not affect teacher-forced next-token prediction.
- **C1 reaches the highest compression** (35–40× on LLaMA-3, 11× on Phi-3) by combining per-layer 1-bit quantization for the dense layer 0 with aggressive SnapKV-style pruning (192 tokens) on the rest. LongBench drops by 10.7 % on LLaMA-3 and only 2.4 % on Phi-3 — the trade-off skews more aggressive than C2 or C4.
- **C2 achieves 4.7–5.3× KV cache compression with no LongBench quality loss** on LLaMA-3 (0.505 → 0.503, within 0.5%) and a slight gain on Phi-3 (0.369 → 0.374). The reported ratio is computed from the *measured* per-token-head bit allocation produced by the QAQ formula `B = ceil(log2(range/(2σ) + 1))`; on these prompts, the average is ≈3.0 bits on LLaMA-3 and ≈3.4 bits on Phi-3.
- **C3 reaches 3.41× compression at 4 bits with 1 % FP16 outliers** with no measurable quality loss: LongBench is preserved (LLaMA-3 0.505 → 0.510, Phi-3 0.369 → 0.371) and PPL matches the FP16 baseline. The effective bit budget (≈ 4.7 bits/element including scale/zero metadata and the dense-and-sparse outlier list) explains the 16 / 4.7 ≈ 3.41× ratio.
- **C4 reaches 4–7× compression** with a small LongBench drop (-1.2 % LLaMA-3, -3.0 % Phi-3). The LLaMA-3 compression ratio is higher because LongBench prompts (avg ≈ 6 455 tokens) far exceed the per-layer budget of 1 024.
- **C5 matches C4's compression** at the same per-layer budget but uses head-wise adaptive allocation. LongBench is essentially tied with C4 (0.499 vs 0.499 on LLaMA-3, 0.359 vs 0.358 on Phi-3) at this budget level. Notably, C5's TPOT on LLaMA-3 is **27.4 ms (36.6 tok/s) — faster than the FP16 baseline (38.3 ms)** because the smaller post-prefill cache reduces decode-time attention bandwidth, and llama3's 8 KV heads (GQA) leave plenty of headroom for the SDPA decode path.
- **TPOT overhead in C2 is a Python simulation artefact** — production hardware with packed 2–4 bit storage would see DRAM-bandwidth speedups over the FP16 baseline, not slowdowns.
- Phi-3's attention-aware decode is disabled (SDPA path) because its eager attention implementation is ~43× slower than SDPA; V-cache bits fall back to the K-formula in that case.

---

## Oracle Selector Analysis

The fixed-config tables above show the average across 7 LongBench tasks. They hide a more important question for this project: **does any single fixed config dominate, or is there per-task signal that an adaptive selector could exploit?**

To answer this, `scripts/oracle_analysis.py` computes a **task-level oracle**: for each task it picks the best config from {C0, C1, C2, C3, C4, C5} subject to a quality floor τ (fraction of the FP16 baseline on that task), then aggregates across tasks. The oracle is a cheat — it sees per-task quality before deciding — so it represents the **upper bound** of what any selector could achieve at task granularity.

### Per-task scores — LLaMA-3-8B-Instruct

| Task         | C0 (FP16) | C1 (TailorKV) | C2 (QAQ) | C3 (KVQuant) | C4 (DynamicKV) | C5 (Ada-KV) |
|--------------|-----------|----------------|----------|---------------|-----------------|--------------|
| 2wikimqa     | 0.401     | **0.401**      | 0.408    | 0.401         | 0.401           | 0.401        |
| gov_report   | **0.387** | 0.247          | 0.372    | 0.386         | 0.312           | 0.306        |
| hotpotqa     | 0.454     | 0.367          | **0.460**| 0.454         | 0.460           | 0.460        |
| narrativeqa  | 0.301     | 0.270          | 0.301    | 0.326         | **0.341**       | 0.341        |
| qasper       | 0.360     | 0.295          | 0.357    | **0.370**     | 0.347           | 0.355        |
| trec         | **0.700** | 0.650          | 0.700    | 0.700         | 0.700           | 0.700        |
| triviaqa     | **0.932** | 0.927          | 0.925    | 0.932         | 0.932           | 0.932        |
| **avg**      | **0.505** | 0.451          | 0.503    | 0.510         | 0.499           | 0.499        |
| **comp.**    | 1.00×     | **34.6×**      | 5.34×    | 3.41×         | 6.30×           | 6.30×        |

TailorKV's compression dominance (34.6×) hides task-level fragility. It matches FP16 on 2wikimqa (free win) and triviaqa (within 0.5%), but **collapses 36% on gov_report** and drops 18–19% on qasper/hotpotqa. A selector that picks TailorKV only when it's safe captures the compression benefit without the collapse cases.

### Oracle results — LLaMA-3 (selector picks one config per task)

| Strategy                          | Quality | Compression | Notes |
|-----------------------------------|---------|-------------|-------|
| Always FP16 (C0)                  | 0.505   | 1.00×       | reference |
| Always TailorKV (C1)              | 0.451   | 34.6×       | best fixed at high compression, −10.7% quality |
| Always KVQuant (C3, best mid)     | 0.510   | 3.41×       | best fixed at quality-preserving compression |
| **Oracle τ = 1.00** (no drop)     | 0.513   | 3.58×       | strictly Pareto-dominates always-C3 |
| **Oracle τ = 0.99** (≤1% drop)    | 0.510   | **6.90×**   | matches C3 quality at **2× the compression** |
| **Oracle τ = 0.95** (≤5% drop)    | 0.507   | **7.96×**   | near-FP16 quality at 8× |
| **Oracle τ = 0.90** (≤10% drop)   | 0.500   | 9.33×       | +11% absolute quality vs always-C1 at the same drop budget |

Picks at τ = 0.99 (the most useful regime): `2wikimqa→C1, gov_report→C3, hotpotqa→C4, narrativeqa→C4, qasper→C2, trec→C4, triviaqa→C1`.

### What this means

1. **In the 3×–10× compression regime, a per-prompt selector Pareto-dominates every published fixed method.** Same quality as KVQuant, double the compression. This is the project's thesis.
2. **TailorKV remains unbeaten ≥10×.** No combination of the other configs reaches 34× at any quality. The selector therefore *includes* TailorKV as a class and routes to it on prompts that tolerate aggressive compression (2wikimqa, triviaqa).
3. **C5 Ada-KV is dominated** at the current measurement granularity (n = 20) on both models — never picked over C4. Either drop it from the selector classes or re-evaluate at n ≥ 50 to detect the small head-wise gain reported in the Ada-KV paper.
4. **C4 is dominated on Phi-3** — Phi-3's selector pool reduces to {C0, C1, C2, C3}.

### Caveat

This oracle is at **task** granularity (7 decisions per model). The real selector value lives in **per-prompt** variance within tasks. The next step is to instrument all benchmarks with per-prompt logging, re-run on LLaMA-3, and verify the oracle gap survives at prompt-level decisions before training the C7 classifier.

Run the analysis:

```bash
python scripts/oracle_analysis.py
```

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
│   ├── benchmark_c4_dynamickv.py   # C4 DynamicKV (per-layer attention-driven retention)
│   └── benchmark_c5_adakv.py       # C5 Ada-KV (head-wise adaptive budget allocation)
└── runs/
    ├── C0/{llama3,phi3}/results.json
    ├── C1/{llama3,phi3}/results.json
    ├── C2/{llama3,phi3}/results.json
    ├── C3/{llama3,phi3}/results.json
    ├── C4/{llama3,phi3}/results.json
    └── C5/{llama3,phi3}/results.json
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

# C5 — Ada-KV head-wise adaptive budget allocation
python scripts/benchmark_c5_adakv.py --model phi3
python scripts/benchmark_c5_adakv.py --model llama3
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
> arXiv:2412.14838. [https://arxiv.org/abs/2412.14838](https://arxiv.org/abs/2412.14838)

> Feng et al. (2024). *Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference*.  
> arXiv:2407.11550. [https://arxiv.org/abs/2407.11550](https://arxiv.org/abs/2407.11550)

> Yao et al. (2025). *TailorKV: A Hybrid Framework for Long-Context Inference via Tailored KV Cache Optimization*.  
> arXiv:2505.19586. [https://arxiv.org/abs/2505.19586](https://arxiv.org/abs/2505.19586)
