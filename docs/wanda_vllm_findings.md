# Wanda 2:4 + vLLM — Reproduction Results and Findings

**Status (2026-07-16):** this is a separate historical pruning/vLLM study. It is
not an AdaptiveServe-KV C0-C6 artifact and must not be combined with the P0
KV-cache results.

**Subject:** evaluation of Wanda 2:4 structured weight sparsity on Llama-3.1-8B served by vLLM, intended as a kernel-complete alternative to R-Sparse following an earlier planning document that is not retained in this repository. Documents the results actually obtained, the integration issues discovered, and the gap between the speedup claim in that planning document and what the current vLLM release can deliver.

**Hardware:** Colab Pro+, A100-SXM4-40GB. Notebook: [wanda_vllm_colab.ipynb](../notebooks/wanda_vllm_colab.ipynb).

---

## 1. Executive summary

The end-to-end Wanda-pruned-via-`llm-compressor` → vLLM-served evaluation pipeline now runs successfully, but neither headline number from the planning doc reproduced:

- **Accuracy:** Wanda 2:4 PIQA `acc_norm = 0.6300` on Llama-3.1-8B vs dense `0.8118` — an **18.2 pp drop**, far outside the 1-3 pp range cited for properly-tuned Wanda 2:4 in [Sun et al. 2024](https://arxiv.org/abs/2306.11695).
- **Speedup:** Wanda vs dense vLLM tok/s ratio is **1.19×**, within run-to-run noise on an 8-prompt benchmark. This is **not** a kernel speedup — vLLM 0.20 has removed the `Sparse24` kernel that would have produced one.

The pipeline correctness work is reusable. The speedup conclusion in the planning doc (`r_sparse_latency_analysis.md` §6, "Wanda + 2:4 + vLLM is kernel-complete and routinely cited") is no longer reachable on current vLLM and needs revision.

---

## 2. Setup

| | value |
|---|---|
| Base model | `meta-llama/Llama-3.1-8B` (BASE, not Instruct) |
| Sparsity method | Wanda 2:4 structured weight pruning |
| Sparsifier toolchain | `llmcompressor==0.10.0.2` |
| Calibration | `open_platypus`, 512 samples, `max_seq_length=2048` |
| Eval engine | vLLM 0.20.1 |
| Eval harness | `lm-eval==0.4.11`, vLLM backend |
| Task | PIQA, 0-shot, batch 1, full 1838 examples |
| Latency probe | 8 fixed prompts, greedy, `max_tokens=64` |
| GPU | NVIDIA A100-SXM4-40GB |

The dense-vs-sparse pair both use Llama-3.1-8B BASE so the only varying axis is sparsity. The original notebook compared Llama-3.0-Instruct (dense) against Llama-3.1-base (sparse), which mixed three changes into one number; that was corrected in this run.

---

## 3. Results

Full `cross_method_summary.json` rows. R-Sparse rows are reproduced from the earlier, untracked planning analysis for context.

| family | method | engine | piqa acc_norm | tok/s | within-engine speedup |
|---|---|---|---:|---:|---:|
| llama3 | dense | HF | 0.8906 | 183.7 | — |
| llama3 | R-Sparse | HF | 0.8750 | 66.5 | 0.36× |
| llama3 | **dense** | **vLLM** | **0.8118** | **497.7** | (baseline) |
| llama3 | **Wanda 2:4** | **vLLM** | **0.6300** | **593.8** | **1.19×** *(noise — see §5.4)* |
| mistral | dense | HF | 0.8906 | 180.2 | — |
| mistral | R-Sparse | HF | 0.8594 | 66.0 | 0.37× |
| llama32_3b | dense | HF | 0.8750 | 203.4 | — |
| llama32_3b | R-Sparse | HF | 0.7812 | 77.3 | 0.38× |

The bolded rows are this report's contribution. Everything else is prior context.

### Apples-to-apples reads
- **Within HF:** R-Sparse drops ~1-9 pp `acc_norm` and runs ~0.36-0.38× the speed of dense across all three families (consistent with the earlier latency analysis: kernels not engaged).
- **Within vLLM (this report's data):** Wanda 2:4 drops 18 pp `acc_norm` and runs 1.19× the speed of dense.
- **Cross-engine (HF vs vLLM dense):** 0.89 vs 0.81 looks like an 8 pp engine difference but is actually an artifact (§5.5).

---

## 4. The new pipeline (after fixes)

[`wanda_vllm_colab.ipynb`](../notebooks/wanda_vllm_colab.ipynb) §8 was modified to write a vLLM-loadable checkpoint. The fixes, in order applied:

1. **`save_compressed=False`** on `oneshot()`. Default `True` writes sparse-bitmask shards (`*.bitmask`, `*.compressed`, `*.shape`) which vLLM 0.20 cannot decompress (see §5.1).
2. **`save_pretrained(..., save_compressed=False)`** as belt-and-suspenders if the running `llmcompressor` version ignores the kwarg in `oneshot`.
3. **Strip `quantization_config` from `config.json` after save.** Even with dense weights, `llmcompressor` writes a `sparsity_config = {sparsity_structure: "2:4"}` block. vLLM 0.20 sees this and dispatches to a removed code path (§5.2).
4. **In-cell sanity check** that asserts `bitmask tensors == 0` and `45 < zeros% < 55` before declaring the checkpoint ready. Catches the bug in the sparsifier runtime instead of at eval time.

Effect: the saved checkpoint is plain dense fp16 with the Wanda 2:4 zero pattern preserved. vLLM loads it as a regular Llama, does no sparsity-specific dispatch, gets dense kernel performance.

---

## 5. Findings

### 5.1 vLLM 0.20 removed `Sparse24` kernel support

Stack trace from a pre-fix attempt:
```
NotImplementedError: Sparse24 models are no longer supported by vLLM
  at compressed_tensors_24.py:27
```

Any checkpoint declaring 2:4 sparsity via `quantization_config.sparsity_config` — including RedHatAI's `Sparse-Llama-3.1-8B-2of4`, Neural Magic's variants, or any `llmcompressor`-saved Wanda checkpoint — fails to load on vLLM 0.20.

**Implication for the earlier planning document.** It recommended Wanda 2:4 + vLLM specifically because "vLLM dispatches to native NVIDIA Sparse Tensor Cores automatically." That is no longer true on 0.20. The kernel-complete claim now requires either a pinned older vLLM (≤ 0.9 had `Sparse24`) or a different inference stack.

### 5.2 `llmcompressor` writes sparsity metadata even when saving dense weights

Setting `save_compressed=False` produces dense `.safetensors` shards with no bitmask tensors — but still writes `quantization_config.sparsity_config` to `config.json`. On vLLM 0.20 this routes through `CompressedTensors24.__init__` and triggers the §5.1 error before any weights load.

The only reliable way to get a vLLM-0.20-loadable Wanda checkpoint is to **strip the `quantization_config` block** after `llmcompressor` saves. This loses any future ability to auto-dispatch to a sparse kernel (irrelevant on this vLLM version anyway).

### 5.3 Wanda 2:4 accuracy is 8-15 pp below published baselines

| source | model | piqa acc_norm |
|---|---|---:|
| this run | Llama-3.1-8B Wanda 2:4 | **0.6300** |
| RedHatAI `Sparse-Llama-3.1-8B-2of4` | Llama-3.1-8B (their Wanda variant) | 0.8014 (verified independently in earlier eval — see notebook §13 results history) |
| Sun et al. 2024 Table 7 | Llama-2-7B Wanda 2:4 | ~0.747 |

**Probable causes**, ordered by likelihood:
1. **Calibration size.** 512 samples on `open_platypus` is below the original Wanda paper's setup (typically 1024-2048 on C4). For an 8B model this is the most likely driver.
2. **Calibration domain mismatch.** `open_platypus` is instruction-tuning data; the original Wanda evaluation used `c4` (general web text), which has broader distribution coverage.
3. **Sequence length.** `max_seq_length=2048` is fine for PIQA but may under-represent long-context patterns Llama-3.1 was trained on.
4. **Llama-3.1 is more pruning-sensitive than Llama-3.** RedHatAI's 0.80 number rules this out as a *full* explanation, but their checkpoint was almost certainly produced with much larger calibration.

**Not investigated:** rerunning §8 with `num_calibration_samples=2048, dataset='c4'` would isolate (1) and (2). Each rerun is ~60 min on A100.

### 5.4 The 1.19× tok/s "speedup" is measurement noise

The latency probe is 8 prompts × 64 tokens = 512 generated tokens, ~1 second wall-clock per run. At that scale, A100 run-to-run variance is easily ±20% from KV-cache state, allocation jitter, and scheduling. With vLLM 0.20's `Sparse24` removed (§5.1), there is no kernel path that would produce a real speedup — the engine is running the dense GEMM kernel over weights that happen to be 50% zero, doing the same FLOPs as the dense baseline.

A more honest read: **Wanda 2:4 vLLM ≈ dense vLLM in tok/s on this engine version.**

To get a credible number either way, the latency probe needs ~100+ prompts and ~256+ tokens each. As-is, the +1.19× should not be cited.

### 5.5 The HF baseline rows (0.89 acc_norm) are not directly comparable

The cross-method summary shows three families at ~0.89 `acc_norm` for HF dense. Llama-3.1-8B's true 0-shot PIQA `acc_norm` is ~0.81 (vLLM dense in this same run agrees at 0.8118). The 0.89 figures come from `colab_standalone.ipynb` runs that used `LIMIT=64` on PIQA — a 64-example subset has stderr ≈ 0.05, so the ~0.89 numbers are within noise of 0.81 but visually misleading.

**Action item:** rerun the HF-side baselines with `LIMIT=None` to get full-PIQA numbers comparable to the vLLM rows. Until that happens, do not cross-compare HF and vLLM rows in the same table.

---

## 6. Pipeline issues encountered (and resolved)

These are documented for the next person who attempts this — each consumed real debugging time.

| # | symptom | root cause | fix |
|---|---|---|---|
| 1 | `KeyError: 'layers.0.mlp.down_proj.bitmask'` on vLLM load | bitmask shards written but no `quantization_config` to dispatch decompressor | initially patched config.json; ultimately switched to `save_compressed=False` |
| 2 | Patched config still failed with the same KeyError | schema mismatch between `compressed-tensors` 0.15 expectations and the patch | dropped this approach |
| 3 | Re-saving via `transformers.AutoModelForCausalLM` produced a checkpoint scoring 0.53 acc_norm (≈ random) | transformers loaded bitmask bytes as raw fp16 because the partial `quantization_config` patch was unrecognized; `save_pretrained` then dumped that nonsense as dense | wrong direction; abandoned for the proper `save_compressed=False` rerun |
| 4 | `FileNotFoundError: 'lm_eval'` after kernel restart | Colab eval runtime needs §4 (vllm/lm-eval install) every fresh session | rerun §4 |
| 5 | `NotImplementedError: Sparse24 models are no longer supported` | vLLM 0.20 deprecation | strip `quantization_config` from saved checkpoint's config.json |

The notebook §8 now prevents 1, 2, 3, and 5 from recurring on a future run.

---

## 7. Recommendations

### For the accuracy story
The 0.6300 number is publishable as "Wanda 2:4 with minimal calibration" but should not be cited as a representative Wanda 2:4 result. **Rerun §8 with `num_calibration_samples=2048, dataset='c4'`** before drawing any conclusion about Wanda's accuracy on Llama-3.1.

### For the speedup story
Three options, ordered by effort:

1. **Drop the speedup angle from this notebook.** Re-frame as "Wanda accuracy on vLLM-deployable models." The `r_sparse_latency_analysis.md` recommendation to use Wanda+vLLM for production speed needs an asterisk: "as of vLLM 0.20, Sparse24 dispatch was removed; check current release support before relying on this."
2. **Pin `vllm<=0.9`.** Sparse24 was supported. Will likely require resolving torch/transformers compatibility constraints in §4 of the notebook.
3. **Switch inference engines.** TensorRT-LLM, sparseml-server, and recent SGLang releases retain 2:4 dispatch paths. None integrate with the current notebook scaffolding without ~half-day of rewiring.

### For the earlier R-Sparse planning document
Update the §7 recommendation to reflect that vLLM's sparsity story has regressed since the doc was written. Wanda 2:4 + vLLM is no longer the safe-default kernel-complete path it was in vLLM 0.6-0.9.

---

## 8. Artifacts

On Drive at `/content/drive/MyDrive/r_sparse/`:

```
results/
  llama3_dense_vllm.json              ← this report, dense baseline
  llama3_wanda_2of4_vllm.json         ← this report, Wanda
  llama3_baseline.json                ← from colab_standalone (HF dense, LIMIT=64)
  llama3_rsparse.json                 ← from colab_standalone (HF R-Sparse)
  mistral_*.json, llama32_3b_*.json   ← prior runs
  cross_method_summary.json           ← stitched table from §13

wanda_checkpoints/
  llama31-8b-wanda-2of4/              ← dense fp16, 50% zeros, vLLM-loadable
  llama31-8b-wanda-2of4-OLD-bitmask/  ← original failed save, kept for forensics
```

---

## 9. Open questions

1. Does Wanda 2:4 on Llama-3.1-8B with proper calibration (C4, 2048 samples) reach the 0.78-0.80 range RedHatAI's checkpoint sits at, or is there a Llama-3.1-specific accuracy regression vs. Llama-3.0?
2. Is there a vLLM commit between 0.9 and 0.20 where Sparse24 was retained, or did removal happen in a single release? Affects the difficulty of pinning back.
3. Does `llmcompressor` 0.11+ change the save format such that the `quantization_config`-strip step in §8 becomes unnecessary?

None block reporting the current results. All affect what the next iteration of this work would look like.
