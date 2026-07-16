# AdaptiveServe-KV Results

This report consolidates the completed **P0 corrected rerun**. It uses only
the artifacts produced by the full rerun, which started on 2026-07-15 at
17:20 EEST and completed on 2026-07-16 at 04:34 EEST. The full execution log
is available in [runs/rerun_p0.log](runs/rerun_p0.log).

## Scope and Reading Guide

| Item | Value |
| --- | --- |
| Models | Phi-3-mini, LLaMA-3-8B, LLaMA-3.2-3B, LLaMA-3.1-8B |
| Prompts per model | 220 LongBench prompts: 20 prompts from each of 11 tasks |
| Fixed configurations | C0 through C5 |
| Router configuration | C6, seven prompt-only features and one regressor per fixed configuration |
| Primary router evaluation | 10-fold task-stratified cross-validation over all 220 prompts |
| Router thresholds | $\tau \in \{0.99, 0.95, 0.90\}$ |
| Quality metric | Mean per-prompt LongBench score; higher is better |
| Compression metric | Harmonic mean of measured per-prompt KV-cache compression ratios; higher is better |

The primary evidence is the fair 10-fold CV evaluation in
[runs/cv_router_220_tau0.99.json](runs/cv_router_220_tau0.99.json),
[runs/cv_router_220_tau0.95.json](runs/cv_router_220_tau0.95.json), and
[runs/cv_router_220_tau0.9.json](runs/cv_router_220_tau0.9.json). It routes
each prompt with a model that did not train on that prompt, then compares the
result against fixed methods on the same full 220-prompt set.

The P0 GPU phase used `--skip-speed-ppl`. Consequently, this document does
not present TTFT, TPOT, throughput, VRAM, or perplexity as newly measured P0
results. The router overhead values below were measured separately by C6 and
include feature extraction, tokenization, scaling, and six regressor calls.

## Configurations

| ID | Configuration | Role |
| --- | --- | --- |
| C0 | FP16 full KV cache | Uncompressed quality reference; never selected by C6 |
| C1 | TailorKV-inspired hybrid | Aggressive retention plus quantization |
| C2 | QAQ | Attention-aware variable-bit quantization |
| C3 | KVQuant | 4-bit K/V quantization with FP16 outliers |
| C4 | DynamicKV | Cross-layer adaptive token-retention budget |
| C5 | Ada-KV-inspired | Head-weighted token retention |
| C6 | AdaptiveServe-KV router | Selects one of C1-C5 per prompt |

## Fixed-Method Baselines

Each cell is `quality / compression`. These are the fixed-method results on
all 220 prompts and are the baseline values used by the primary CV analysis.

| Config | Phi-3-mini | LLaMA-3-8B | LLaMA-3.2-3B | LLaMA-3.1-8B |
| --- | ---: | ---: | ---: | ---: |
| C0 FP16 | 0.3033 / 1.000x | 0.4293 / 1.000x | 0.4031 / 1.000x | 0.4358 / 1.000x |
| C1 hybrid | 0.2934 / 6.803x | 0.3743 / 18.329x | 0.3353 / 17.506x | 0.3662 / 16.144x |
| C2 QAQ | 0.3175 / 4.622x | 0.4231 / 4.607x | 0.4046 / 4.647x | 0.4348 / 4.602x |
| C3 KVQuant | 0.3050 / 2.880x | 0.4302 / 3.240x | 0.4013 / 3.215x | 0.4352 / 3.196x |
| C4 DynamicKV | 0.2950 / 3.242x | 0.4233 / 4.697x | 0.3960 / 4.647x | 0.4297 / 4.632x |
| C5 Ada-KV-inspired | 0.3000 / 3.215x | 0.4280 / 4.705x | 0.3961 / 4.660x | 0.4315 / 4.633x |

## Primary Router Results: Fair 10-Fold CV

The C6 router is evaluated on all 220 prompts. `Delta quality vs C0` is the
absolute and relative difference from the FP16 baseline on the same prompts.

<table>
	<thead>
		<tr>
			<th>Model</th>
			<th>Tau</th>
			<th>Router quality</th>
			<th>Router compression</th>
			<th>Delta quality vs C0</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<th rowspan="3" scope="rowgroup">Phi-3-mini</th>
			<td>0.99</td>
			<td>0.3108</td>
			<td>4.520x</td>
			<td>+0.0075 (+2.5%)</td>
		</tr>
		<tr>
			<td>0.95</td>
			<td>0.3041</td>
			<td>4.650x</td>
			<td>+0.0008 (+0.3%)</td>
		</tr>
		<tr>
			<td>0.90</td>
			<td>0.3040</td>
			<td>4.810x</td>
			<td>+0.0007 (+0.2%)</td>
		</tr>
		<tr>
			<th rowspan="3" scope="rowgroup">LLaMA-3-8B</th>
			<td>0.99</td>
			<td>0.4274</td>
			<td>5.247x</td>
			<td>-0.0019 (-0.4%)</td>
		</tr>
		<tr>
			<td>0.95</td>
			<td>0.4211</td>
			<td>5.542x</td>
			<td>-0.0082 (-1.9%)</td>
		</tr>
		<tr>
			<td>0.90</td>
			<td>0.4189</td>
			<td>5.993x</td>
			<td>-0.0104 (-2.4%)</td>
		</tr>
		<tr>
			<th rowspan="3" scope="rowgroup">LLaMA-3.2-3B</th>
			<td>0.99</td>
			<td>0.3907</td>
			<td>4.785x</td>
			<td>-0.0124 (-3.1%)</td>
		</tr>
		<tr>
			<td>0.95</td>
			<td>0.3869</td>
			<td>4.988x</td>
			<td>-0.0162 (-4.0%)</td>
		</tr>
		<tr>
			<td>0.90</td>
			<td>0.3779</td>
			<td>5.409x</td>
			<td>-0.0252 (-6.3%)</td>
		</tr>
		<tr>
			<th rowspan="3" scope="rowgroup">LLaMA-3.1-8B</th>
			<td>0.99</td>
			<td>0.4239</td>
			<td>5.017x</td>
			<td>-0.0119 (-2.7%)</td>
		</tr>
		<tr>
			<td>0.95</td>
			<td>0.4229</td>
			<td>5.238x</td>
			<td>-0.0129 (-3.0%)</td>
		</tr>
		<tr>
			<td>0.90</td>
			<td>0.4227</td>
			<td>5.415x</td>
			<td>-0.0131 (-3.0%)</td>
		</tr>
	</tbody>
</table>

### Pareto Summary at $\tau = 0.99$

| Model | Router outcome against fixed methods |
| --- | --- |
| Phi-3-mini | C6 dominates C0, C3, C4, and C5. C2 QAQ dominates C6. C1 remains a higher-compression/lower-quality trade-off. |
| LLaMA-3-8B | C6 dominates C2 QAQ and C4 DynamicKV. C0, C1, C3, and C5 remain non-dominated trade-offs. |
| LLaMA-3.2-3B | No fixed configuration strictly dominates C6, and C6 strictly dominates none. |
| LLaMA-3.1-8B | No fixed configuration strictly dominates C6, and C6 strictly dominates none. |

### C6 Selection Mix at $\tau = 0.99$

Counts show which compressor C6 chose across the 220 held-out CV decisions.
C0 is absent because it is only a quality reference.

| Model | C1 | C2 | C3 | C4 | C5 |
| --- | ---: | ---: | ---: | ---: |
| Phi-3-mini | 76 | 69 | 26 | 22 | 27 |
| LLaMA-3-8B | 47 | 46 | 39 | 20 | 68 |
| LLaMA-3.2-3B | 42 | 69 | 38 | 34 | 37 |
| LLaMA-3.1-8B | 47 | 59 | 53 | 23 | 38 |

## Supplementary Router Evaluations

These outputs are useful diagnostics but are not the primary comparison.
LOTO is leave-one-task-out evaluation, where each held-out task is unseen at
training time. The 70/30 split is one random seed-0 split and therefore has a
smaller, noisier 66-prompt test set.

<table>
	<thead>
		<tr>
			<th>Model</th>
			<th>Tau</th>
			<th>LOTO quality</th>
			<th>LOTO compression</th>
			<th>70/30 quality</th>
			<th>70/30 compression</th>
			<th>Median overhead</th>
			<th>P95 overhead</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<th rowspan="3" scope="rowgroup">Phi-3-mini</th>
			<td>0.99</td><td>0.3135</td><td>4.824x</td><td>0.3280</td><td>4.136x</td><td>22.731 ms</td><td>40.636 ms</td>
		</tr>
		<tr>
			<td>0.95</td><td>0.3092</td><td>4.957x</td><td>0.3285</td><td>4.439x</td><td>21.651 ms</td><td>41.578 ms</td>
		</tr>
		<tr>
			<td>0.90</td><td>0.3071</td><td>5.123x</td><td>0.3285</td><td>4.500x</td><td>22.386 ms</td><td>42.481 ms</td>
		</tr>
		<tr>
			<th rowspan="3" scope="rowgroup">LLaMA-3-8B</th>
			<td>0.99</td><td>0.4152</td><td>5.170x</td><td>0.4077</td><td>6.390x</td><td>23.568 ms</td><td>50.237 ms</td>
		</tr>
		<tr>
			<td>0.95</td><td>0.4136</td><td>5.643x</td><td>0.4087</td><td>6.489x</td><td>24.862 ms</td><td>47.100 ms</td>
		</tr>
		<tr>
			<td>0.90</td><td>0.4106</td><td>6.337x</td><td>0.4082</td><td>7.019x</td><td>23.643 ms</td><td>45.248 ms</td>
		</tr>
		<tr>
			<th rowspan="3" scope="rowgroup">LLaMA-3.2-3B</th>
			<td>0.99</td><td>0.3773</td><td>4.866x</td><td>0.3817</td><td>4.477x</td><td>24.691 ms</td><td>48.831 ms</td>
		</tr>
		<tr>
			<td>0.95</td><td>0.3737</td><td>4.992x</td><td>0.3779</td><td>5.018x</td><td>23.956 ms</td><td>45.954 ms</td>
		</tr>
		<tr>
			<td>0.90</td><td>0.3670</td><td>5.300x</td><td>0.3779</td><td>5.110x</td><td>23.049 ms</td><td>46.912 ms</td>
		</tr>
		<tr>
			<th rowspan="3" scope="rowgroup">LLaMA-3.1-8B</th>
			<td>0.99</td><td>0.4042</td><td>5.249x</td><td>0.3837</td><td>4.259x</td><td>24.722 ms</td><td>45.251 ms</td>
		</tr>
		<tr>
			<td>0.95</td><td>0.4033</td><td>5.312x</td><td>0.3839</td><td>4.355x</td><td>23.846 ms</td><td>52.268 ms</td>
		</tr>
		<tr>
			<td>0.90</td><td>0.4026</td><td>5.553x</td><td>0.3858</td><td>4.556x</td><td>23.841 ms</td><td>48.395 ms</td>
		</tr>
	</tbody>
</table>

## Router Regressor Fit: 70/30 Split Diagnostic

C6 is not a classifier with one accuracy value. It trains six regressors to
predict continuous per-prompt LongBench quality, one each for C0-C5. The
relevant fit measures are $R^2$ and mean absolute error (MAE), not
classification accuracy.

These numbers come from the one seed-0 70/30 split used by C6: 154 training
prompts and 66 held-out prompts per model. They are saved under
`regressor_fit` in each `runs/C6/<model>/results_tau0.99.json` file. The
regressor fit is independent of $\tau$, because $\tau$ changes the routing
rule after quality prediction rather than the regressor training; the same
fit values are therefore repeated in the $\tau = 0.95$ and $0.90$ files.

These tau-specific files are the post-fix P0 outputs: the pipeline first
reran C0-C5, rebuilt each 220-prompt dataset, then retrained C6. Use these
files rather than unversioned `results.json` files or `_filtered` ablation
artifacts elsewhere under `runs/C6`.

| Model | Train $R^2$ (mean C0-C5) | Held-out $R^2$ (mean C0-C5) | Train MAE (mean C0-C5) | Held-out MAE (mean C0-C5) |
| --- | ---: | ---: | ---: | ---: |
| Phi-3-mini | 0.8403 | -0.0774 | 0.0999 | 0.2344 |
| LLaMA-3-8B | 0.7765 | -0.0661 | 0.1389 | 0.2831 |
| LLaMA-3.2-3B | 0.7562 | 0.1382 | 0.1415 | 0.2546 |
| LLaMA-3.1-8B | 0.7883 | 0.1969 | 0.1333 | 0.2292 |

### Held-Out Fit by Regressor

Each cell is `held-out R2 / held-out MAE` on the same 66-prompt split. A
negative $R^2$ means that regressor predicts worse than a constant
test-set-mean score predictor for that target configuration.

| Model | C0 | C1 | C2 | C3 | C4 | C5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phi-3-mini | -0.1546 / 0.2425 | 0.0179 / 0.2224 | -0.1744 / 0.2620 | -0.0676 / 0.2268 | -0.0213 / 0.2183 | -0.0643 / 0.2346 |
| LLaMA-3-8B | -0.0932 / 0.2875 | 0.0109 / 0.2743 | -0.0451 / 0.2712 | -0.1270 / 0.2930 | -0.0739 / 0.2878 | -0.0685 / 0.2846 |
| LLaMA-3.2-3B | 0.1081 / 0.2543 | 0.1202 / 0.2493 | 0.1266 / 0.2693 | 0.1567 / 0.2523 | 0.1600 / 0.2508 | 0.1578 / 0.2513 |
| LLaMA-3.1-8B | 0.1747 / 0.2278 | 0.1701 / 0.2423 | 0.2739 / 0.2180 | 0.1891 / 0.2263 | 0.1855 / 0.2295 | 0.1880 / 0.2316 |

### What This Does and Does Not Establish

| Question | Answer |
| --- | --- |
| Is regression accuracy calculated? | Yes. C6 calculates per-config train/test $R^2$ and MAE for the 70/30 split. |
| Is there one router classification accuracy? | No. C6 predicts six continuous quality scores and then applies an iso-quality routing rule; it is not trained to reproduce one discrete label. |
| Do the primary 10-fold CV files contain aggregate regression $R^2$/MAE? | No. [scripts/cv_router_220.py](scripts/cv_router_220.py) evaluates routed quality and compression, but does not save fold-aggregated score-prediction metrics. |
| What do the current diagnostics say? | The train $R^2$ values are high while held-out $R^2$ is weak or negative for Phi-3 and LLaMA-3. This is evidence of limited score-prediction generalization and should qualify any strong routing claim. |

## Fixed Methods on the Exact 66-Prompt Split Test Set

This table addresses the earlier comparison mismatch by calculating every
fixed method on the exact 66 prompts used by the seed-0 70/30 split router
evaluation. Each cell is `quality / compression`.

| Config | Phi-3-mini | LLaMA-3-8B | LLaMA-3.2-3B | LLaMA-3.1-8B |
| --- | ---: | ---: | ---: | ---: |
| C0 FP16 | 0.3285 / 1.000x | 0.4440 / 1.000x | 0.4230 / 1.000x | 0.4249 / 1.000x |
| C1 hybrid | 0.3057 / 6.048x | 0.3700 / 15.466x | 0.3402 / 14.703x | 0.3522 / 13.470x |
| C2 QAQ | 0.3465 / 4.619x | 0.4234 / 4.607x | 0.4227 / 4.645x | 0.4095 / 4.602x |
| C3 KVQuant | 0.3283 / 2.765x | 0.4431 / 3.176x | 0.4200 / 3.140x | 0.4264 / 3.111x |
| C4 DynamicKV | 0.3072 / 3.049x | 0.4327 / 4.419x | 0.4135 / 4.374x | 0.4081 / 4.350x |
| C5 Ada-KV-inspired | 0.3138 / 2.992x | 0.4368 / 4.426x | 0.4087 / 4.391x | 0.4062 / 4.338x |

## Interpretation

| Question | Result supported by P0 |
| --- | --- |
| Does C6 beat FP16 on every model? | No. At $\tau = 0.99$, only Phi-3 improves over C0 quality; the three LLaMA models trade a small-to-moderate quality loss for substantially higher compression. |
| Does C6 beat every fixed compressor? | No. On Phi-3, C2 QAQ dominates C6 at $\tau = 0.99$. On LLaMA-3, C6 dominates C2 and C4 but not all fixed methods. On LLaMA-3.1 and LLaMA-3.2, there is no strict dominance. |
| What is the strongest conservative claim? | C6 can create useful quality/compression operating points, especially on Phi-3 and LLaMA-3, but its benefit is model- and threshold-dependent. |
| Is the old microsecond router-overhead claim current? | No. The P0 C6 measurements are approximately 22-25 ms median per long prompt when feature extraction and tokenization are included. |

## Source Artifacts

| Artifact | Contents |
| --- | --- |
| [scripts/run_full_pipeline.sh](scripts/run_full_pipeline.sh) | P0 execution plan and corrected-rerun scope |
| [runs/rerun_p0.log](runs/rerun_p0.log) | Complete P0 execution log and completion marker |
| [runs/cv_router_220_tau0.99.json](runs/cv_router_220_tau0.99.json) | Primary CV results at $\tau = 0.99$ |
| [runs/cv_router_220_tau0.95.json](runs/cv_router_220_tau0.95.json) | Primary CV results at $\tau = 0.95$ |
| [runs/cv_router_220_tau0.9.json](runs/cv_router_220_tau0.9.json) | Primary CV results at $\tau = 0.90$ |
| [runs/fair_comparison_66.json](runs/fair_comparison_66.json) | Fixed-method results on the exact split-test prompts |
| [runs/C6](runs/C6) | Per-model C6 LOTO, split, overhead, and per-prompt outputs |
| [runs/figs](runs/figs) | Updated quality/compression Pareto figures |