# LongBench16 24K Evaluation

This namespace contains the expanded evaluation specified in
[`dataset.md`](../../dataset.md):

- all 16 non-Chinese LongBench v1 tasks and all 3,750 official test examples;
- one immutable, tokenizer-finalized input panel capped at 24,000 tokens;
- C0–C5 quality and effective KV-storage labels (22,500 generations);
- post-hoc oracle analyses; and
- leakage-safe, held-out C6 routing without another language-model run.

Speed and perplexity are deliberately outside this milestone.

## Layout

Prepared token IDs are versioned separately from generation run IDs:

```text
runs/longbench16_24k/
├── README.md
├── prepared/
│   └── llama31_8b/
│       └── <prep-id>/
│           ├── manifest.json
│           ├── status.json
│           ├── index.jsonl
│           └── tokens/
└── llama31_8b/
    └── <run-id>/
        ├── manifest.json
        ├── environment.json
        ├── C0/ ... C6/
        ├── pilots/worst_case_24k_512/C0/ ... C5/
        └── analysis/
            ├── joined.jsonl
            ├── summary.json
            ├── c6_per_prompt.jsonl
            └── c6_summary.json
```

`llama31_8b` is the alias for the pinned
`meta-llama/Llama-3.1-8B-Instruct` revision in
[`configs/longbench16_24k.json`](../../configs/longbench16_24k.json).

## Canonical workflow

Set shell variables for one new prepared artifact and one new run ID:

```bash
PREP_DIR=runs/longbench16_24k/prepared/llama31_8b/<prep-id>
RUN_ID=<run-id>
RUN_DIR=runs/longbench16_24k/llama31_8b/$RUN_ID
```

The repository must be on a clean committed revision. Preparation pins that
commit, and every pilot, generation process, C6 evaluation, and finalization
must use the same clean revision.

The benchmark data remains external to Git. If the pinned release is not
already present at `~/.cache/longbench/data/`, fetch and verify it before
preparation:

```bash
mkdir -p "$HOME/.cache/longbench"
curl --fail --location \
  https://huggingface.co/datasets/zai-org/LongBench/resolve/5e628be450b7e67fb7ae6e201bd6d8f7056f7672/data.zip \
  --output "$HOME/.cache/longbench/data.zip"
echo "cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64  $HOME/.cache/longbench/data.zip" \
  | sha256sum --check
unzip -q "$HOME/.cache/longbench/data.zip" -d "$HOME/.cache/longbench"
```

Prepare and validate all 3,750 final token-ID sequences once:

```bash
python scripts/longbench16_prepare.py --output-dir "$PREP_DIR"
```

Run the six isolated memory-feasibility pilots, one process at a time:

```bash
python scripts/longbench16_run_config.py --configuration C0 \
  --prepared-dir "$PREP_DIR" --run-id "$RUN_ID" \
  --pilot --pilot-profile worst_case_24k_512
```

Repeat that command for C1, C2, C3, C4, and C5. Each pilot forces the complete
512-token decode after a 24,000-token input and must retain at least 1.5 GiB of
conservative physical-VRAM headroom. Final-mode generation is blocked until
all six pilot journals pass. Keep the same GPU, driver, Python environment,
package versions, and CUDA visibility for every process in the run.

Then run each canonical configuration:

```bash
python scripts/longbench16_run_config.py --configuration C0 \
  --prepared-dir "$PREP_DIR" --run-id "$RUN_ID"
```

Repeat for C1–C5. The journals are append-only and the same command safely
resumes incomplete work; completed examples are not regenerated.

After all 22,500 records exist, build the strict join and oracle analyses:

```bash
python scripts/longbench16_join.py \
  --run-dir "$RUN_DIR" --prepared-dir "$PREP_DIR"
```

Evaluate C6 from the joined labels; this is CPU-only and performs no LLM
generation:

```bash
python scripts/longbench16_c6.py --run-dir "$RUN_DIR"
```

Finally, independently revalidate and hash every prepared, pilot, C0–C6,
joined, oracle, and environment artifact before changing the run manifest
from `running` to immutable `complete`:

```bash
python scripts/longbench16_finalize.py \
  --run-dir "$RUN_DIR" --prepared-dir "$PREP_DIR"
```

Do not place expanded outputs under `runs/p0`; that namespace is protected
corrected evidence from the earlier experiment.
