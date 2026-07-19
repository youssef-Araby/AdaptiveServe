# Planned LongBench16 24K Evaluation

This directory is reserved for the expanded evaluation defined in
[`dataset.md`](../../dataset.md):

- all 16 non-Chinese LongBench tasks;
- all 3,750 official test examples;
- a uniform 24,000-token formatted-input cap; and
- C0 through C5 fixed-configuration generations followed by held-out C6
  evaluation.

## Planned model and run layout

Outputs are separated by model alias and then by an immutable run ID:

```text
runs/longbench16_24k/
├── README.md
├── llama31_8b/
│   └── <run-id>/
└── <confirmation-model-alias>/
    └── <run-id>/
```

The primary model alias `llama31_8b` refers to
`meta-llama/Llama-3.1-8B-Instruct`. The confirmation checkpoint remains
undecided; when selected, it receives its own model alias rather than sharing
the primary model's directory. Each run directory must contain its own
machine-readable manifest and configuration-specific records so retries or
different source revisions cannot be mixed.

No model directories, run IDs, manifests, or expanded-run results exist yet.
The tree above defines the future layout only. Implementation must validate the
manifest requirements in `dataset.md` and must never reuse the corrected P0
`.p0_done` markers under `runs/p0/C0`–`runs/p0/C5`.
