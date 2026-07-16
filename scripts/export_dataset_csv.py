#!/usr/bin/env python3
"""
Export the routing dataset to CSV files for submission.

Produces:
  runs/dataset/adaptiveserve_kv_dataset.csv   — master file (all 880 rows, 4 models)
  runs/dataset/per_model/<model>.csv          — 4 split files (220 rows each)
  runs/dataset/README.md                      — column reference
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "runs" / "dataset"
OUT_PER_MODEL = SRC / "per_model"
OUT_PER_MODEL.mkdir(exist_ok=True)

MODELS = ["phi3", "llama3", "llama32_3b", "llama31_8b"]
CONFIGS = ["C0", "C1", "C2", "C3", "C4", "C5"]
FEATURES = ["seq_len_tokens", "seq_len_chars", "token_entropy",
            "gzip_ratio", "unique_token_ratio", "question_position", "newline_density"]


def flatten(model: str, row: dict) -> dict:
    out = {
        "model":      model,
        "task":       row["task"],
        "sample_idx": row["sample_idx"],
        "metric":     row["metric"],
    }
    for f in FEATURES:
        out[f] = row["features"][f]
    for c in CONFIGS:
        out[f"score_{c.lower()}"] = row["scores"][c]
        out[f"cr_{c.lower()}"]    = row["compression"][c]
    out["best_quality_config"] = row.get("best_quality", "")
    out["best_quality_score"]  = row.get("best_quality_score", "")
    out["best_iso_tau_0.99"]   = row.get("best_iso_quality_99", "")
    out["best_iso_tau_0.95"]   = row.get("best_iso_quality_95", "")
    out["best_iso_tau_0.90"]   = row.get("best_iso_quality_90", "")
    return out


def main() -> None:
    all_rows: list[dict] = []
    for model in MODELS:
        rows = [flatten(model, json.loads(line))
                for line in (SRC / f"{model}.jsonl").read_text().splitlines()
                if line.strip()]
        per_path = OUT_PER_MODEL / f"{model}.csv"
        with per_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  wrote {per_path.relative_to(ROOT)}  ({len(rows)} rows)")
        all_rows.extend(rows)

    master = SRC / "adaptiveserve_kv_dataset.csv"
    with master.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  wrote {master.relative_to(ROOT)}  ({len(all_rows)} rows)")

    readme = SRC / "README.md"
    readme.write_text("""# AdaptiveServe-KV Routing Dataset

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
""")
    print(f"  wrote {readme.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
