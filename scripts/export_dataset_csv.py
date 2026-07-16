#!/usr/bin/env python3
"""
Export the routing dataset to CSV files for submission.

Produces corrected-P0 exports:
    runs/dataset/adaptiveserve_kv_dataset.csv   — master file (all 880 rows, 4 models)
    runs/dataset/per_model/<model>.csv          — 4 split files (220 rows each)
    runs/dataset/manifest.json                  — source hashes and export provenance
    runs/dataset/README.md                      — column reference
    runs/dataset/AdaptiveServe-KV-Dataset.zip   — portable export bundle
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "runs" / "dataset"
OUT_PER_MODEL = SRC / "per_model"
OUT_PER_MODEL.mkdir(exist_ok=True)

MODELS = ["phi3", "llama3", "llama32_3b", "llama31_8b"]
CONFIGS = ["C0", "C1", "C2", "C3", "C4", "C5"]
FEATURES = ["seq_len_tokens", "seq_len_chars", "token_entropy",
            "gzip_ratio", "unique_token_ratio", "question_position", "newline_density"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


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
        out[f"kv_bytes_{c.lower()}"] = row["kv_bytes"][c]
        out[f"kv_bytes_fp16_{c.lower()}"] = row["kv_bytes_fp16"][c]
    out["best_quality_config"] = row.get("best_quality", "")
    out["best_quality_score"]  = row.get("best_quality_score", "")
    out["best_iso_tau_0.99"]   = row.get("best_iso_quality_99", "")
    out["best_iso_tau_0.95"]   = row.get("best_iso_quality_95", "")
    out["best_iso_tau_0.90"]   = row.get("best_iso_quality_90", "")
    return out


def main() -> None:
    all_rows: list[dict] = []
    source_files = []
    for model in MODELS:
        source_path = SRC / f"{model}.jsonl"
        rows = [flatten(model, json.loads(line))
                for line in source_path.read_text().splitlines()
                if line.strip()]
        per_path = OUT_PER_MODEL / f"{model}.csv"
        with per_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  wrote {per_path.relative_to(ROOT)}  ({len(rows)} rows)")
        all_rows.extend(rows)
        source_files.append({
            "model": model,
            "path": str(source_path.relative_to(ROOT)),
            "sha256": sha256_file(source_path),
            "rows": len(rows),
        })

    master = SRC / "adaptiveserve_kv_dataset.csv"
    with master.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  wrote {master.relative_to(ROOT)}  ({len(all_rows)} rows)")

    manifest = SRC / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "adaptive-serve-p0-csv-export/v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_commit": current_commit(),
        "source_run": "corrected P0 rerun; see runs/rerun_p0.log",
        "source_jsonl": source_files,
        "rows": len(all_rows),
    }, indent=2, sort_keys=True) + "\n")
    print(f"  wrote {manifest.relative_to(ROOT)}")

    readme = SRC / "README.md"
    readme.write_text("""# AdaptiveServe-KV Routing Dataset (Corrected P0)

A dataset of 880 (model, prompt, configuration) labelled rows from 220 LongBench
prompts evaluated under six KV-cache configurations on four open-weight LLMs.
It was regenerated from the corrected P0 JSONL inputs. See `manifest.json` for
the source hashes and source revision.

## Files

- `adaptiveserve_kv_dataset.csv` — master file, 880 rows (4 models × 220 prompts)
- `per_model/{phi3,llama3,llama31_8b,llama32_3b}.csv` — split by model, 220 rows each
- `manifest.json` — P0 source JSONL hashes and export provenance
- `AdaptiveServe-KV-Dataset.zip` — portable bundle of this README, manifest, CSV, and JSONL files
- Source JSONL files are kept under the same directory for code reproducibility

## Columns

### Identifiers
| column | meaning |
| --- | --- |
| `model` | one of `phi3`, `llama3`, `llama31_8b`, `llama32_3b` |
| `task` | LongBench task ID (e.g., `narrativeqa`, `gov_report`, `triviaqa`) |
| `sample_idx` | 0..19 — index within the task's 20-prompt sample |
| `metric` | LongBench scoring metric for this task (`f1`, `rouge1`, `accuracy`, `em`) |

### Surface features (7, prompt-only)
| column | meaning |
| --- | --- |
| `seq_len_tokens` | tokenizer length of the prompt |
| `seq_len_chars` | character count |
| `token_entropy` | Shannon entropy of token frequencies |
| `gzip_ratio` | gzip(prompt) bytes / raw bytes |
| `unique_token_ratio` | type/token ratio |
| `question_position` | offset of last `?` divided by prompt length |
| `newline_density` | newline count divided by prompt length |

### Per-configuration labels
For each configuration `c0..c5` (FP16, TailorKV-inspired hybrid, QAQ, KVQuant,
DynamicKV, Ada-KV-inspired shared-index eviction):

| column | meaning |
| --- | --- |
| `score_{c0..c5}` | LongBench task score under this configuration, in [0, 1] |
| `cr_{c0..c5}` | Measured KV-cache compression ratio `kv_bytes_fp16 / kv_bytes` for this prompt |
| `kv_bytes_{c0..c5}` | Effective measured KV-cache bytes, including method-specific metadata/decode accounting |
| `kv_bytes_fp16_{c0..c5}` | Sliding-window-aware FP16 reference bytes for the same prompt/configuration |

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

    archive = SRC / "AdaptiveServe-KV-Dataset.zip"
    bundle_paths = [master, manifest, readme]
    bundle_paths.extend(SRC / f"{model}.jsonl" for model in MODELS)
    bundle_paths.extend(OUT_PER_MODEL / f"{model}.csv" for model in MODELS)
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for bundle_path in bundle_paths:
            bundle.write(bundle_path, bundle_path.relative_to(SRC))
    print(f"  wrote {archive.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
