#!/usr/bin/env bash
# Finish the full re-evaluation pipeline after llama3 C2 completes.
#
# Sequence:
#   1. Track C pipeline (C2 llama32_3b + full C0–C5 llama31_8b; builds datasets
#      for those two models on completion). Existing results.json files are
#      skipped — so on llama32_3b only C2 runs; on llama31_8b all 6 run.
#   2. Rebuild llama3 and phi3 datasets (track_c does not touch them, but their
#      C2 per_prompt.jsonl was overwritten by the paper-faithful re-run).
#   3. Train + evaluate the C6 router for all four models at three iso-quality
#      thresholds each.
#   4. Generate Pareto plots for all four models.
#
# Persistent log: runs/finish_all.log.
# Usage: launch in background only after llama3 C2 has finished.

set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
LOG="$REPO/runs/finish_all.log"
MODELS=("llama3" "phi3" "llama32_3b" "llama31_8b")
TAUS=("0.99" "0.95" "0.90")
START=$(date +%s)

banner() {
  local msg="$1"
  local now=$(date +"%H:%M:%S")
  local elapsed=$(( $(date +%s) - START ))
  printf "\n========================================================================\n"
  printf "[%s | +%5ds] %s\n" "$now" "$elapsed" "$msg"
  printf "========================================================================\n"
}

run() {
  local desc="$1"; shift
  banner "$desc"
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    banner "FAIL ($rc): $desc"
    exit $rc
  fi
}

banner "finish_all START"
echo "Log file: $LOG"

# 0. Resume LLaMA-3 C2 if it was interrupted (per_prompt.jsonl < 220 rows or
#    results.json older than the C2 code). Skip if already complete with the
#    new logger (i.e. row 1 has a "compression" field).
LLAMA3_C2_PP="runs/C2/llama3/per_prompt.jsonl"
need_llama3_c2=1
if [[ -f "$LLAMA3_C2_PP" ]]; then
  rows=$(wc -l < "$LLAMA3_C2_PP")
  has_cr=$(head -n1 "$LLAMA3_C2_PP" 2>/dev/null | grep -c '"compression"' || true)
  if [[ "$rows" == "220" && "$has_cr" == "1" ]]; then
    need_llama3_c2=0
  fi
fi
if [[ $need_llama3_c2 == 1 ]]; then
  run "C2 llama3 (re-run)" python scripts/benchmark_c2_qaq.py \
      --model llama3 --skip-speed-ppl
else
  banner "SKIP C2 llama3 (already complete with new logger)"
fi

# 1. Track C: C2 llama32_3b + full pipeline for llama31_8b, plus their datasets.
run "Track C pipeline" bash scripts/run_track_c_pipeline.sh

# 2. Rebuild datasets for the original 2 (their C2 per_prompt.jsonl just changed).
for m in llama3 phi3; do
  run "Build dataset: $m" python scripts/build_dataset.py --model "$m"
done

# 3. Train + evaluate C6 router for all 4 models × 3 taus.
for m in "${MODELS[@]}"; do
  for tau in "${TAUS[@]}"; do
    run "C6 router: $m @tau=$tau" python scripts/benchmark_c6_classifier.py \
        --model "$m" --tau "$tau"
  done
done

# 4. Pareto plots.
for m in "${MODELS[@]}"; do
  run "Pareto plot: $m" python scripts/plot_pareto.py --model "$m"
done

banner "finish_all COMPLETE  total=$(( $(date +%s) - START ))s"
