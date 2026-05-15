#!/usr/bin/env bash
# Track C data pipeline: run C0–C5 LongBench for llama32_3b and llama31_8b,
# then build their joined datasets. Sequential by design (single GPU).
#
# Progress markers print to stdout so you can watch the terminal directly.
# A persistent log is also written to runs/track_c_pipeline.log.

set -uo pipefail

cd "$(dirname "$0")/.."
REPO="$PWD"
LOG="$REPO/runs/track_c_pipeline.log"
mkdir -p "$REPO/runs"

# Order: C0 (baseline), then cheap configs (C3, C4, C5), then heavier (C1, C2 last).
CONFIGS=("c0_baseline" "c3_kvquant" "c4_dynamickv" "c5_adakv" "c1_tailorkv" "c2_qaq")
MODELS=("llama32_3b" "llama31_8b")

START=$(date +%s)
banner() {
  local msg="$1"
  local now=$(date +"%H:%M:%S")
  local elapsed=$(( $(date +%s) - START ))
  printf "\n========================================================================\n"
  printf "[%s | +%4ds] %s\n" "$now" "$elapsed" "$msg"
  printf "========================================================================\n"
}

banner "Track C pipeline START  (models: ${MODELS[*]})"
echo "Log file: $LOG"

for model in "${MODELS[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    script="scripts/benchmark_${cfg}.py"
    cfg_id=$(echo "$cfg" | cut -d_ -f1 | tr a-z A-Z)
    out_dir="runs/${cfg_id}/${model}"

    # Skip if results.json already exists (lets you resume after a crash).
    if [[ -f "$out_dir/results.json" ]]; then
      banner "SKIP  ${cfg_id}/${model}  (results.json already exists)"
      continue
    fi

    banner "RUN   ${cfg_id}/${model}   (script: $script)"
    python "$script" --model "$model" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
      banner "FAIL  ${cfg_id}/${model}  exit=$rc  — pipeline halted"
      exit $rc
    fi
    banner "DONE  ${cfg_id}/${model}"
  done

  banner "BUILD dataset for ${model}"
  python scripts/build_dataset.py --model "$model" 2>&1 | tee -a "$LOG"
done

banner "Track C pipeline COMPLETE  total=$(( $(date +%s) - START ))s"
