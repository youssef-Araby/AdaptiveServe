#!/bin/bash
# Full P0 re-run: every config x every model with the corrected scoring
# (official LongBench metrics), fixed compressors (QAQ decode bug, KVQuant
# outlier ranges, real DynamicKV), and the unified measured compression basis
# (metadata-inclusive bytes, sliding-window-aware FP16 reference).
#
# GPU phase : 6 configs x 4 models, LongBench only (--skip-speed-ppl).
# CPU phase : dataset build -> CV router (3 taus) -> candidate-pool sweep
#             -> C6 LOTO/split (3 taus) -> fair comparison -> Pareto figures.
#
# Resume-safe: each completed unit drops a .p0_done marker and is skipped on
# re-invocation. Logs to runs/p0/rerun_p0.log (tee'd by the caller).
set -u
cd "$(dirname "$0")/.."
unset ADAPTIVESERVE_LB_N   # full 20 prompts/task — never inherit a smoke value
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODELS="phi3 llama32_3b llama3 llama31_8b"
TAUS="0.99 0.95 0.90"

echo "=== P0 re-run started $(date) ==="

# ---- GPU phase -------------------------------------------------------------
for m in $MODELS; do
  for n in 0 1 2 3 4 5; do
    marker="runs/p0/C${n}/${m}/.p0_done"
    if [ -f "$marker" ]; then
      echo "--- SKIP C${n}/${m} (marker present)"
      continue
    fi
    script=$(ls scripts/benchmark_c${n}_*.py)
    echo "--- RUN C${n}/${m} ($script) $(date +%F' '%T)"
    python "$script" --model "$m" --skip-speed-ppl
    ec=$?
    if [ $ec -ne 0 ]; then
      echo "!!! FAIL C${n}/${m} exit=$ec — aborting so the failure is visible"
      exit 1
    fi
    touch "$marker"
  done
done

# ---- CPU phase -------------------------------------------------------------
for m in $MODELS; do
  echo "--- build_dataset $m $(date +%T)"
  python scripts/build_dataset.py --model "$m" || exit 1
done

# cv_router_220 iterates all models itself — run it once per tau:
for t in $TAUS; do
  echo "--- cv_router_220 tau=$t $(date +%T)"
  python scripts/cv_router_220.py --tau "$t" || exit 1
done

echo "--- candidate_pool_sweep corrected P0 $(date +%T)"
python scripts/candidate_pool_sweep.py \
  --run-id p0_corrected_2026-07-16 \
  --verify-existing-cv || exit 1

for m in $MODELS; do
  for t in $TAUS; do
    echo "--- c6 headline/LOTO $m tau=$t $(date +%T)"
    python scripts/benchmark_c6_classifier.py --model "$m" --tau "$t" || exit 1
  done
done

echo "--- fair_comparison_66 $(date +%T)"
python scripts/fair_comparison_66.py || echo "WARN: fair_comparison_66 failed (non-fatal)"

echo "--- figures $(date +%T)"
python scripts/plot_pareto_cv.py --all || exit 1
for m in $MODELS; do
  python scripts/plot_pareto.py --model "$m" || echo "WARN: plot_pareto $m failed (non-fatal)"
done

echo "=== P0 re-run COMPLETE $(date) ==="
