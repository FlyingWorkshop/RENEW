#!/bin/bash
# run_finetune.sh — Finetune (naive + RENEW) for multiple seeds
#
# Assumes pretraining is already done (run run_pretrain.sh first).
#
# Usage:
#   bash renew/run_finetune.sh                    # seeds 0-9
#   bash renew/run_finetune.sh 0 1 2 3 4          # specific seeds
#   ENV=maze10 BUDGET=3200 bash renew/run_finetune.sh

set -e

# --- Config (override via env vars) ---
ENV="${ENV:-maze20}"
MAZE_SIZE="${MAZE_SIZE:-20}"
BUDGET="${BUDGET:-1600}"
NUM_ROUNDS="${NUM_ROUNDS:-3}"
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-3}"

# Seeds: use args if provided, else 0-9
if [ $# -gt 0 ]; then
    SEEDS=("$@")
else
    SEEDS=(0 1 2 3 4 5 6 7 8 9)
fi

RESULTS_BASE="out/${ENV}"
echo "=============================================="
echo "  RENEW — Finetuning (Naive + RENEW)"
echo "  Environment: ${ENV} (${MAZE_SIZE}x${MAZE_SIZE})"
echo "  Seeds: ${SEEDS[@]}"
echo "  Pref budget: ${BUDGET}"
echo "  RENEW rounds: ${NUM_ROUNDS}"
echo "  Ensemble size: ${ENSEMBLE_SIZE}"
echo "  Results: ${RESULTS_BASE}/"
echo "=============================================="

TOTAL_START=$(date +%s)

for SEED in "${SEEDS[@]}"; do
    SEED_DIR="${RESULTS_BASE}/seed_${SEED}"
    CKPT="${SEED_DIR}/pretrained.pkl"

    echo ""
    echo "====== SEED ${SEED} ======"

    if [ ! -f "${CKPT}" ]; then
        echo "  ERROR: No pretrained checkpoint at ${CKPT}"
        echo "  Run: bash renew/run_pretrain.sh ${SEED}"
        continue
    fi

    SEED_START=$(date +%s)

    if [ -f "${SEED_DIR}/renew/metrics.json" ] && [ -f "${SEED_DIR}/naive/metrics.json" ]; then
        echo "  Finetune results exist, skipping."
    else
        echo "  Finetuning (naive + RENEW, ensemble_size=${ENSEMBLE_SIZE})..."
        python renew/finetune.py \
            --checkpoint "${CKPT}" \
            --seed "${SEED}" \
            --maze-size "${MAZE_SIZE}" \
            --pref-budget "${BUDGET}" \
            --num-rounds "${NUM_ROUNDS}" \
            --ensemble-size "${ENSEMBLE_SIZE}" \
            --results-dir "${SEED_DIR}"
    fi

    SEED_END=$(date +%s)
    echo "  Seed ${SEED} done in $((SEED_END - SEED_START))s"
done

TOTAL_END=$(date +%s)
echo ""
echo "=============================================="
echo "  All ${#SEEDS[@]} seeds finetuned in $((TOTAL_END - TOTAL_START))s"
echo "  Results: ${RESULTS_BASE}/"
echo ""
echo "  To plot:"
echo "    python renew/plot_results.py --env-name ${ENV}"
echo "=============================================="