#!/bin/bash
# run_pretrain.sh — Pretrain world models for multiple seeds
#
# Usage:
#   bash renew/run_pretrain.sh                    # seeds 0-9
#   bash renew/run_pretrain.sh 0 1 2 3 4          # specific seeds
#   ENV=maze10 PRETRAIN_STEPS=500 bash renew/run_pretrain.sh

set -e

# --- Config (override via env vars) ---
ENV="${ENV:-maze20}"
MAZE_SIZE="${MAZE_SIZE:-20}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-2000}"
DATASET_SIZE="${DATASET_SIZE:-50}"
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-3}"

# Seeds: use args if provided, else 0-9
if [ $# -gt 0 ]; then
    SEEDS=("$@")
else
    SEEDS=(0 1 2 3 4 5 6 7 8 9)
fi

RESULTS_BASE="out/${ENV}"
echo "=============================================="
echo "  RENEW — Pretraining"
echo "  Environment: ${ENV} (${MAZE_SIZE}x${MAZE_SIZE})"
echo "  Seeds: ${SEEDS[@]}"
echo "  Pretrain steps: ${PRETRAIN_STEPS}"
echo "  Dataset size: ${DATASET_SIZE}"
echo "  Ensemble size: ${ENSEMBLE_SIZE}"
echo "  Results: ${RESULTS_BASE}/"
echo "=============================================="

TOTAL_START=$(date +%s)

for SEED in "${SEEDS[@]}"; do
    SEED_DIR="${RESULTS_BASE}/seed_${SEED}"
    mkdir -p "${SEED_DIR}"

    echo ""
    echo "====== SEED ${SEED} ======"
    SEED_START=$(date +%s)

    # Check if already done (ensemble or single)
    if [ "${ENSEMBLE_SIZE}" -gt 1 ]; then
        CKPT="${SEED_DIR}/pretrained_0.pkl"
    else
        CKPT="${SEED_DIR}/pretrained.pkl"
    fi

    if [ -f "${CKPT}" ]; then
        echo "  Pretrained checkpoint exists, skipping."
    else
        echo "  Pretraining (ensemble_size=${ENSEMBLE_SIZE})..."
        python renew/pretrain.py \
            --seed "${SEED}" \
            --maze-size "${MAZE_SIZE}" \
            --pretrain-steps "${PRETRAIN_STEPS}" \
            --offline-dataset-size "${DATASET_SIZE}" \
            --ensemble-size "${ENSEMBLE_SIZE}" \
            --results-dir "${SEED_DIR}"
    fi

    SEED_END=$(date +%s)
    echo "  Seed ${SEED} done in $((SEED_END - SEED_START))s"
done

TOTAL_END=$(date +%s)
echo ""
echo "=============================================="
echo "  All ${#SEEDS[@]} seeds pretrained in $((TOTAL_END - TOTAL_START))s"
echo "  Results: ${RESULTS_BASE}/"
echo ""
echo "  Next: bash renew/run_finetune.sh"
echo "=============================================="