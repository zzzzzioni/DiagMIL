#!/usr/bin/env bash
# Preprocess patch features into graph_v2, then train DiagMIL.
#
# Usage (from this directory):
#   bash main.sh
#
# Optional overrides:
#   DATASET=BRACS GPU=0 bash main.sh
#   SKIP_PREPROCESS=1 bash main.sh   # train only (graphs must exist)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONWARNINGS="ignore"
export TOKENIZERS_PARALLELISM=false

DATASET="${DATASET:-BRACS}"          # BRACS | BCNB | inhouse
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"

MAG_LOW="${MAG_LOW:-x10}"
MAG_HIGH="${MAG_HIGH:-x20}"
TILE="${TILE:-256}"
SPLITS="${SPLITS:-train,val,test}"

EPOCHS="${EPOCHS:-100}"
BATCH=1
# Default LR: BCNB 1e-3, BRACS (and others) 5e-4 - override with LR=...
if [[ "${DATASET}" == "BCNB" ]]; then
  LR="${LR:-0.001}"
else
  LR="${LR:-0.0005}"
fi
NUM_WORKERS="${NUM_WORKERS:-0}"
LAMBDA_ALIGN="${LAMBDA_ALIGN:-0.05}"
LAMBDA_H="${LAMBDA_H:-0.1}"
SEED="${SEED:-0}"

TS=$(date +%Y%m%d_%H%M%S)

declare -A NUM_CLASSES_MAP
NUM_CLASSES_MAP["BRACS"]=7
NUM_CLASSES_MAP["BCNB"]=4
# NUM_CLASSES_MAP["inhouse"]=7

declare -A PATCH_ROOT_MAP
PATCH_ROOT_MAP["BRACS"]="./data/BRACS"
PATCH_ROOT_MAP["BCNB"]="./data/BCNB"
# PATCH_ROOT_MAP["inhouse"]="./data/inhouse"

declare -A GRAPH_ROOT_MAP
GRAPH_ROOT_MAP["BRACS"]="./data/BRACS/graph_v2"
GRAPH_ROOT_MAP["BCNB"]="./data/BCNB/graph_v2"
# GRAPH_ROOT_MAP["inhouse"]="./data/inhouse/graph_v2"

if [[ -z "${NUM_CLASSES_MAP[$DATASET]+x}" ]]; then
  echo "[ERROR] Unknown DATASET=${DATASET}. Use BRACS, BCNB, or inhouse."
  exit 1
fi

NUM_CLASSES="${NUM_CLASSES_MAP[$DATASET]}"
PATCH_ROOT="${PATCH_ROOT_MAP[$DATASET]}"
GRAPH_ROOT="${GRAPH_ROOT_MAP[$DATASET]}"
EXP_NAME="${DATASET}/diagmil_${TS}_seed${SEED}_la${LAMBDA_ALIGN}_lh${LAMBDA_H}"

echo "============================================================"
echo " DiagMIL pipeline"
echo "  dataset       : ${DATASET}"
echo "  patch root    : ${PATCH_ROOT}"
echo "  graph root    : ${GRAPH_ROOT}"
echo "  splits        : ${SPLITS}"
echo "  mag           : ${MAG_LOW} <-> ${MAG_HIGH}"
echo "  lr            : ${LR}"
echo "  skip preprocess: ${SKIP_PREPROCESS}"
echo "  exp_name      : ${EXP_NAME}"
echo "============================================================"

# -----------------------------------------------------------------------------
# Step 1: graph preprocessing
# -----------------------------------------------------------------------------
if [[ "${SKIP_PREPROCESS}" == "1" ]]; then
  echo "[Step 1] Skipped (SKIP_PREPROCESS=1)"
else
  if [[ ! -d "${PATCH_ROOT}" ]]; then
    echo "[ERROR] Patch root not found: ${PATCH_ROOT}"
    echo "        Place preprocessed patch pkl files there before running."
    exit 1
  fi

  echo ""
  echo "[Step 1/2] preprocess_graph_v2.py"
  python preprocess_graph_v2.py \
    --root "${PATCH_ROOT}" \
    --tile "${TILE}" \
    --splits "${SPLITS}" \
    --mag_low "${MAG_LOW}" \
    --mag_high "${MAG_HIGH}" \
    --out_root "${GRAPH_ROOT}" \
    --min_patches_low 4 \
    --min_patches_high 4
fi

if [[ ! -d "${GRAPH_ROOT}" ]]; then
  echo "[ERROR] Graph root not found after preprocessing: ${GRAPH_ROOT}"
  exit 1
fi

# -----------------------------------------------------------------------------
# Step 2: training
# -----------------------------------------------------------------------------
echo ""
echo "[Step 2/2] train.py"
SEED="${SEED}" python train.py \
  --num_classes "${NUM_CLASSES}" \
  --graph_root "${GRAPH_ROOT}" \
  --low_mag "${MAG_LOW}" \
  --high_mag "${MAG_HIGH}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH}" \
  --lr "${LR}" \
  --deterministic \
  --num_workers "${NUM_WORKERS}" \
  --save_every 1 \
  --lambda_align "${LAMBDA_ALIGN}" \
  --lambda_H "${LAMBDA_H}" \
  --exp_name "${EXP_NAME}"

echo ""
echo "Done. Results under: results/${EXP_NAME}/"
