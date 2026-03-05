#!/usr/bin/env bash
set -euo pipefail

# RUN ROOT
: "${RUN_ROOT:=/path/to/experiments/evo2/bend}"

: "${POOL_SCRIPT_DIR:=/fine-tuned/Evo-2/variant-pooling}"
POOL_PY="${POOL_SCRIPT_DIR}/ft_evo2_variant_pooling.py"


FT_OUTPUT_ROOT="${RUN_ROOT}/ft/output"
BEST_FILE="${FT_OUTPUT_ROOT}/best_run.txt"

BP=100
VARIANT_LIST="/path/to/data/variant_list.feather"
DNV="/path/to/data/variant.feather"

MODEL_NAME="evo2_7b"
LAYER_NAME="blocks.26.mlp.l3"

REVERSE_MODE="reverse"

OUT_DIR="${RUN_ROOT}/mut_pool/output_best"
LOG_DIR="${RUN_ROOT}/mut_pool/logs_best"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

GPUS=(0)
NUM_SPLITS="${#GPUS[@]}"

BATCH_SIZE=400

SKIP_IF_EXISTS=0

if [[ ! -f "${BEST_FILE}" ]]; then
  exit 1
fi

BEST_RUN_DIR="$(cat "${BEST_FILE}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
if [[ "${BEST_RUN_DIR}" != /* ]]; then
  BEST_RUN_DIR="${FT_OUTPUT_ROOT}/${BEST_RUN_DIR}"
fi

if [[ ! -d "${BEST_RUN_DIR}" ]]; then
  exit 1
fi

RUN_NAME="$(basename "${BEST_RUN_DIR}")"
PTH="${BEST_RUN_DIR}/best.pth"

FINAL_OUT="${OUT_DIR}/${RUN_NAME}_evo2_${MODEL_NAME}_bp${BP}_VARONLY_mutidxMAX_fwdrevCONCAT_split${NUM_SPLITS}.feather"
MERGE_LOG="${LOG_DIR}/merge_${RUN_NAME}.log"

if [[ "${SKIP_IF_EXISTS}" == "1" && -f "${FINAL_OUT}" ]]; then
  exit 0
fi

TEMP_DIR="${OUT_DIR}/temp_splits_${RUN_NAME}_bp${BP}"
rm -rf "${TEMP_DIR}"
mkdir -p "${TEMP_DIR}"

PIDS=()

for i in "${!GPUS[@]}"; do
  GPU_ID="${GPUS[$i]}"

  OUT_PART="${TEMP_DIR}/${RUN_NAME}_bp${BP}_split${i}of${NUM_SPLITS}.feather"
  LOG_PART="${LOG_DIR}/pool_${RUN_NAME}_gpu${GPU_ID}_split${i}of${NUM_SPLITS}.log"

  POOL_ARGS=(
    --bp "${BP}"
    --variant_list_path "${VARIANT_LIST}"
    --dnv_path "${DNV}"
    --model_name "${MODEL_NAME}"
    --layer_name "${LAYER_NAME}"
    --device "cuda:0"
    --batch_size "${BATCH_SIZE}"
    --reverse_mode "${REVERSE_MODE}"
    --out_path "${OUT_PART}"
    --data_split "${i}"
    --num_splits "${NUM_SPLITS}"
    --verify_weights 1
    --verify_n 2
  )

  if [[ -f "${PTH}" ]]; then
    POOL_ARGS+=( --ft_pth_path "${PTH}" )
  else
    exit 1
  fi

  (
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    python "${POOL_PY}" "${POOL_ARGS[@]}"
  ) >> "${LOG_PART}" 2>&1 &

  PIDS+=("$!")
done

for idx in "${!PIDS[@]}"; do
  PID="${PIDS[$idx]}"
  GPU_ID="${GPUS[$idx]}"
  if ! wait "${PID}"; then
    exit 1
  fi
done

python - <<EOF > "${MERGE_LOG}" 2>&1
import os
import pandas as pd

temp_dir = "${TEMP_DIR}"
run_name = "${RUN_NAME}"
bp = ${BP}
num_splits = ${NUM_SPLITS}
final_out = "${FINAL_OUT}"

parts = []
for i in range(num_splits):
    p = os.path.join(temp_dir, f"{run_name}_bp{bp}_split{i}of{num_splits}.feather")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing part: {p}")
    parts.append(p)

dfs = []
for p in parts:
    df = pd.read_feather(p)
    dfs.append(df)

merged = pd.concat(dfs, axis=0, ignore_index=True)
os.makedirs(os.path.dirname(final_out), exist_ok=True)
merged.to_feather(final_out)
EOF

rm -rf "${TEMP_DIR}"