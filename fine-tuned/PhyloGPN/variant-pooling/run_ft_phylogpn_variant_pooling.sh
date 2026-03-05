#!/usr/bin/env bash
set -euo pipefail

# RUN ROOT
: "${RUN_ROOT:=/path/to/experiments/phylogpn/bend}"

: "${POOL_SCRIPT_DIR:=/fine-tuned/PhyloGPN/variant-pooling}"
POOL_PY="${POOL_SCRIPT_DIR}/ft_phylogpn_variant_pooling.py"

FT_OUTPUT_ROOT="${RUN_ROOT}/ft/output"
BEST_FILE="${FT_OUTPUT_ROOT}/best_run.txt"

BP=200

VARIANT_LIST="/path/to/data/variant_list.feather"
DNV="/path/to/data/variant.feather"
BASE="songlab/PhyloGPN"
MERGE_LORA=0

REVERSE_MODE="forward"
POOLING="max"

OUT_DIR="${RUN_ROOT}/mut_pool/output_best"
LOG_DIR="${RUN_ROOT}/mut_pool/logs_best"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

GPUS=(0)
NUM_GPUS=${#GPUS[@]}
NUM_SPLITS=${NUM_GPUS}

BATCH_SIZE=3000

SKIP_IF_EXISTS=0

echo "RUN_ROOT        = ${RUN_ROOT}"
echo "FT_OUTPUT_ROOT  = ${FT_OUTPUT_ROOT}"
echo "BEST_FILE       = ${BEST_FILE}"
echo "POOL_PY         = ${POOL_PY}"
echo "BP              = ${BP}"
echo "BASE            = ${BASE}"
echo "REVERSE_MODE    = ${REVERSE_MODE}"
echo "POOLING         = ${POOLING}"
echo "GPUS            = ${GPUS[*]}"
echo "BATCH_SIZE      = ${BATCH_SIZE}"
echo

if [[ ! -f "${POOL_PY}" ]]; then
  echo "POOL_PY not found: ${POOL_PY}"
  exit 1
fi
if [[ ! -f "${VARIANT_LIST}" ]]; then
  echo "VARIANT_LIST not found: ${VARIANT_LIST}"
  exit 1
fi
if [[ ! -f "${DNV}" ]]; then
  echo "DNV not found: ${DNV}"
  exit 1
fi

if [[ ! -f "${BEST_FILE}" ]]; then
  echo "best_run.txt not found: ${BEST_FILE}"
  exit 1
fi

BEST_RUN_DIR="$(cat "${BEST_FILE}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

if [[ "${BEST_RUN_DIR}" != /* ]]; then
  BEST_RUN_DIR="${FT_OUTPUT_ROOT}/${BEST_RUN_DIR}"
fi

if [[ ! -d "${BEST_RUN_DIR}" ]]; then
  echo "BEST_RUN_DIR not found: ${BEST_RUN_DIR}"
  exit 1
fi

RUN_NAME="$(basename "${BEST_RUN_DIR}")"
ADIR="${BEST_RUN_DIR}/adapter_best"
PTH="${BEST_RUN_DIR}/best.pth"

echo "BEST_RUN_DIR = ${BEST_RUN_DIR}"
echo "RUN_NAME     = ${RUN_NAME}"
echo

FINAL_OUT="${OUT_DIR}/${RUN_NAME}_bp${BP}_mutmax_fwdrev_concat_split${NUM_SPLITS}.feather"
MERGE_LOG="${LOG_DIR}/merge_${RUN_NAME}.log"

if [[ "${SKIP_IF_EXISTS}" == "1" && -f "${FINAL_OUT}" ]]; then
  echo "Final output exists: ${FINAL_OUT}"
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
    --base_ckpt "${BASE}"
    --device "cuda:0"
    --batch_size "${BATCH_SIZE}"
    --reverse_mode "${REVERSE_MODE}"
    --pooling "${POOLING}"
    --out_path "${OUT_PART}"
    --data_split "${i}"
    --num_splits "${NUM_SPLITS}"
  )

  if [[ -d "${ADIR}" ]]; then
    echo "LoRA adapter (split ${i})" >> "${LOG_PART}"
    POOL_ARGS+=( --lora_adapter_dir "${ADIR}" )
    if [[ "${MERGE_LORA}" == "1" ]]; then
      POOL_ARGS+=( --merge_lora )
    fi
  elif [[ -f "${PTH}" ]]; then
    echo "Full FT checkpoint (split ${i})" >> "${LOG_PART}"
    POOL_ARGS+=( --ft_pth_path "${PTH}" )
  else
    echo "Neither adapter_best nor best.pth found under: ${BEST_RUN_DIR}"
    exit 1
  fi

  if [[ ${NUM_GPUS} -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${POOL_PY}" "${POOL_ARGS[@]}" 2>&1 | tee "${LOG_PART}" &
    PIDS+=($!)
  else
    if [[ $i -eq 0 ]]; then
      CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${POOL_PY}" "${POOL_ARGS[@]}" 2>&1 | tee "${LOG_PART}" &
    else
      CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${POOL_PY}" "${POOL_ARGS[@]}" > "${LOG_PART}" 2>&1 &
    fi
    PIDS+=($!)
  fi

  if [[ $i -lt $((NUM_GPUS - 1)) ]]; then
    sleep 3
  fi
done

for idx in "${!PIDS[@]}"; do
  PID="${PIDS[$idx]}"
  GPU_ID="${GPUS[$idx]}"
  if wait "${PID}"; then
    echo "DONE GPU ${GPU_ID} split ${idx}"
  else
    echo "FAIL GPU ${GPU_ID} split ${idx} (PID=${PID})"
    exit 1
  fi
done

python - <<PYEOF > "${MERGE_LOG}" 2>&1
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

dfs = [pd.read_feather(p) for p in parts]
merged = pd.concat(dfs, axis=0, ignore_index=True)

os.makedirs(os.path.dirname(final_out), exist_ok=True)
merged.to_feather(final_out)

print("merged saved:", final_out)
print("parts:", len(parts))
print("shape:", merged.shape)
print("cols :", list(merged.columns))
PYEOF

rm -rf "${TEMP_DIR}"

echo
echo "Parallel split pooling DONE"
echo "Best run : ${BEST_RUN_DIR}"
echo "Output   : ${FINAL_OUT}"
echo "Logs     : ${LOG_DIR}"