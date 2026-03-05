
#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ROOT:=/path/to/experiments/dnabert/bend}"
POOL_SCRIPT_DIR=/fine-tuned/DNABERT/variant-pooling
POOL_PY="${POOL_SCRIPT_DIR}/ft_dnabert_variant_pooling.py"

FT_OUTPUT_ROOT="${RUN_ROOT}/ft/output"
BEST_FILE="${FT_OUTPUT_ROOT}/best_run.txt"

BP=50
VARIANT_LIST=/path/to/data/variant_list.feather
DNV=/path/to/data/variant.feather
BASE="zhihan1996/DNA_bert_6"
MERGE_LORA=0

REVERSE_MODE="forward" 
POOLING="max"

OUT_DIR="${RUN_ROOT}/mut_pool_ref/output_best"
LOG_DIR="${RUN_ROOT}/mut_pool_ref/logs_best"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

GPUS=(0)  
NUM_GPUS=${#GPUS[@]}


echo "============================================================"
echo "[INFO] RUN_ROOT        = ${RUN_ROOT}"
echo "[INFO] FT_OUTPUT_ROOT  = ${FT_OUTPUT_ROOT}"
echo "[INFO] GPUs            = ${GPUS[@]}"
echo "[INFO] NUM_GPUS        = ${NUM_GPUS}"
echo "[INFO] BP              = ${BP}"
echo "[INFO] REVERSE_MODE    = ${REVERSE_MODE}"
echo "[INFO] POOLING         = ${POOLING}"
echo "============================================================"
echo

if [[ ! -f "${BEST_FILE}" ]]; then
  echo "[ERROR] best_run.txt not found: ${BEST_FILE}"
  echo "You must run FT sweep + pick_best.py (AUROC) before pooling."
  exit 1
fi

BEST_RUN_DIR="$(cat "${BEST_FILE}")"
BEST_RUN_DIR="$(echo "${BEST_RUN_DIR}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

if [[ "${BEST_RUN_DIR}" != /* ]]; then
  BEST_RUN_DIR="${FT_OUTPUT_ROOT}/${BEST_RUN_DIR}"
fi

if [[ ! -d "${BEST_RUN_DIR}" ]]; then
  echo "[ERROR] BEST_RUN_DIR not found: ${BEST_RUN_DIR}"
  exit 1
fi

echo "[INFO] BEST_RUN_DIR = ${BEST_RUN_DIR}"
echo

RUN_NAME="$(basename "${BEST_RUN_DIR}")"
ADIR="${BEST_RUN_DIR}/adapter_best"
PTH="${BEST_RUN_DIR}/best.pth"

TEMP_DIR="${OUT_DIR}/temp_splits"
mkdir -p "${TEMP_DIR}"

echo "============================================================"
echo "[INFO] Starting parallel GPU pooling with ${NUM_GPUS} GPUs"
echo "============================================================"

PIDS=()

for i in "${!GPUS[@]}"; do
    GPU_ID=${GPUS[$i]}
    
    OUT_PART="${TEMP_DIR}/${RUN_NAME}_bp${BP}_gpu${GPU_ID}_part${i}.feather"
    LOG_PART="${LOG_DIR}/pool_${RUN_NAME}_gpu${GPU_ID}_part${i}.log"
    
    echo "[GPU ${GPU_ID}] Starting process ${i}/${NUM_GPUS}..."
    
    POOL_ARGS=(
        --bp "${BP}"
        --variant_list_path "${VARIANT_LIST}"
        --dnv_path "${DNV}"
        --base_ckpt "${BASE}"
        --device "cuda:0"
        --batch_size 1024
        --reverse_mode "${REVERSE_MODE}"
        --pooling "${POOLING}"
        --out_path "${OUT_PART}"
        --data_split "${i}"
        --num_splits "${NUM_GPUS}"
    )
    
    if [[ -d "${ADIR}" ]]; then
        POOL_ARGS+=( --lora_adapter_dir "${ADIR}" )
        if [[ "${MERGE_LORA}" == "1" ]]; then
            POOL_ARGS+=( --merge_lora )
        fi
    elif [[ -f "${PTH}" ]]; then
        POOL_ARGS+=( --ft_pth_path "${PTH}" )
    else
        echo "[ERROR] Neither adapter_best nor best.pth found"
        exit 1
    fi
    

    if [[ $i -eq 0 ]]; then
        CUDA_VISIBLE_DEVICES=${GPU_ID} python "${POOL_PY}" "${POOL_ARGS[@]}" \
            2>&1 | tee "${LOG_PART}" &
    else
        CUDA_VISIBLE_DEVICES=${GPU_ID} python "${POOL_PY}" "${POOL_ARGS[@]}" \
            > "${LOG_PART}" 2>&1 &
    fi
    
    PIDS+=($!)
    echo "[GPU ${GPU_ID}] PID: ${PIDS[$i]}"
done

echo
echo "============================================================"
echo "All ${NUM_GPUS} processes started. Waiting for completion..."
echo "============================================================"

for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    GPU_ID=${GPUS[$i]}
    wait ${PID}
    EXIT_CODE=$?
    if [[ ${EXIT_CODE} -eq 0 ]]; then
        echo "[GPU ${GPU_ID}] Completed (PID: ${PID})"
    else
        echo "[GPU ${GPU_ID}] Failed with exit code ${EXIT_CODE} (PID: ${PID})"
        exit 1
    fi
done

echo
echo "============================================================"
echo "Merging results from all GPUs..."
echo "============================================================"

FINAL_OUT="${OUT_DIR}/${RUN_NAME}_bp${BP}_mutmax_fwdrev_concat.feather"
MERGE_LOG="${LOG_DIR}/merge_${RUN_NAME}.log"

python - "${TEMP_DIR}" "${FINAL_OUT}" <<'PY' 2>&1 | tee "${MERGE_LOG}"
import pandas as pd
import glob
import sys

if len(sys.argv) < 3:
    print("Error: Missing arguments", file=sys.stderr)
    sys.exit(1)

temp_dir = sys.argv[1]
final_out = sys.argv[2]

parts = sorted(glob.glob(temp_dir + "/*.feather"))
print("Found {} parts to merge".format(len(parts)))

if not parts:
    print("Error: No feather files found in {}".format(temp_dir), file=sys.stderr)
    sys.exit(1)

dfs = [pd.read_feather(p) for p in parts]
merged = pd.concat(dfs, axis=0, ignore_index=True)

merged.to_feather(final_out)
print("Merged output saved: {}".format(final_out))
print("   Shape: {}".format(merged.shape))
print("   Columns: {}".format(list(merged.columns)))
PY

rm -rf "${TEMP_DIR}"

echo
echo "============================================================"
echo "   Parallel Pooling DONE"
echo "   Best run  : ${BEST_RUN_DIR}"
echo "   Output    : ${FINAL_OUT}"
echo "   Merge log : ${MERGE_LOG}"
echo "   GPU logs  : ${LOG_DIR}/pool_${RUN_NAME}_gpu*.log"
echo "============================================================"
```