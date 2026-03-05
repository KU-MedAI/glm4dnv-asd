#!/usr/bin/env bash
set -euo pipefail

export WANDB_ENTITY="${WANDB_ENTITY:-autism}"
export WANDB_PROJECT="${WANDB_PROJECT:-evo2_ft_bend}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"

: "${RUN_ROOT:=/path/to/experiments/evo2/ncre}"
export RUN_ROOT


BASE_DIR="/fine-tuned/Evo-2/regression"

DATA_PATH="/path/to/data/ncre_annot.parquet"
SPAN_BP="${SPAN_BP:-100}"
SEQ_COL="${SEQ_COL:-NCREs_seq}"
LABEL_COL="${LABEL_COL:-activity_score}"
SPLIT_COL="${SPLIT_COL:-split}"

PY="${BASE_DIR}/ft_evo2_regression.py"

CHECKPOINT="${CHECKPOINT:-evo2_7b_base}"
EMB_LAYER="${EMB_LAYER:-blocks.31}"
POOLING="${POOLING:-last}"
MAX_LENGTH="${MAX_LENGTH:-0}"
MAX_LENGTH_CAP="${MAX_LENGTH_CAP:-0}"

FT_OUTPUT_ROOT="${RUN_ROOT}/ft/output"
FT_LOG_ROOT="${RUN_ROOT}/ft/logs"
mkdir -p "${FT_OUTPUT_ROOT}" "${FT_LOG_ROOT}"

WANDB_BASE="${RUN_ROOT}/ft/wandb"
mkdir -p "${WANDB_BASE}"
export WANDB_DIR="${WANDB_BASE}"

GPU_IDS=(1)
SEED="${SEED:-42}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
EVAL_BS="${EVAL_BS:-64}"
LR_SCHED="${LR_SCHED:-linear}"
BF16="${BF16:-1}"
EARLY_STOP="${EARLY_STOP:-1}"
EARLY_PATIENCE="${EARLY_PATIENCE:-5}"
EARLY_THRESH="${EARLY_THRESH:-0.0}"
LOAD_BEST="${LOAD_BEST:-1}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
USE_LORA="${USE_LORA:-1}"
LORA_TARGETS="${LORA_TARGETS:-Wqkv,out_proj,out_filter_dense}"
SAVE_ADAPTER_ONLY="${SAVE_ADAPTER_ONLY:-1}"
LORA_GRADUAL="${LORA_GRADUAL:-0}"
LORA_GRADUAL_INIT_K="${LORA_GRADUAL_INIT_K:-1}"
LORA_GRADUAL_STEP="${LORA_GRADUAL_STEP:-1}"
LORA_GRADUAL_EVERY="${LORA_GRADUAL_EVERY:-1}"

PYTHON_BIN="${PYTHON_BIN:-$(which python)}"
export PYTHON_BIN
cd "${BASE_DIR}"

AGENT_RUNNER="${RUN_ROOT}/ft/agent_runner.sh"
export AGENT_RUNNER

cat > "${AGENT_RUNNER}" <<'RUNNER_SH'
#!/usr/bin/env bash
set -euo pipefail
PY="${PYTHON_BIN:-python}"
LR=""; WD=""; BS=""; R=""; DP=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --learning_rate=*) LR="${1#*=}"; shift ;;
    --weight_decay=*) WD="${1#*=}"; shift ;;
    --per_device_train_batch_size=*) BS="${1#*=}"; shift ;;
    --lora_r=*) R="${1#*=}"; shift ;;
    --lora_dropout=*) DP="${1#*=}"; shift ;;
    *) shift ;;
  esac
done
R_INT="$(printf "%.0f" "${R}")"
ALPHA=$((R_INT * 2))
RID="${WANDB_RUN_ID:-local}"
RUN_NAME="${WANDB_RUN_NAME:-${RID}}"
OUT_SUB="NA"; LOG_SUB="NA"

CMD=( "$PY" "${PY}"
  --data_path "${DATA_PATH}"
  --span_bp "${SPAN_BP}"
  --seq_col "${SEQ_COL}"
  --label_col "${LABEL_COL}"
  --split_col "${SPLIT_COL}"
  --checkpoint "${CHECKPOINT}"
  --emb_layer "${EMB_LAYER}"
  --pooling "${POOLING}"
  --output_root "${FT_OUTPUT_ROOT}"
  --output_subdir "${OUT_SUB}"
  --log_root "${FT_LOG_ROOT}"
  --log_subdir "${LOG_SUB}"
  --num_train_epochs "${NUM_EPOCHS}"
  --per_device_train_batch_size "${BS}"
  --per_device_eval_batch_size "${EVAL_BS}"
  --learning_rate "${LR}"
  --weight_decay "${WD}"
  --lr_scheduler_type "${LR_SCHED}"
  --warmup_steps "${WARMUP_STEPS}"
  --seed "${SEED}"
  --grad_clip "${GRAD_CLIP}"
  --max_length "${MAX_LENGTH}"
  --max_length_cap "${MAX_LENGTH_CAP}"
  --report_to "wandb"
  --run_name "${RUN_NAME}"
)
[[ "${BF16}" == "1" ]] && CMD+=( --bf16 --device "cuda:0" ) || CMD+=( --device "cuda:0" )
[[ "${EARLY_STOP}" == "1" ]] && CMD+=( --early_stopping --early_stopping_patience "${EARLY_PATIENCE}" --early_stopping_threshold "${EARLY_THRESH}" )
[[ "${LOAD_BEST}" == "1" ]] && CMD+=( --load_best_model_at_end )
if [[ "${USE_LORA}" == "1" ]]; then
  CMD+=( --use_lora --lora_r "${R}" --lora_alpha "${ALPHA}" --lora_dropout "${DP}" --lora_targets "${LORA_TARGETS}" )
  [[ "${LORA_GRADUAL}" == "1" ]] && CMD+=( --lora_gradual --lora_gradual_init_k "${LORA_GRADUAL_INIT_K}" --lora_gradual_step "${LORA_GRADUAL_STEP}" --lora_gradual_every "${LORA_GRADUAL_EVERY}" )
fi
[[ "${SAVE_ADAPTER_ONLY}" == "1" ]] && CMD+=( --save_adapter_only )
"${CMD[@]}"
RUNNER_SH

chmod +x "${AGENT_RUNNER}"

export PY DATA_PATH SPAN_BP SEQ_COL LABEL_COL SPLIT_COL CHECKPOINT EMB_LAYER POOLING FT_OUTPUT_ROOT FT_LOG_ROOT SEED WARMUP_STEPS NUM_EPOCHS EVAL_BS LR_SCHED BF16 EARLY_STOP EARLY_PATIENCE EARLY_THRESH LOAD_BEST GRAD_CLIP MAX_LENGTH MAX_LENGTH_CAP USE_LORA LORA_TARGETS SAVE_ADAPTER_ONLY LORA_GRADUAL LORA_GRADUAL_INIT_K LORA_GRADUAL_STEP LORA_GRADUAL_EVERY AGENT_RUNNER

SWEEP_OUT="$("${PYTHON_BIN}" - <<'PY'
import os, wandb
ENTITY = os.environ["WANDB_ENTITY"]; PROJECT = os.environ["WANDB_PROJECT"]; AGENT_RUNNER = os.environ["AGENT_RUNNER"]
cfg = {
    "method": "grid",
    "metric": {"name": "test/loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"values": [5e-5]},
        "weight_decay": {"values": [0.01]},
        "per_device_train_batch_size": {"values": [64]},
        "lora_r": {"values": [128]},
        "lora_dropout": {"values": [0.05]},
    },
    "command": ["${env}", "bash", AGENT_RUNNER, "${args}"],
}
sid = wandb.sweep(cfg, entity=ENTITY, project=PROJECT)
print(f"{ENTITY}/{PROJECT}/{sid}")
PY
)"

SWEEP_PATH="$(echo "${SWEEP_OUT}" | grep -Eo '[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/[A-Za-z0-9]+' | tail -n 1)"

PIDS=()
for gpu_id in "${GPU_IDS[@]}"; do
(
  export CUDA_VISIBLE_DEVICES="${gpu_id}"
  export WANDB_DIR="${WANDB_BASE}/gpu${gpu_id}"
  mkdir -p "${WANDB_DIR}"
  wandb agent "${SWEEP_PATH}"
) &
PIDS+=($!)
sleep 0.5
done

for pid in "${PIDS[@]}"; do wait "$pid"; done

BEST_TXT="${FT_OUTPUT_ROOT}/best_run.txt"
BEST_RUN_PATH="$("${PYTHON_BIN}" - <<'PY'
import os, glob, pandas as pd, math
out_root = os.environ["FT_OUTPUT_ROOT"]
csvs = glob.glob(os.path.join(out_root, "*", "test_metrics.csv"))
best_path = None; best_score = None
for csv_path in sorted(csvs):
    run_dir = os.path.dirname(csv_path)
    try:
        df = pd.read_csv(csv_path)
        col = [c for c in df.columns if "loss" in c.lower() and "test" in c.lower()][0]
        s = float(df.loc[0, col])
        if math.isnan(s) or math.isinf(s): continue
        if (best_score is None) or (s < best_score):
            best_score = s; best_path = run_dir
    except: continue
if best_path: print(best_path)
PY
)"

if [[ -n "${BEST_RUN_PATH}" && -d "${BEST_RUN_PATH}" ]]; then
  echo "${BEST_RUN_PATH}" > "${BEST_TXT}"
fi