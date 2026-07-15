#!/usr/bin/env bash
# InsightBench batch evaluation script.
#
# Iterates over all (object, asset, task) combinations and calls
# scripts/evaluate.py for each. Jobs are distributed round-robin across
# the specified GPUs; each GPU runs its queue sequentially.
#
# Usage (single GPU):
#   ./scripts/eval_batch.sh --config configs/eval/pi0.yaml --gpu 0
#
# Usage (multi-GPU):
#   ./scripts/eval_batch.sh --config configs/eval/pi0.yaml --gpus 0,1,2,3
#
# Options:
#   --config   <path>   eval YAML config            (default: configs/eval/pi0.yaml)
#   --num_envs <n>      parallel envs per run        (default: 8)
#   --gpu      <id>     single GPU id                (default: 0)
#   --gpus     <ids>    comma-separated GPU ids      (overrides --gpu)
#   --log_dir  <path>   per-run log directory        (default: ./outputs/eval_logs)
#   --override <k=v>    OmegaConf override; repeat for multiple values
#   --pos_rand          enable object position randomization on each reset (default: off)
#
# All output is logged per-run to $LOG_DIR/<object>_<asset>_task<n>.log
# Failed runs are collected into $LOG_DIR/../retry_failed.sh

set -uo pipefail

# ─── Defaults ─────────────────────────────────────────────────────────────────
CONFIG="configs/eval/pi0.yaml"
NUM_ENVS=8
GPUS="0"
LOG_DIR="./outputs/eval_logs"
POS_RAND=""
OVERRIDES=()

# ─── Parse flags ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)   CONFIG="$2";      shift 2 ;;
        --num_envs) NUM_ENVS="$2";    shift 2 ;;
        --gpu)      GPUS="$2";        shift 2 ;;
        --gpus)     GPUS="$2";        shift 2 ;;
        --log_dir)  LOG_DIR="$2";     shift 2 ;;
        --override) OVERRIDES+=("$2"); shift 2 ;;
        --pos_rand) POS_RAND="--pos_rand"; shift 1 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

IFS=',' read -ra GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}

RETRY_SCRIPT="$(dirname "$LOG_DIR")/retry_failed.sh"
mkdir -p "$LOG_DIR"
echo "#!/usr/bin/env bash" > "$RETRY_SCRIPT"
chmod +x "$RETRY_SCRIPT"

# ─── Build full job list from configs/task/*.yaml ─────────────────────────────
JOB_OBJECTS=()
JOB_ASSETS=()
JOB_TASKS=()

JOB_GENERATOR="$(dirname "${BASH_SOURCE[0]}")/generate_eval_jobs.py"
JOB_FILE="$(mktemp)"
trap 'rm -f "$JOB_FILE"' EXIT

if ! python3 "$JOB_GENERATOR" > "$JOB_FILE"; then
    echo "Failed to generate eval jobs from configs/task/*.yaml" >&2
    exit 1
fi

while IFS=$'\t' read -r OBJECT ASSET TASK; do
    [[ -z "$OBJECT" ]] && continue
    JOB_OBJECTS+=("$OBJECT")
    JOB_ASSETS+=("$ASSET")
    JOB_TASKS+=("$TASK")
done < "$JOB_FILE"

TOTAL=${#JOB_OBJECTS[@]}
echo "Total jobs : $TOTAL"
echo "GPUs       : ${GPU_LIST[*]}"
echo "Jobs/GPU   : $(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS )) (approx)"
echo "Overrides  : ${#OVERRIDES[@]}"
echo ""

# ─── Per-GPU worker function ───────────────────────────────────────────────────
run_gpu_worker() {
    local GPU_ID="$1"
    local WORKER_IDX="$2"   # 0-based index in GPU_LIST
    local FAILED_LOG="$LOG_DIR/.failed_gpu${GPU_ID}.txt"
    : > "$FAILED_LOG"

    local job_idx=$WORKER_IDX
    while (( job_idx < TOTAL )); do
        local OBJECT="${JOB_OBJECTS[$job_idx]}"
        local ASSET="${JOB_ASSETS[$job_idx]}"
        local TASK="${JOB_TASKS[$job_idx]}"
        local LOG="$LOG_DIR/${OBJECT}_${ASSET}_task${TASK}.log"

        echo "[GPU$GPU_ID] ($((job_idx+1))/$TOTAL) $OBJECT / $ASSET / task$TASK"

        local CMD=(
            env
            "PYTHONUNBUFFERED=1"
            "CUDA_VISIBLE_DEVICES=$GPU_ID"
            python scripts/evaluate.py
            --config "$CONFIG"
            --object "$OBJECT"
            --asset_path "$ASSET"
            --task_idx "$TASK"
            --num_envs "$NUM_ENVS"
        )
        if [[ -n "$POS_RAND" ]]; then
            CMD+=("$POS_RAND")
        fi
        CMD+=(--enable_cameras --headless)
        CMD+=("${OVERRIDES[@]}")

        if "${CMD[@]}" > "$LOG" 2>&1; then
            echo "[GPU$GPU_ID]   OK"
        else
            echo "[GPU$GPU_ID]   FAILED — $LOG"
            echo "# $OBJECT $ASSET task$TASK" >> "$FAILED_LOG"
            printf '%q ' "${CMD[@]}" >> "$FAILED_LOG"
            printf '\n' >> "$FAILED_LOG"
        fi

        job_idx=$(( job_idx + NUM_GPUS ))
    done
}

# ─── Launch one worker per GPU ─────────────────────────────────────────────────
PIDS=()
for (( i=0; i<NUM_GPUS; i++ )); do
    GPU_ID="${GPU_LIST[$i]}"
    run_gpu_worker "$GPU_ID" "$i" &
    PIDS+=($!)
done

# ─── Wait for all workers ──────────────────────────────────────────────────────
FAILED=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || FAILED=$(( FAILED + 1 ))
done

# ─── Collect failed runs ───────────────────────────────────────────────────────
for GPU_ID in "${GPU_LIST[@]}"; do
    FAILED_LOG="$LOG_DIR/.failed_gpu${GPU_ID}.txt"
    if [[ -s "$FAILED_LOG" ]]; then
        cat "$FAILED_LOG" >> "$RETRY_SCRIPT"
    fi
    rm -f "$FAILED_LOG"
done

echo ""
echo "Done. Total: $TOTAL  Worker failures: $FAILED"
[[ -s "$RETRY_SCRIPT" ]] && echo "Retry failed runs: bash $RETRY_SCRIPT"
