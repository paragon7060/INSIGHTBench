#!/usr/bin/env bash
# InsightBench batch data collection script.
#
# Iterates over all (object, asset, scene_key) combinations under Assets/TrainSuite/
# and calls scripts/collect_demo.py for each. Jobs are distributed round-robin
# across the specified GPUs; each GPU runs its queue sequentially.
#
# Usage (single GPU):
#   ./scripts/collect_batch.sh --dataset_name YOUR_HF_USERNAME/INSIGHT-demo --gpu 0
#
# Usage (multi-GPU):
#   ./scripts/collect_batch.sh --dataset_name YOUR_HF_USERNAME/INSIGHT-demo --gpus 0,1,2,3
#
# Options:
#   --dataset_name  <name>    HuggingFace dataset repo (required)
#   --target_episodes <n>     Successful episodes per (asset, scene_key)  (default: 100)
#   --num_envs      <n>       Parallel envs per run                        (default: 8)
#   --gpu           <id>      Single GPU id                                (default: 0)
#   --gpus          <ids>     Comma-separated GPU ids                      (overrides --gpu)
#   --no_guide                Collect without guide asset in scene
#   --pos_rand                Enable object base position randomization     (default)
#   --no_pos_rand             Disable object base position randomization
#   --log_dir       <path>    Per-run log directory  (default: ./outputs/collect_logs)
#   --asset_dir     <path>    Override TrainSuite root  (default: ./Assets/TrainSuite)
#
# Per-run logs: $LOG_DIR/<object>_<asset>_<scene_key>.log
# Failed runs:  $LOG_DIR/../collect_retry_failed.sh

set -uo pipefail

# ─── Defaults ─────────────────────────────────────────────────────────────────
DATASET_NAME=""
TARGET_EPISODES=100
NUM_ENVS=8
GPUS="0"
NO_GUIDE=""
POS_RAND="--pos_rand"
LOG_DIR="./outputs/collect_logs"
ASSET_DIR="./Assets/TrainSuite"

# ─── Parse flags ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_name)     DATASET_NAME="$2";     shift 2 ;;
        --target_episodes)  TARGET_EPISODES="$2";  shift 2 ;;
        --num_envs)         NUM_ENVS="$2";         shift 2 ;;
        --gpu)              GPUS="$2";             shift 2 ;;
        --gpus)             GPUS="$2";             shift 2 ;;
        --no_guide)         NO_GUIDE="--no_guide"; shift 1 ;;
        --pos_rand)         POS_RAND="--pos_rand"; shift 1 ;;
        --no_pos_rand|--fixed_pos) POS_RAND="--no_pos_rand"; shift 1 ;;
        --log_dir)          LOG_DIR="$2";          shift 2 ;;
        --asset_dir)        ASSET_DIR="$2";        shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$DATASET_NAME" ]]; then
    echo "Error: --dataset_name is required." >&2
    echo "  e.g. ./scripts/collect_batch.sh --dataset_name paragon7060/INSIGHT-demo --gpus 0,1,2,3" >&2
    exit 1
fi

IFS=',' read -ra GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}

RETRY_SCRIPT="$(dirname "$LOG_DIR")/collect_retry_failed.sh"
mkdir -p "$LOG_DIR"
echo "#!/usr/bin/env bash" > "$RETRY_SCRIPT"
chmod +x "$RETRY_SCRIPT"

# ─── Scene key definitions ─────────────────────────────────────────────────────
# cabinet: single scene_key per asset (drawer variants are asset-specific)
CABINET_KEYS=("1ext")
DOOR_KEYS=("3a" "3b" "3c" "3d")
BOTTLE_KEYS=("5a" "5b" "5c" "5d" "5e" "5f" "5g" "5h")

# ─── Build full job list from TrainSuite directories ───────────────────────────
JOB_OBJECTS=()
JOB_ASSETS=()
JOB_KEYS=()

CABINET_DIR="$ASSET_DIR/cabinet_suite"
DOOR_DIR="$ASSET_DIR/door_suite"
BOTTLE_DIR="$ASSET_DIR/bottle_suite"

for DIR in "$CABINET_DIR" "$DOOR_DIR" "$BOTTLE_DIR"; do
    if [[ ! -d "$DIR" ]]; then
        echo "Warning: $DIR not found, skipping." >&2
    fi
done

if [[ -d "$CABINET_DIR" ]]; then
    for ASSET in $(ls "$CABINET_DIR"); do
        [[ -d "$CABINET_DIR/$ASSET" ]] || continue
        for KEY in "${CABINET_KEYS[@]}"; do
            JOB_OBJECTS+=("cabinet"); JOB_ASSETS+=("$ASSET"); JOB_KEYS+=("$KEY")
        done
    done
fi

if [[ -d "$DOOR_DIR" ]]; then
    for ASSET in $(ls "$DOOR_DIR"); do
        [[ -d "$DOOR_DIR/$ASSET" ]] || continue
        for KEY in "${DOOR_KEYS[@]}"; do
            JOB_OBJECTS+=("door"); JOB_ASSETS+=("$ASSET"); JOB_KEYS+=("$KEY")
        done
    done
fi

if [[ -d "$BOTTLE_DIR" ]]; then
    for ASSET in $(ls "$BOTTLE_DIR"); do
        [[ -d "$BOTTLE_DIR/$ASSET" ]] || continue
        for KEY in "${BOTTLE_KEYS[@]}"; do
            JOB_OBJECTS+=("bottle"); JOB_ASSETS+=("$ASSET"); JOB_KEYS+=("$KEY")
        done
    done
fi

TOTAL=${#JOB_OBJECTS[@]}
echo "Total jobs : $TOTAL"
echo "GPUs       : ${GPU_LIST[*]}"
echo "Jobs/GPU   : $(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS )) (approx)"
echo "Dataset    : $DATASET_NAME"
echo "Target eps : $TARGET_EPISODES per job"
echo "Pos rand   : $POS_RAND"
echo ""

# ─── Per-GPU worker function ───────────────────────────────────────────────────
run_gpu_worker() {
    local GPU_ID="$1"
    local WORKER_IDX="$2"
    local FAILED_LOG="$LOG_DIR/.failed_collect_gpu${GPU_ID}.txt"
    : > "$FAILED_LOG"

    local job_idx=$WORKER_IDX
    while (( job_idx < TOTAL )); do
        local OBJECT="${JOB_OBJECTS[$job_idx]}"
        local ASSET="${JOB_ASSETS[$job_idx]}"
        local KEY="${JOB_KEYS[$job_idx]}"
        local OBJECT_ASSET_DIR="$ASSET_DIR/${OBJECT}_suite"
        local LOG="$LOG_DIR/${OBJECT}_${ASSET}_${KEY}.log"

        echo "[GPU$GPU_ID] ($((job_idx+1))/$TOTAL) $OBJECT / $ASSET / $KEY"

        CMD="PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/collect_demo.py \
            --object $OBJECT \
            --asset_id $ASSET \
            --scene_key $KEY \
            --dataset_name $DATASET_NAME \
            --target_episodes $TARGET_EPISODES \
            --num_envs $NUM_ENVS \
            --asset_dir $OBJECT_ASSET_DIR \
            $NO_GUIDE \
            $POS_RAND \
            --enable_cameras --headless"

        if eval "$CMD" > "$LOG" 2>&1; then
            echo "[GPU$GPU_ID]   OK"
        else
            echo "[GPU$GPU_ID]   FAILED — $LOG"
            echo "# $OBJECT $ASSET $KEY" >> "$FAILED_LOG"
            echo "$CMD" >> "$FAILED_LOG"
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
FAILED_WORKERS=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || FAILED_WORKERS=$(( FAILED_WORKERS + 1 ))
done

# ─── Collect failed runs ───────────────────────────────────────────────────────
for GPU_ID in "${GPU_LIST[@]}"; do
    FAILED_LOG="$LOG_DIR/.failed_collect_gpu${GPU_ID}.txt"
    if [[ -s "$FAILED_LOG" ]]; then
        cat "$FAILED_LOG" >> "$RETRY_SCRIPT"
    fi
    rm -f "$FAILED_LOG"
done

echo ""
echo "Done. Total: $TOTAL  Worker failures: $FAILED_WORKERS"
[[ -s "$RETRY_SCRIPT" ]] && echo "Retry failed runs: bash $RETRY_SCRIPT"
