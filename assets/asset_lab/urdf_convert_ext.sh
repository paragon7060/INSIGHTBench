#!/bin/bash

# ===================================================================
# 0. 인자 확인
# ===================================================================
# 스크립트 실행 시 인자가 1개 미만이면 사용법을 안내하고 종료합니다.
if [ $# -lt 1 ]; then
    echo "Usage: $0 [bottle|door|drawer]"
    exit 1
fi

COMMAND=$1

# ===================================================================
# 1. 명령어에 따른 설정 변경
# ===================================================================
CONVEX_DECOMPOSE_FLAG=""
# 입력된 명령어(COMMAND)에 따라 기본 경로(BASE_DIRS),
# 찾을 폴더 패턴(SEARCH_PATTERN), 변환할 URDF 파일 목록(TARGET_URDF_FILENAMES)을 설정합니다.
case "$COMMAND" in
    bottle)
        BASE_DIRS=("./Assets/AdaManip/bottle")
        SEARCH_PATTERN="*" # 모든 하위 폴더를 대상으로 함
        TARGET_URDF_FILENAMES=("mobility.urdf" "mobility_reversed.urdf")
        ;;
    door)
        BASE_DIRS=(
            "./Assets/AdaManip/door"
            "./Assets/UniDoorManip/Datasets/LeverDoor"  # 예시: 두 번째 door 에셋 경로
        )
        SEARCH_PATTERN="*" # 모든 하위 폴더를 대상으로 함
        TARGET_URDF_FILENAMES=("mobility_pull_cw.urdf" "mobility_pull_ccw.urdf" "mobility_push_cw.urdf" "mobility_push_ccw.urdf")
        CONVEX_DECOMPOSE_FLAG="--convex-decompose True"
        ;;
    cabinet|drawer)
        BASE_DIRS=("./Assets/PartManip/release/data2/haoran/RL-Pose/PoseOrientedGym/assets/assets/drawer")
        SEARCH_PATTERN="StorageFurniture-*" # 'StorageFurniture-'로 시작하는 폴더만 대상으로 함
        TARGET_URDF_FILENAMES=("mobility_new.urdf")
        ;;
    *)
        echo "Error: Unknown command '$COMMAND'. Use 'bottle', 'door','cabinet' or 'drawer'."
        exit 1
        ;;
esac

# ===================================================================
# 2. 변환 작업 실행
# ===================================================================

# <--- 설정된 모든 기본 경로(BASE_DIRS)에 대해 반복 실행하는 루프 추가
for base_dir in "${BASE_DIRS[@]}"; do
    echo "====================================================="
    echo "Processing base directory: $base_dir"
    echo "====================================================="

    # 설정된 경로(base_dir)에서 지정된 패턴(SEARCH_PATTERN)의 폴더를 찾아서 순회합니다.
    find "$base_dir" -type d -name "$SEARCH_PATTERN" | while read -r asset_dir; do
        # 각 폴더 내에서 변환할 URDF 파일 목록(TARGET_URDF_FILENAMES)을 순회합니다.
        for urdf_file in "${TARGET_URDF_FILENAMES[@]}"; do
            input_urdf="$asset_dir/$urdf_file"

            # 변환할 URDF 파일이 실제로 존재하는지 확인합니다.
            if [ -f "$input_urdf" ]; then
                # 출력할 .usd 파일 경로를 생성합니다. (예: a.urdf -> a.usd)
                output_usd="${input_urdf%.urdf}.usd"

                echo "-----------------------------------------------------"
                echo "Found target: $input_urdf"
                echo "Converting to: $output_usd"
                echo "-----------------------------------------------------"

                # Isaac Lab 변환 스크립트를 실행합니다.
                ./isaaclab.sh -p assets/asset_lab/convert_urdf.py \
                  "$input_urdf" \
                  "$output_usd" \
                  --fix-base \
                  --headless \
                  $CONVEX_DECOMPOSE_FLAG
            fi
        done
    done
done

echo ""
echo "All conversions for command '$COMMAND' complete."