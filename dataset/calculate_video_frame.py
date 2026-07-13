import argparse
import os

import cv2
from tqdm import tqdm


def count_videos_and_frames(root_path):
    """
    지정된 루트 경로의 하위 폴더들을 탐색하여 총 비디오 개수와 프레임 수를 계산합니다.
    - 루트 경로 바로 아래의 각 폴더 (예: 'observation.images.left_shoulder')를 탐색합니다.
    - 각 폴더 내의 .mp4 비디오 파일을 대상으로 계산합니다.
    """
    if not os.path.isdir(root_path):
        print(f"오류: '{root_path}'는 유효한 디렉터리가 아닙니다.")
        return

    total_video_count = 0
    total_frame_count = 0

    # root_path 바로 아래에 있는 모든 항목을 가져옵니다.
    try:
        subfolders = [f.path for f in os.scandir(root_path) if f.is_dir()]
    except FileNotFoundError:
        print(f"오류: 경로를 찾을 수 없습니다 -> {root_path}")
        return

    print(f"'{root_path}' 경로에서 하위 폴더들을 탐색합니다...")

    # tqdm을 사용하여 진행 상황을 시각적으로 표시합니다.
    for subfolder_path in tqdm(subfolders, desc="폴더 처리 중"):
        try:
            for filename in os.listdir(subfolder_path):
                # 비디오 파일 확장자(.mp4)를 확인합니다. 다른 형식이 있다면 추가하세요.
                if filename.lower().endswith('.mp4'):
                    video_path = os.path.join(subfolder_path, filename)
                    
                    # 비디오 파일 열기
                    cap = cv2.VideoCapture(video_path)
                    
                    if cap.isOpened():
                        # 프레임 수 읽기
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        total_video_count += 1
                        total_frame_count += frame_count
                        
                        cap.release()
                    else:
                        print(f"\n경고: 비디오 파일을 열 수 없습니다: {video_path}")
        except Exception as e:
            print(f"\n오류 발생 ({subfolder_path}): {e}")


    print("\n--- 계산 완료 ---")
    print(f"📂 총 비디오 파일 개수: {total_video_count}개")
    print(f"🎞️ 모든 비디오의 총 프레임 수: {total_frame_count}개")

def parse_args():
    parser = argparse.ArgumentParser(description="Count MP4 files and frames in dataset video folders.")
    parser.add_argument(
        "root_path",
        metavar="ROOT_PATH",
        help="Directory containing video chunk subdirectories.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    count_videos_and_frames(args.root_path)
