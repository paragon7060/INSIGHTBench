import argparse
import logging
from pathlib import Path

# 필요한 라이브러리를 임포트합니다.
try:
    import av
    from tqdm import tqdm
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
except ImportError as e:
    print(f"Error: A required library is not installed. {e}")
    print("Please ensure your conda environment is activated and has lerobot, tqdm, av installed.")
    exit()

# 기본 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def verify_video_integrity(video_path: Path) -> tuple[bool, str | None]:
    """
    비디오 파일의 모든 프레임을 디코딩하여 데이터 무결성을 검사합니다.
    (제공해주신 코드의 함수를 그대로 사용)
    """
    try:
        with av.open(str(video_path)) as container:
            # 비디오 스트림의 모든 프레임을 디코딩 시도
            for frame in container.decode(video=0):
                pass
        return True, None
    except Exception as e:
        # 에러 발생 시 실패로 간주하고 에러 메시지 반환
        return False, f"{type(e).__name__}: {e}"

def main():
    parser = argparse.ArgumentParser(
        description="Verify all video files in a LeRobotDataset and find corrupted episodes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("repo_id", type=str, help="The repo_id of the dataset (e.g., 'user/dataset_name').")
    parser.add_argument("root", type=str, help="The local root directory of the dataset.")
    args = parser.parse_args()

    print("--- Loading Dataset Metadata ---")
    try:
        # 데이터 전체가 아닌 메타데이터만 로드하여 빠르게 처리합니다.
        meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=args.root)
        print("Metadata loaded successfully.")
    except Exception as e:
        print(f"Failed to load dataset metadata: {e}")
        return

    video_keys = meta.video_keys
    if not video_keys:
        print("This dataset does not contain any videos to verify.")
        return

    print(f"\n--- Verifying all videos for {meta.total_episodes} episodes ---")
    
    corrupted_episodes = set()
    all_episode_indices = list(meta.episodes.keys())

    # tqdm을 사용하여 전체 진행 상황을 보여줍니다.
    for ep_idx in tqdm(all_episode_indices, desc="Scanning Episodes"):
        for vid_key in video_keys:
            video_path = meta.root / meta.get_video_file_path(ep_idx, vid_key)

            if not video_path.exists():
                logging.warning(f"File does not exist for episode {ep_idx}, key '{vid_key}': {video_path}")
                continue

            is_ok, error_msg = verify_video_integrity(video_path)
            if not is_ok:
                # tqdm 진행률 바 위에 에러 메시지를 출력하기 위해 logging.error 사용
                logging.error(f"\n[CORRUPTED] Found error in episode {ep_idx}")
                logging.error(f"  - File: {video_path}")
                logging.error(f"  - Details: {error_msg}")
                corrupted_episodes.add(ep_idx)
                # 이 에피소드에서 손상된 파일을 찾았으므로 다음 에피소드로 넘어갑니다.
                break 
    
    print("\n" + "=" * 60)
    print("Verification Complete.")
    if corrupted_episodes:
        sorted_episodes = sorted(list(corrupted_episodes))
        print(f"\nFound {len(sorted_episodes)} corrupted episode(s):")
        print(sorted_episodes)
        
        print("\n# You can copy this set into your train_act.py script:")
        print(f"CORRUPTED_EPISODE_INDICES = {set(sorted_episodes)}")
    else:
        print("\n🎉 All video files passed the integrity check!")
    print("=" * 60)


if __name__ == "__main__":
    main()