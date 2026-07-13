import argparse
from datasets import disable_caching
from lerobot.datasets.lerobot_dataset import LeRobotDataset

parser = argparse.ArgumentParser()
parser.add_argument("--repo_id", type=str, default="paragon7060/INSIGHT-NG-final")
parser.add_argument("--local_root_dir", type=str, default="./data/paragon7060/INSIGHT-NG-final")

args_cli = parser.parse_args()

disable_caching()
dataset = LeRobotDataset(repo_id=args_cli.repo_id, root=args_cli.local_root_dir)
dataset.push_to_hub(upload_large_folder=True)