#!/usr/bin/env python3
"""
Dataset task distribution analysis script (English version)
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_episodes_data(episodes_file_path):
    """Load episodes.jsonl file and return data"""
    episodes = []
    with open(episodes_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            episode_data = json.loads(line.strip())
            episodes.append(episode_data)
    return episodes

def find_all_datasets(root_path):
    """Find all datasets under root_path and return them"""
    root = Path(root_path)
    datasets = []
    
    if not root.exists():
        print(f"Error: Path not found: {root}")
        return datasets
    
    # Check all folders under data/paragon7060
    for dataset_dir in root.iterdir():
        if dataset_dir.is_dir():
            # Check if meta/episodes.jsonl file exists
            episodes_file = dataset_dir / "meta" / "episodes.jsonl"
            if episodes_file.exists():
                datasets.append({
                    'name': dataset_dir.name,
                    'path': dataset_dir,
                    'episodes_file': episodes_file
                })
                print(f"Dataset found: {dataset_dir.name}")
            else:
                print(f"Warning: {dataset_dir.name} does not have episodes.jsonl.")
    
    return datasets

def analyze_single_dataset(dataset_info):
    """Analyze task distribution of a single dataset"""
    dataset_name = dataset_info['name']
    episodes_file = dataset_info['episodes_file']
    
    print(f"\nAnalyzing: {dataset_name}")
    
    try:
        episodes = load_episodes_data(episodes_file)
        total_episodes = len(episodes)
        
        task_counter = Counter()
        episode_lengths = defaultdict(list)
        task_episode_counts = defaultdict(int)
        
        for episode in episodes:
            tasks = episode['tasks']
            length = episode['length']
            
            # Count tasks in each episode
            for task in tasks:
                task_counter[task] += 1
                task_episode_counts[task] += 1
                episode_lengths[task].append(length)
        
        return {
            'dataset_name': dataset_name,
            'total_episodes': total_episodes,
            'task_counter': task_counter,
            'episode_lengths': episode_lengths,
            'task_episode_counts': task_episode_counts,
            'episodes': episodes
        }
    
    except Exception as e:
        print(f"Error: Failed to analyze {dataset_name}: {e}")
        return None

def analyze_all_datasets(datasets):
    """Analyze all datasets"""
    results = []
    
    for dataset_info in datasets:
        result = analyze_single_dataset(dataset_info)
        if result:
            results.append(result)
    
    return results

def print_summary_statistics(results):
    """Print overall summary statistics"""
    print("\n" + "=" * 80)
    print("Overall Dataset Summary Statistics")
    print("=" * 80)
    
    total_episodes = sum(r['total_episodes'] for r in results)
    total_datasets = len(results)
    
    print(f"Total number of datasets: {total_datasets}")
    print(f"Total number of episodes: {total_episodes}")
    print()
    
    # Episodes per dataset
    print("Episodes per dataset:")
    print("-" * 50)
    print(f"{'Dataset Name':<30} {'Episodes':<15} {'Ratio(%)':<10}")
    print("-" * 50)
    
    for result in sorted(results, key=lambda x: x['total_episodes'], reverse=True):
        dataset_name = result['dataset_name']
        episodes = result['total_episodes']
        ratio = (episodes / total_episodes) * 100
        print(f"{dataset_name:<30} {episodes:<15} {ratio:<10.1f}")
    
    print("-" * 50)
    print(f"{'Total':<30} {total_episodes:<15} {100.0:<10.1f}")

def print_task_statistics(results):
    """Print task statistics"""
    print("\n" + "=" * 80)
    print("Overall Task Distribution Analysis")
    print("=" * 80)
    
    # Combine all tasks
    global_task_counter = Counter()
    global_episode_lengths = defaultdict(list)
    global_task_episode_counts = defaultdict(int)
    
    for result in results:
        for task, count in result['task_counter'].items():
            global_task_counter[task] += count
            global_task_episode_counts[task] += result['task_episode_counts'][task]
            global_episode_lengths[task].extend(result['episode_lengths'][task])
    
    total_episodes = sum(r['total_episodes'] for r in results)
    
    print(f"Total episodes: {total_episodes}")
    print(f"Total task instances: {sum(global_task_counter.values())}")
    print()
    
    print("Task statistics:")
    print("-" * 80)
    print(f"{'Task':<20} {'Episodes':<12} {'Ratio(%)':<10} {'Avg Length':<12} {'Min Length':<12} {'Max Length':<12}")
    print("-" * 80)
    
    for task in sorted(global_task_counter.keys()):
        episode_count = global_task_episode_counts[task]
        ratio = (episode_count / total_episodes) * 100
        lengths = global_episode_lengths[task]
        avg_length = np.mean(lengths)
        min_length = np.min(lengths)
        max_length = np.max(lengths)
        
        print(f"{task:<20} {episode_count:<12} {ratio:<10.1f} {avg_length:<12.1f} {min_length:<12} {max_length:<12}")

def create_dataset_comparison_table(results):
    """Create dataset comparison table"""
    print("\n" + "=" * 100)
    print("Detailed Dataset Comparison")
    print("=" * 100)
    
    # Collect all unique tasks
    all_tasks = set()
    for result in results:
        all_tasks.update(result['task_counter'].keys())
    all_tasks = sorted(all_tasks)
    
    # Table header
    header = f"{'Dataset Name':<25}"
    for task in all_tasks:
        header += f"{task:<12}"
    header += f"{'Total Episodes':<12}"
    print(header)
    print("-" * len(header))
    
    # Row for each dataset
    for result in sorted(results, key=lambda x: x['total_episodes'], reverse=True):
        row = f"{result['dataset_name']:<25}"
        for task in all_tasks:
            count = result['task_episode_counts'].get(task, 0)
            row += f"{count:<12}"
        row += f"{result['total_episodes']:<12}"
        print(row)
    
    # Total row
    print("-" * len(header))
    total_row = f"{'Total':<25}"
    for task in all_tasks:
        total_count = sum(r['task_episode_counts'].get(task, 0) for r in results)
        total_row += f"{total_count:<12}"
    total_episodes = sum(r['total_episodes'] for r in results)
    total_row += f"{total_episodes:<12}"
    print(total_row)

def plot_dataset_comparison(results, output_dir=None):
    """Visualize dataset comparison"""
    if not results:
        return
    
    # Episodes per dataset
    dataset_names = [r['dataset_name'] for r in results]
    episode_counts = [r['total_episodes'] for r in results]
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.figure(figsize=(15, 10))
    
    # 1. Bar chart of episodes per dataset
    plt.subplot(2, 2, 1)
    bars = plt.bar(range(len(dataset_names)), episode_counts, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.title('Episodes per Dataset', fontsize=14, fontweight='bold')
    plt.xlabel('Dataset')
    plt.ylabel('Number of Episodes')
    plt.xticks(range(len(dataset_names)), dataset_names, rotation=45, ha='right')
    
    # Display values on bars
    for bar, count in zip(bars, episode_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(episode_counts)*0.01, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # 2. Pie chart of dataset distribution
    plt.subplot(2, 2, 2)
    colors = plt.cm.Set3(np.linspace(0, 1, len(dataset_names)))
    wedges, texts, autotexts = plt.pie(episode_counts, labels=dataset_names, autopct='%1.1f%%', 
                                      colors=colors, startangle=90)
    plt.title('Dataset Distribution Ratio', fontsize=14, fontweight='bold')
    
    # 3. Overall task distribution
    plt.subplot(2, 2, 3)
    global_task_counter = Counter()
    for result in results:
        global_task_counter.update(result['task_counter'])
    
    tasks = list(global_task_counter.keys())
    counts = list(global_task_counter.values())
    
    bars = plt.bar(tasks, counts, color='lightcoral', edgecolor='darkred', alpha=0.7)
    plt.title('Overall Task Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Task')
    plt.ylabel('Number of Episodes')
    plt.xticks(rotation=45, ha='right')
    
    # 4. Task diversity per dataset
    plt.subplot(2, 2, 4)
    task_diversity = [len(r['task_counter']) for r in results]
    bars = plt.bar(range(len(dataset_names)), task_diversity, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    plt.title('Task Diversity per Dataset', fontsize=14, fontweight='bold')
    plt.xlabel('Dataset')
    plt.ylabel('Number of Unique Tasks')
    plt.xticks(range(len(dataset_names)), dataset_names, rotation=45, ha='right')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'dataset_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Dataset comparison graph saved: {output_path}")
    
    plt.show()

def save_comprehensive_report(results, output_dir=None):
    """Save comprehensive report to file"""
    if not output_dir or not results:
        return
    
    output_path = Path(output_dir) / 'comprehensive_dataset_analysis.txt'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Comprehensive Dataset Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall summary
        total_episodes = sum(r['total_episodes'] for r in results)
        total_datasets = len(results)
        f.write(f"Total number of datasets: {total_datasets}\n")
        f.write(f"Total number of episodes: {total_episodes}\n\n")
        
        # Detailed information per dataset
        f.write("Detailed Information per Dataset:\n")
        f.write("-" * 80 + "\n")
        
        for result in sorted(results, key=lambda x: x['total_episodes'], reverse=True):
            f.write(f"\nDataset: {result['dataset_name']}\n")
            f.write(f"  - Total episodes: {result['total_episodes']}\n")
            f.write(f"  - Unique tasks: {len(result['task_counter'])}\n")
            f.write(f"  - Task distribution:\n")
            
            for task, count in sorted(result['task_counter'].items()):
                ratio = (result['task_episode_counts'][task] / result['total_episodes']) * 100
                f.write(f"    * {task}: {result['task_episode_counts'][task]} episodes ({ratio:.1f}%)\n")
        
        # Overall task statistics
        f.write("\nOverall Task Statistics:\n")
        f.write("-" * 80 + "\n")
        
        global_task_counter = Counter()
        global_episode_lengths = defaultdict(list)
        global_task_episode_counts = defaultdict(int)
        
        for result in results:
            for task, count in result['task_counter'].items():
                global_task_counter[task] += count
                global_task_episode_counts[task] += result['task_episode_counts'][task]
                global_episode_lengths[task].extend(result['episode_lengths'][task])
        
        for task in sorted(global_task_counter.keys()):
            episode_count = global_task_episode_counts[task]
            ratio = (episode_count / total_episodes) * 100
            lengths = global_episode_lengths[task]
            avg_length = np.mean(lengths)
            std_length = np.std(lengths)
            min_length = np.min(lengths)
            max_length = np.max(lengths)
            
            f.write(f"\nTask: {task}\n")
            f.write(f"  - Total episodes: {episode_count}\n")
            f.write(f"  - Ratio: {ratio:.2f}%\n")
            f.write(f"  - Average length: {avg_length:.2f} ± {std_length:.2f}\n")
            f.write(f"  - Length range: {min_length} - {max_length}\n")
    
    print(f"Comprehensive report saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze task distribution of all datasets under data/paragon7060")
    parser.add_argument("--root_path", type=str, 
                       default="data/paragon7060",
                       help="Root path to analyze (default: data/paragon7060)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save results (optional)")
    parser.add_argument("--no_plot", action="store_true",
                       help="Do not display graphs")
    
    args = parser.parse_args()
    
    # Check root path
    root_path = Path(args.root_path)
    if not root_path.exists():
        print(f"Error: Path not found: {root_path}")
        return
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
    
    print(f"Starting dataset analysis: {root_path}")
    
    # Find all datasets
    datasets = find_all_datasets(root_path)
    
    if not datasets:
        print("No datasets found to analyze.")
        return
    
    print(f"\nFound {len(datasets)} datasets in total.")
    
    # Analyze all datasets
    results = analyze_all_datasets(datasets)
    
    if not results:
        print("No datasets available for analysis.")
        return
    
    # Print results
    print_summary_statistics(results)
    print_task_statistics(results)
    create_dataset_comparison_table(results)
    
    # Generate graphs (optional)
    if not args.no_plot:
        try:
            plot_dataset_comparison(results, output_dir)
        except ImportError:
            print("Cannot generate graphs: matplotlib not installed.")
            print("Install with: pip install matplotlib")
    
    # Save comprehensive report
    if output_dir:
        save_comprehensive_report(results, output_dir)

if __name__ == "__main__":
    main()