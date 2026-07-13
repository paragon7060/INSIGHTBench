#!/usr/bin/env python3
"""
데이터셋의 태스크 분포를 분석하는 스크립트
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np

def load_episodes_data(episodes_file_path):
    """episodes.jsonl 파일을 로드하여 데이터를 반환"""
    episodes = []
    with open(episodes_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            episode_data = json.loads(line.strip())
            episodes.append(episode_data)
    return episodes

def analyze_task_distribution(episodes):
    """태스크 분포를 분석"""
    task_counter = Counter()
    episode_lengths = defaultdict(list)
    task_episode_counts = defaultdict(int)
    
    for episode in episodes:
        tasks = episode['tasks']
        length = episode['length']
        
        # 각 에피소드의 태스크들을 카운트
        for task in tasks:
            task_counter[task] += 1
            task_episode_counts[task] += 1
            episode_lengths[task].append(length)
    
    return task_counter, episode_lengths, task_episode_counts

def print_task_statistics(task_counter, episode_lengths, task_episode_counts, total_episodes):
    """태스크 통계를 출력"""
    print("=" * 60)
    print("데이터셋 태스크 분포 분석")
    print("=" * 60)
    print(f"총 에피소드 수: {total_episodes}")
    print(f"총 태스크 인스턴스 수: {sum(task_counter.values())}")
    print()
    
    print("태스크별 통계:")
    print("-" * 60)
    print(f"{'태스크':<15} {'에피소드 수':<12} {'비율(%)':<10} {'평균 길이':<12} {'최소 길이':<12} {'최대 길이':<12}")
    print("-" * 60)
    
    for task in sorted(task_counter.keys()):
        episode_count = task_episode_counts[task]
        ratio = (episode_count / total_episodes) * 100
        lengths = episode_lengths[task]
        avg_length = np.mean(lengths)
        min_length = np.min(lengths)
        max_length = np.max(lengths)
        
        print(f"{task:<15} {episode_count:<12} {ratio:<10.1f} {avg_length:<12.1f} {min_length:<12} {max_length:<12}")
    
    print("-" * 60)
    print(f"{'총계':<15} {total_episodes:<12} {100.0:<10.1f} {np.mean([ep['length'] for ep in episodes]):<12.1f} {min([ep['length'] for ep in episodes]):<12} {max([ep['length'] for ep in episodes]):<12}")

def plot_task_distribution(task_counter, output_dir=None):
    """태스크 분포를 시각화"""
    tasks = list(task_counter.keys())
    counts = list(task_counter.values())
    
    plt.figure(figsize=(12, 8))
    
    # 막대 그래프
    plt.subplot(2, 1, 1)
    bars = plt.bar(tasks, counts, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.title('태스크별 에피소드 수 분포', fontsize=14, fontweight='bold')
    plt.xlabel('태스크', fontsize=12)
    plt.ylabel('에피소드 수', fontsize=12)
    plt.xticks(rotation=45)
    
    # 막대 위에 값 표시
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # 파이 차트
    plt.subplot(2, 1, 2)
    colors = plt.cm.Set3(np.linspace(0, 1, len(tasks)))
    wedges, texts, autotexts = plt.pie(counts, labels=tasks, autopct='%1.1f%%', 
                                      colors=colors, startangle=90)
    plt.title('태스크별 비율 분포', fontsize=14, fontweight='bold')
    
    # 파이 차트 텍스트 스타일 조정
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'task_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"그래프가 저장되었습니다: {output_path}")
    
    plt.show()

def plot_episode_length_distribution(episode_lengths, output_dir=None):
    """에피소드 길이 분포를 시각화"""
    plt.figure(figsize=(15, 10))
    
    tasks = list(episode_lengths.keys())
    n_tasks = len(tasks)
    
    # 서브플롯 생성
    cols = min(3, n_tasks)
    rows = (n_tasks + cols - 1) // cols
    
    for i, task in enumerate(tasks):
        plt.subplot(rows, cols, i + 1)
        lengths = episode_lengths[task]
        
        plt.hist(lengths, bins=20, alpha=0.7, color='lightcoral', edgecolor='darkred')
        plt.title(f'{task} (n={len(lengths)})', fontweight='bold')
        plt.xlabel('에피소드 길이')
        plt.ylabel('빈도')
        plt.grid(True, alpha=0.3)
        
        # 통계 정보 추가
        mean_len = np.mean(lengths)
        std_len = np.std(lengths)
        plt.axvline(mean_len, color='red', linestyle='--', linewidth=2, 
                   label=f'평균: {mean_len:.1f}')
        plt.legend()
    
    plt.suptitle('태스크별 에피소드 길이 분포', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'episode_length_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"에피소드 길이 분포 그래프가 저장되었습니다: {output_path}")
    
    plt.show()

def save_detailed_report(task_counter, episode_lengths, task_episode_counts, 
                        total_episodes, output_dir=None):
    """상세 보고서를 파일로 저장"""
    if not output_dir:
        return
    
    output_path = Path(output_dir) / 'task_analysis_report.txt'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("데이터셋 태스크 분포 상세 분석 보고서\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"총 에피소드 수: {total_episodes}\n")
        f.write(f"총 태스크 인스턴스 수: {sum(task_counter.values())}\n\n")
        
        f.write("태스크별 상세 통계:\n")
        f.write("-" * 60 + "\n")
        
        for task in sorted(task_counter.keys()):
            episode_count = task_episode_counts[task]
            ratio = (episode_count / total_episodes) * 100
            lengths = episode_lengths[task]
            avg_length = np.mean(lengths)
            std_length = np.std(lengths)
            min_length = np.min(lengths)
            max_length = np.max(lengths)
            median_length = np.median(lengths)
            
            f.write(f"\n태스크: {task}\n")
            f.write(f"  - 에피소드 수: {episode_count}\n")
            f.write(f"  - 비율: {ratio:.2f}%\n")
            f.write(f"  - 평균 길이: {avg_length:.2f} ± {std_length:.2f}\n")
            f.write(f"  - 중간값 길이: {median_length:.2f}\n")
            f.write(f"  - 최소 길이: {min_length}\n")
            f.write(f"  - 최대 길이: {max_length}\n")
            f.write(f"  - 길이 범위: {max_length - min_length}\n")
    
    print(f"상세 보고서가 저장되었습니다: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="데이터셋 태스크 분포 분석")
    parser.add_argument("--episodes_file", type=str, 
                       default="data/paragon7060/INSIGHT-guide-color-1/meta/episodes.jsonl",
                       help="episodes.jsonl 파일 경로")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="결과를 저장할 디렉토리 (선택사항)")
    parser.add_argument("--no_plot", action="store_true",
                       help="그래프를 표시하지 않음")
    
    args = parser.parse_args()
    
    # 파일 경로 확인
    episodes_file = Path(args.episodes_file)
    if not episodes_file.exists():
        print(f"에러: 파일을 찾을 수 없습니다: {episodes_file}")
        return
    
    # 출력 디렉토리 생성
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
    
    print(f"데이터셋 분석 중: {episodes_file}")
    
    # 데이터 로드
    episodes = load_episodes_data(episodes_file)
    total_episodes = len(episodes)
    
    # 태스크 분포 분석
    task_counter, episode_lengths, task_episode_counts = analyze_task_distribution(episodes)
    
    # 통계 출력
    print_task_statistics(task_counter, episode_lengths, task_episode_counts, total_episodes)
    
    # 그래프 생성 (옵션)
    if not args.no_plot:
        try:
            plot_task_distribution(task_counter, output_dir)
            plot_episode_length_distribution(episode_lengths, output_dir)
        except ImportError:
            print("matplotlib이 설치되지 않아 그래프를 생성할 수 없습니다.")
            print("pip install matplotlib로 설치하세요.")
    
    # 상세 보고서 저장
    if output_dir:
        save_detailed_report(task_counter, episode_lengths, task_episode_counts, 
                           total_episodes, output_dir)

if __name__ == "__main__":
    main()
