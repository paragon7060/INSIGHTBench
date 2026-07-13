import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 정의
tasks = ['1ext', '3a', '3b', '3c', '3d', '5a', '5b', '5c', '5d', '5e', '5f', '5g', '5h']
insight_fixpos = np.array([1730, 406, 330, 699, 605, 235, 236, 249, 240, 241, 233, 249, 219])
insight_fixpos_noguide = np.array([1788, 480, 381, 304, 307, 221, 276, 286, 338, 306, 334, 251, 322])

# 두 데이터셋 중 최소값 계산 (Balanced Dataset)
balanced_counts = np.minimum(insight_fixpos, insight_fixpos_noguide)

# 2. 그래프 설정
fig, ax = plt.subplots(figsize=(10, 5))

# 3. 막대 그리기 (단일 막대)
# 색상은 논문용으로 차분한 파란색 계열 (#4C72B0) 사용
ax.bar(tasks, balanced_counts, color='#4C72B0', edgecolor='black', linewidth=0.6, alpha=0.9, width=0.65)

# 4. 레이블 및 스타일 설정
ax.set_ylabel('Number of Episodes', fontsize=12, fontweight='bold')
ax.set_xlabel('Task Variation ID', fontsize=12, fontweight='bold')
# ax.set_title('Balanced Task Distribution', fontsize=14) # 논문에서는 보통 캡션을 쓰므로 제목은 생략 가능

# 격자 추가 (가독성)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)

# 불필요한 테두리 제거 (깔끔하게)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 5. 저장
plt.tight_layout()
plt.savefig('balanced_task_distribution.png', format='png', bbox_inches='tight')
plt.show()