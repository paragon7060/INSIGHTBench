import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 사용자 환경에 맞는 한글 폰트를 설정해주세요.
# 예: Apple Gothic, NanumGothic 등
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.rcParams['font.family'] = 'Nanum Gothic'
plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지


# 데이터프레임 생성
data = {
    'Dataset Name': ['INSIGHTfixpos', 'INSIGHTfixpos-noguide'],
    '1ext': [1730, 1788],
    '3a': [406, 480],
    '3b': [330, 381],
    '3c': [699, 304],
    '3d': [605, 307],
    '5a': [235, 221],
    '5b': [236, 276],
    '5c': [249, 286],
    '5d': [240, 338],
    '5e': [241, 306],
    '5f': [233, 334],
    '5g': [249, 251],
    '5h': [219, 322],
    'Total Episodes': [4889, 4386]
}

df = pd.DataFrame(data)

# 시각화의 목적에 맞지 않는 'Total Episodes' 열은 제외합니다.
df_plot = df.drop(columns='Total Episodes')
df_plot.set_index('Dataset Name', inplace=True)

# 1. 그룹형 막대 그래프 (Grouped Bar Chart)
df_plot.T.plot(kind='bar', figsize=(18, 8), width=0.8)
plt.title('Task distribution for each dataset', fontsize=20)
plt.xlabel('Task', fontsize=14)
plt.ylabel('Episode number', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Dataset Name', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('grouped_bar_chart.png')
plt.show()
