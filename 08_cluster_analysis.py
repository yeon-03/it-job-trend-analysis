#!/usr/bin/env python3
"""
[Phase 4-2] 클러스터 분석 
- GPU 작업 후 로컬에서 실행
- 클러스터별 특성 분석
- 클러스터 명명
- 시각화

실행: python 08_cluster_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import platform
from collections import Counter

if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("📊 Phase 4-2: 클러스터 분석")
print("="*70)

# ============================================================
# 1. 클러스터 결과 로드
# ============================================================
print("\n[1] 클러스터 결과 로드...")

try:
    df = pd.read_csv('data/analysis/clusters.csv')
    print(f"  공고 수: {len(df):,}개")
    print(f"  클러스터 수: {df['cluster'].nunique()}개")
except FileNotFoundError:
    print("  ❌ clusters.csv 없음 - GPU 서버에서 07_bert_clustering.py 먼저 실행!")
    exit(1)

os.makedirs('outputs/visualizations', exist_ok=True)

# ============================================================
# 2. 클러스터별 상세 분석
# ============================================================
print("\n[2] 클러스터별 상세 분석...")

cluster_names = {}  # 클러스터 이름 (자동 명명)

print(f"\n{'='*70}")
for c in sorted(df['cluster'].unique()):
    cdf = df[df['cluster'] == c]
    count = len(cdf)

    # 직무 분포
    job_dist = cdf['직무카테고리'].value_counts()
    top_job = job_dist.index[0]
    top_job_pct = job_dist.iloc[0] / count * 100

    # 기술 스택 Top 10
    all_techs = []
    for tech_str in cdf['tech_all'].dropna():
        if tech_str and str(tech_str) != 'nan':
            all_techs.extend([t.strip() for t in str(tech_str).split(',') if t.strip()])
    top_techs = Counter(all_techs).most_common(10)

    # 회사 규모
    size_dist = cdf['회사규모'].value_counts()

    # 클러스터 자동 명명
    top3_techs = [t for t, _ in top_techs[:3]]
    cluster_names[c] = f"C{c:02d}-{top_job}({'+'.join(top3_techs[:2])})"

    print(f"\n[클러스터 {c}] {count:,}개 공고")
    print(f"  직무: {top_job} ({top_job_pct:.1f}%) | {', '.join(job_dist.index[:3].tolist())}")
    print(f"  기술 Top 5: {', '.join([t for t, _ in top_techs[:5]])}")
    print(f"  회사규모: {dict(size_dist.head(2))}")

# ============================================================
# 3. 그래프: 클러스터별 직무 분포 히트맵
# ============================================================
print("\n\n[3] 그래프: 클러스터 직무 히트맵...")

job_cats = ['백엔드', '프론트엔드', 'AI_ML', 'DevOps', '데이터', '모바일', '기타']
clusters = sorted(df['cluster'].unique())
heatmap = np.zeros((len(clusters), len(job_cats)))

for i, c in enumerate(clusters):
    cdf = df[df['cluster'] == c]
    for j, job in enumerate(job_cats):
        heatmap[i, j] = (cdf['직무카테고리'] == job).sum() / len(cdf) * 100

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(heatmap, cmap='Blues', aspect='auto')
ax.set_xticks(range(len(job_cats)))
ax.set_xticklabels(job_cats, rotation=45, ha='right')
ax.set_yticks(range(len(clusters)))
ax.set_yticklabels([f'클러스터 {c}' for c in clusters])
ax.set_title('클러스터별 직무 카테고리 분포 (%)', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, label='비율 (%)')

for i in range(len(clusters)):
    for j in range(len(job_cats)):
        val = heatmap[i, j]
        if val > 0:
            color = 'white' if val > 40 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/visualizations/10_cluster_job_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  저장: 10_cluster_job_heatmap.png")

# ============================================================
# 4. 그래프: 클러스터별 기술 스택
# ============================================================
print("\n[4] 그래프: 클러스터별 기술 스택...")

TOP_TECHS = ['Python', 'Java', 'AWS', 'React', 'Spring', 'Docker',
             'TypeScript', 'Kubernetes', 'MySQL', 'Git', 'LLM', 'Go']

tech_heatmap = np.zeros((len(clusters), len(TOP_TECHS)))

for i, c in enumerate(clusters):
    cdf = df[df['cluster'] == c]
    for j, tech in enumerate(TOP_TECHS):
        rate = cdf['tech_all'].astype(str).str.contains(
            tech, case=False, na=False
        ).sum() / len(cdf) * 100
        tech_heatmap[i, j] = rate

fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(tech_heatmap, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(TOP_TECHS)))
ax.set_xticklabels(TOP_TECHS, rotation=45, ha='right')
ax.set_yticks(range(len(clusters)))
ax.set_yticklabels([f'클러스터 {c}' for c in clusters])
ax.set_title('클러스터별 기술 스택 요구 비율 (%)', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, label='요구 비율 (%)')

for i in range(len(clusters)):
    for j in range(len(TOP_TECHS)):
        val = tech_heatmap[i, j]
        if val > 5:
            color = 'white' if val > 50 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/visualizations/11_cluster_tech_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  저장: 11_cluster_tech_heatmap.png")

# ============================================================
# 5. 클러스터 요약 저장
# ============================================================
print("\n[5] 클러스터 요약 저장...")

summary_rows = []
for c in sorted(df['cluster'].unique()):
    cdf = df[df['cluster'] == c]
    all_techs = []
    for tech_str in cdf['tech_all'].dropna():
        if tech_str and str(tech_str) != 'nan':
            all_techs.extend([t.strip() for t in str(tech_str).split(',') if t.strip()])
    top_techs = [t for t, _ in Counter(all_techs).most_common(5)]

    summary_rows.append({
        '클러스터': c,
        '공고수': len(cdf),
        '주요직무': cdf['직무카테고리'].value_counts().index[0],
        '주요기술': ', '.join(top_techs),
        '주요회사규모': cdf['회사규모'].value_counts().index[0],
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('data/analysis/cluster_final_summary.csv',
                  index=False, encoding='utf-8-sig')
print("  저장: data/analysis/cluster_final_summary.csv")

print("\n" + "="*70)
print("✅ Phase 4-2 완료!")
print("="*70)
print(f"""
  산출물:
  - outputs/visualizations/10_cluster_job_heatmap.png
  - outputs/visualizations/11_cluster_tech_heatmap.png
  - data/analysis/cluster_final_summary.csv

  모든 작업 완료!
""")
