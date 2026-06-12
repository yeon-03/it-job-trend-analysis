#!/usr/bin/env python3
"""
[Step 1-4] EDA (탐색적 데이터 분석)
- 기본 통계
- 월별 채용 트렌드
- 기술 스택 분포
- 직무/회사규모 분포
- 시각화 저장

실행: python 04_eda.py
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 화면 없이 저장
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

# ============================================================
# 한글 폰트 설정
# ============================================================
def set_korean_font():
    """한글 폰트 자동 설정"""
    import platform
    system = platform.system()
    
    if system == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif system == 'Darwin':
        plt.rcParams['font.family'] = 'AppleGothic'
    else:
        # Linux
        try:
            plt.rcParams['font.family'] = 'NanumGothic'
        except:
            pass
    
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

print("="*70)
print("📊 Step 1-4: EDA (탐색적 데이터 분석)")
print("="*70)

# ============================================================
# 1. 데이터 로드
# ============================================================
print("\n[1] 데이터 로드...")
df = pd.read_csv('data/processed/cleaned_jobs.csv')
print(f"  → {len(df):,}개")

os.makedirs('outputs/visualizations', exist_ok=True)

# ============================================================
# 2. 기본 통계
# ============================================================
print("\n[2] 기본 통계")
print("="*70)

print(f"\n📌 전체 현황")
print(f"  총 공고:      {len(df):,}개")
print(f"  고유 회사:    {df['회사명'].nunique():,}개")
print(f"  출처:         {dict(df['출처'].value_counts())}")

print(f"\n📌 직무 카테고리")
for cat, cnt in df['직무카테고리'].value_counts().items():
    print(f"  {cat:12s}: {cnt:5,}개 ({cnt/len(df)*100:.1f}%)")

print(f"\n📌 회사 규모")
for size, cnt in df['회사규모'].value_counts().items():
    print(f"  {size:15s}: {cnt:5,}개 ({cnt/len(df)*100:.1f}%)")

print(f"\n📌 지역 Top 5")
for region, cnt in df['지역'].value_counts().head(5).items():
    print(f"  {region:15s}: {cnt:5,}개")

# ============================================================
# 3. 그래프 1: 월별 채용 공고 수
# ============================================================
print("\n[3] 그래프 1: 월별 채용 트렌드...")

monthly_df = df[
    df['게시년월'].notna() &
    (df['게시년월'] != '') &
    (df['게시년월'] != 'NaT') &
    (df['게시년월'] >= '2025-06') &
    (df['게시년월'] <= '2026-03')
]
monthly = monthly_df['게시년월'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(range(len(monthly)), monthly.values, color='steelblue', alpha=0.8)
ax.set_xticks(range(len(monthly)))
ax.set_xticklabels(monthly.index, rotation=45, ha='right')
ax.set_title('월별 IT 채용 공고 수 (2025.06 ~ 2026.03)', fontsize=14, fontweight='bold')
ax.set_xlabel('월')
ax.set_ylabel('공고 수')

# 값 표시
for bar, val in zip(bars, monthly.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f'{val:,}', ha='center', va='bottom', fontsize=9)

# 피크 표시
max_idx = monthly.values.argmax()
bars[max_idx].set_color('tomato')
ax.annotate(f'피크: {monthly.values[max_idx]:,}개',
            xy=(max_idx, monthly.values[max_idx]),
            xytext=(max_idx + 1, monthly.values[max_idx] + 50),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')

plt.tight_layout()
plt.savefig('outputs/visualizations/01_monthly_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("  저장: 01_monthly_trend.png")

# ============================================================
# 4. 그래프 2: 기술 스택 Top 20
# ============================================================
print("\n[4] 그래프 2: 기술 스택 Top 20...")

tech_freq = pd.read_csv('data/analysis/tech_frequency.csv')
top20 = tech_freq.head(20)

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top20)))[::-1]
bars = ax.barh(range(len(top20)), top20['빈도'], color=colors)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20['기술'])
ax.set_title('IT 채용공고 기술 스택 Top 20', fontsize=14, fontweight='bold')
ax.set_xlabel('공고 수')
ax.invert_yaxis()

for bar, val, pct in zip(bars, top20['빈도'], top20['비율(%)']):
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
            f'{val:,} ({pct}%)', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/visualizations/02_tech_stack_top20.png', dpi=150, bbox_inches='tight')
plt.close()
print("  저장: 02_tech_stack_top20.png")

# ============================================================
# 5. 그래프 3: 직무 카테고리 분포
# ============================================================
print("\n[5] 그래프 3: 직무 카테고리...")

job_counts = df['직무카테고리'].value_counts()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 파이차트
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(job_counts)))
wedges, texts, autotexts = ax1.pie(
    job_counts.values,
    labels=job_counts.index,
    autopct='%1.1f%%',
    colors=colors_pie,
    startangle=90
)
ax1.set_title('직무 카테고리 비율', fontsize=13, fontweight='bold')

# 막대차트
ax2.barh(job_counts.index[::-1], job_counts.values[::-1], color='mediumseagreen', alpha=0.8)
ax2.set_title('직무 카테고리 공고 수', fontsize=13, fontweight='bold')
ax2.set_xlabel('공고 수')
for i, (cat, val) in enumerate(zip(job_counts.index[::-1], job_counts.values[::-1])):
    ax2.text(val + 20, i, f'{val:,}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/visualizations/03_job_category.png', dpi=150, bbox_inches='tight')
plt.close()
print("  저장: 03_job_category.png")

# ============================================================
# 6. 그래프 4: 직무별 기술 스택 히트맵
# ============================================================
print("\n[6] 그래프 4: 직무별 기술 스택 히트맵...")

TOP_TECHS = ['Python', 'Java', 'AWS', 'React', 'Spring', 'Docker',
             'TypeScript', 'Kubernetes', 'MySQL', 'Git', 'LLM', 'Go']
TOP_JOBS = ['백엔드', '프론트엔드', 'AI_ML', 'DevOps', '데이터', '모바일']

heatmap_data = np.zeros((len(TOP_JOBS), len(TOP_TECHS)))

for i, job in enumerate(TOP_JOBS):
    job_df = df[df['직무카테고리'] == job]
    total = len(job_df)
    if total == 0:
        continue
    for j, tech in enumerate(TOP_TECHS):
        count = job_df['tech_all'].astype(str).str.contains(tech, case=False, na=False).sum()
        heatmap_data[i, j] = count / total * 100  # 비율(%)

fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')

ax.set_xticks(range(len(TOP_TECHS)))
ax.set_xticklabels(TOP_TECHS, rotation=45, ha='right')
ax.set_yticks(range(len(TOP_JOBS)))
ax.set_yticklabels(TOP_JOBS)
ax.set_title('직무별 기술 스택 요구 비율 (%)', fontsize=14, fontweight='bold')

plt.colorbar(im, ax=ax, label='요구 비율 (%)')

for i in range(len(TOP_JOBS)):
    for j in range(len(TOP_TECHS)):
        val = heatmap_data[i, j]
        color = 'white' if val > 50 else 'black'
        ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                fontsize=9, color=color, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/visualizations/04_job_tech_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  저장: 04_job_tech_heatmap.png")

# ============================================================
# 7. 그래프 5: 회사 규모별 기술 스택
# ============================================================
print("\n[7] 그래프 5: 회사 규모별 기술 스택...")

TOP5_TECHS = ['Python', 'Java', 'AWS', 'React', 'Docker', 'LLM', 'Spring', 'TypeScript']
sizes = ['스타트업/중소', '대기업']
size_tech_data = {}

for size in sizes:
    size_df = df[df['회사규모'] == size]
    techs = []
    for tech in TOP5_TECHS:
        rate = size_df['tech_all'].astype(str).str.contains(tech, case=False, na=False).sum() / len(size_df) * 100
        techs.append(rate)
    size_tech_data[size] = techs

x = np.arange(len(TOP5_TECHS))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, size_tech_data['스타트업/중소'], width,
               label='스타트업/중소', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, size_tech_data['대기업'], width,
               label='대기업', color='tomato', alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(TOP5_TECHS)
ax.set_title('회사 규모별 기술 스택 요구 비율 (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('요구 비율 (%)')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/visualizations/05_company_size_tech.png', dpi=150, bbox_inches='tight')
plt.close()
print("  저장: 05_company_size_tech.png")

# ============================================================
# 8. EDA 요약 출력
# ============================================================
print("\n" + "="*70)
print("🎉 EDA 완료! 주요 인사이트")
print("="*70)

print(f"""
📊 데이터 규모
  - 총 {len(df):,}개 공고, {df['회사명'].nunique():,}개 회사

🔧 기술 스택 Top 5
  1. Java     {tech_freq.iloc[0]['빈도']:,}개 ({tech_freq.iloc[0]['비율(%)']:.1f}%)
  2. AWS      {tech_freq.iloc[1]['빈도']:,}개 ({tech_freq.iloc[1]['비율(%)']:.1f}%)
  3. Git      {tech_freq.iloc[2]['빈도']:,}개 ({tech_freq.iloc[2]['비율(%)']:.1f}%)
  4. Python   {tech_freq.iloc[3]['빈도']:,}개 ({tech_freq.iloc[3]['비율(%)']:.1f}%)
  5. React    {tech_freq.iloc[4]['빈도']:,}개 ({tech_freq.iloc[4]['비율(%)']:.1f}%)

💼 직무 비율
  - 백엔드 50.2% 압도적 1위
  - AI/ML 11.1% (LLM 10.3% 포함)
  - DevOps 8.0%

📅 채용 시즌
  - 피크: 8월, 10월, 1월
  - 분석 가능 기간: 2025.07 ~ 2026.03

🏢 회사 규모
  - 스타트업/중소 88.9%
  - 대기업 10.9%
""")

print("  시각화 저장 위치: outputs/visualizations/")
print("  01_monthly_trend.png       - 월별 트렌드")
print("  02_tech_stack_top20.png    - 기술 스택 Top 20")
print("  03_job_category.png        - 직무 분포")
print("  04_job_tech_heatmap.png    - 직무별 기술 히트맵")
print("  05_company_size_tech.png   - 회사규모별 기술")

print("\n✅ Step 1-4 완료!")
print("\n📌 다음 단계: Phase 2 트렌드 분석 (05_trend_analysis.py)")
