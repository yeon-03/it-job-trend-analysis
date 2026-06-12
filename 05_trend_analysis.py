#!/usr/bin/env python3
"""
[Phase 2] 트렌드 분석
1. 기술 스택 월별 트렌드 (증가/감소율)
2. 떠오르는 기술 Top 10
3. 직무별 채용 시즌 분석
4. 계절성 분석

실행: python 05_trend_analysis.py
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import json
import os
from collections import defaultdict

# 한글 폰트
import platform
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("📈 Phase 2: 트렌드 분석")
print("="*70)

# ============================================================
# 1. 데이터 로드
# ============================================================
print("\n[1] 데이터 로드...")
df = pd.read_csv('data/processed/cleaned_jobs.csv')
df_timed = df[
    df['게시년월'].notna() &
    (df['게시년월'] != '') &
    (df['게시년월'] != 'NaT') &
    (df['게시년월'] >= '2025-07') &
    (df['게시년월'] <= '2026-03')
].copy()

print(f"  전체: {len(df):,}개")
print(f"  시계열 분석 대상: {len(df_timed):,}개 (2025-07 ~ 2026-03)")

os.makedirs('data/analysis', exist_ok=True)
os.makedirs('outputs/visualizations', exist_ok=True)

MONTHS = sorted(df_timed['게시년월'].unique())
print(f"  분석 기간: {MONTHS[0]} ~ {MONTHS[-1]} ({len(MONTHS)}개월)")

# ============================================================
# 2. 기술 스택 월별 빈도 계산
# ============================================================
print("\n[2] 기술 스택 월별 빈도 계산...")

TOP_TECHS = [
    'Python', 'Java', 'AWS', 'React', 'Spring', 'Docker',
    'TypeScript', 'Kubernetes', 'MySQL', 'Git',
    'LLM', 'Go', 'PostgreSQL', 'Redis', 'Kotlin',
    'FastAPI', 'NestJS', 'GCP', 'CI/CD', 'Linux'
]

monthly_tech = {}
monthly_total = {}

for month in MONTHS:
    month_df = df_timed[df_timed['게시년월'] == month]
    monthly_total[month] = len(month_df)
    monthly_tech[month] = {}

    for tech in TOP_TECHS:
        count = month_df['tech_all'].astype(str).str.contains(
            tech, case=False, na=False
        ).sum()
        monthly_tech[month][tech] = count

# DataFrame으로 변환
tech_monthly_df = pd.DataFrame(monthly_tech).T  # 행=월, 열=기술
tech_monthly_pct = tech_monthly_df.div(
    pd.Series(monthly_total), axis=0
) * 100  # 비율(%)

print(f"  완료: {len(TOP_TECHS)}개 기술 × {len(MONTHS)}개월")

# ============================================================
# 3. 증가율 계산 (전반기 vs 후반기)
# ============================================================
print("\n[3] 기술 스택 증가율 계산...")

# 전반기(2025-07~09) vs 후반기(2025-10~12) 비교
first_half = [m for m in MONTHS if '2025-07' <= m <= '2025-09']
second_half = [m for m in MONTHS if '2025-10' <= m <= '2025-12']
recent = [m for m in MONTHS if m >= '2026-01']

growth_data = []
for tech in TOP_TECHS:
    avg_first = tech_monthly_pct.loc[first_half, tech].mean() if first_half else 0
    avg_second = tech_monthly_pct.loc[second_half, tech].mean() if second_half else 0
    avg_recent = tech_monthly_pct.loc[recent, tech].mean() if recent else 0

    if avg_first > 0:
        growth_rate = (avg_second - avg_first) / avg_first * 100
    else:
        growth_rate = 0

    overall_avg = tech_monthly_pct[tech].mean()

    growth_data.append({
        '기술': tech,
        '전반기평균(%)': round(avg_first, 1),
        '후반기평균(%)': round(avg_second, 1),
        '최근평균(%)': round(avg_recent, 1),
        '증가율(%)': round(growth_rate, 1),
        '전체평균(%)': round(overall_avg, 1),
    })

growth_df = pd.DataFrame(growth_data).sort_values('증가율(%)', ascending=False)
growth_df.to_csv('data/analysis/tech_growth.csv', index=False, encoding='utf-8-sig')

print("\n  📈 기술 스택 증가율 Top 10 (전반기 → 후반기)")
print(f"  {'기술':15s} {'전반기':>8s} {'후반기':>8s} {'증가율':>8s}")
print("  " + "-"*45)
for _, row in growth_df.head(10).iterrows():
    trend = "▲" if row['증가율(%)'] > 0 else "▼"
    print(f"  {row['기술']:15s} {row['전반기평균(%)']:>7.1f}% {row['후반기평균(%)']:>7.1f}% {trend}{abs(row['증가율(%)']):>6.1f}%")

print("\n  📉 기술 스택 감소율 Top 5")
for _, row in growth_df.tail(5).iterrows():
    print(f"  {row['기술']:15s} {row['전반기평균(%)']:>7.1f}% {row['후반기평균(%)']:>7.1f}%  ▼{abs(row['증가율(%)']):>5.1f}%")

# ============================================================
# 4. 그래프 6: 기술 스택 월별 트렌드 (주요 6개)
# ============================================================
print("\n[4] 그래프 6: 기술 스택 월별 트렌드...")

# 흥미로운 기술 선택 (변화가 뚜렷한 것)
TREND_TECHS = ['Python', 'Java', 'LLM', 'React', 'Docker', 'TypeScript']
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

fig, ax = plt.subplots(figsize=(13, 6))

x = range(len(MONTHS))
for tech, color in zip(TREND_TECHS, colors):
    values = [tech_monthly_pct.loc[m, tech] for m in MONTHS]
    ax.plot(x, values, marker='o', label=tech, color=color, linewidth=2, markersize=6)

ax.set_xticks(x)
ax.set_xticklabels(MONTHS, rotation=45, ha='right')
ax.set_title('주요 기술 스택 월별 요구 비율 추이', fontsize=14, fontweight='bold')
ax.set_ylabel('요구 비율 (%)')
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0)

plt.tight_layout()
plt.savefig('outputs/visualizations/06_tech_trend_monthly.png', dpi=150, bbox_inches='tight')
plt.close()
print("  저장: 06_tech_trend_monthly.png")

# ============================================================
# 5. 그래프 7: 떠오르는 기술 vs 줄어드는 기술
# ============================================================
print("\n[5] 그래프 7: 떠오르는 기술 vs 줄어드는 기술...")

rising = growth_df.nlargest(8, '증가율(%)')
falling = growth_df.nsmallest(5, '증가율(%)')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 떠오르는 기술
bars1 = ax1.barh(rising['기술'][::-1], rising['증가율(%)'][::-1],
                  color='#2ecc71', alpha=0.85)
ax1.set_title('📈 떠오르는 기술 Top 8\n(전반기→후반기 증가율)', fontsize=12, fontweight='bold')
ax1.set_xlabel('증가율 (%)')
ax1.axvline(x=0, color='black', linewidth=0.8)
for bar, val in zip(bars1, rising['증가율(%)'][::-1]):
    ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             f'+{val:.1f}%', va='center', fontsize=10, color='#27ae60', fontweight='bold')

# 줄어드는 기술
bars2 = ax2.barh(falling['기술'], falling['증가율(%)'],
                  color='#e74c3c', alpha=0.85)
ax2.set_title('📉 줄어드는 기술 Top 5\n(전반기→후반기 감소율)', fontsize=12, fontweight='bold')
ax2.set_xlabel('증가율 (%)')
ax2.axvline(x=0, color='black', linewidth=0.8)
for bar, val in zip(bars2, falling['증가율(%)']):
    ax2.text(bar.get_width() - 0.3, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', ha='right', fontsize=10,
             color='#c0392b', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/visualizations/07_rising_falling_tech.png', dpi=150, bbox_inches='tight')
plt.close()
print("  저장: 07_rising_falling_tech.png")

# ============================================================
# 6. 직무별 채용 시즌 분석
# ============================================================
print("\n[6] 직무별 채용 시즌 분석...")

TOP_JOBS = ['백엔드', '프론트엔드', 'AI_ML', 'DevOps', '데이터', '모바일']
job_monthly = {}

for job in TOP_JOBS:
    job_monthly[job] = []
    for month in MONTHS:
        month_df = df_timed[df_timed['게시년월'] == month]
        count = (month_df['직무카테고리'] == job).sum()
        job_monthly[job].append(count)

# 그래프 8: 직무별 월별 채용 트렌드
print("\n[7] 그래프 8: 직무별 채용 시즌...")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
job_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

for i, (job, color) in enumerate(zip(TOP_JOBS, job_colors)):
    ax = axes[i]
    values = job_monthly[job]
    bars = ax.bar(range(len(MONTHS)), values, color=color, alpha=0.75)

    # 피크 월 표시
    peak_idx = np.argmax(values)
    bars[peak_idx].set_alpha(1.0)
    bars[peak_idx].set_edgecolor('black')
    bars[peak_idx].set_linewidth(2)

    ax.set_title(f'{job}', fontsize=12, fontweight='bold', color=color)
    ax.set_xticks(range(len(MONTHS)))
    ax.set_xticklabels([m[5:] for m in MONTHS], rotation=45, fontsize=8)
    ax.set_ylabel('공고 수', fontsize=9)
    ax.text(peak_idx, values[peak_idx] + 1,
            f'피크\n{MONTHS[peak_idx][5:]}',
            ha='center', fontsize=8, color='black', fontweight='bold')

plt.suptitle('직무별 월별 채용 공고 수 (2025.07~2026.03)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('outputs/visualizations/08_job_monthly_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("  저장: 08_job_monthly_trend.png")

# ============================================================
# 7. 채용 시즌 요약 출력
# ============================================================
print("\n[8] 직무별 최적 지원 시기...")
print(f"\n  {'직무':12s} {'피크월':10s} {'피크공고수':>10s} {'추천 준비 시작':>14s}")
print("  " + "-"*50)

for job in TOP_JOBS:
    values = job_monthly[job]
    peak_idx = np.argmax(values)
    peak_month = MONTHS[peak_idx]
    peak_count = values[peak_idx]

    # 추천 준비 시작: 피크 2개월 전
    peak_year, peak_m = int(peak_month[:4]), int(peak_month[5:])
    prep_m = peak_m - 2
    prep_year = peak_year
    if prep_m <= 0:
        prep_m += 12
        prep_year -= 1
    prep_start = f"{prep_year}-{prep_m:02d}"

    print(f"  {job:12s} {peak_month:10s} {peak_count:>10,}개 {prep_start:>14s}부터")

# ============================================================
# 8. 트렌드 분석 저장
# ============================================================
print("\n[9] 트렌드 분석 결과 저장...")

# 월별 기술 빈도 저장
tech_monthly_pct.to_csv('data/analysis/tech_monthly_pct.csv', encoding='utf-8-sig')

# 직무별 월별 공고수 저장
job_monthly_df = pd.DataFrame(job_monthly, index=MONTHS)
job_monthly_df.to_csv('data/analysis/job_monthly.csv', encoding='utf-8-sig')

print("\n" + "="*70)
print("✅ Phase 2 완료!")
print("="*70)
print("""
  산출물:
  - data/analysis/tech_growth.csv        (기술 증가율)
  - data/analysis/tech_monthly_pct.csv   (월별 기술 비율)
  - data/analysis/job_monthly.csv        (직무별 월별 공고수)
  - outputs/visualizations/06_tech_trend_monthly.png
  - outputs/visualizations/07_rising_falling_tech.png
  - outputs/visualizations/08_job_monthly_trend.png
""")
print("📌 다음 단계: Phase 3 RAG 챗봇 (06_rag_chatbot.py)")
