#!/usr/bin/env python3
"""
[Step 1-2] 시간 정보 추출 v5 (datetime 단위 충돌 완전 우회)
- datetime 컬럼을 문자열(str)로 처리해서 ns/us 충돌 방지

실행: python 02_extract_dates_v5.py
"""

import pandas as pd
import numpy as np
import os

print("="*70)
print("📅 Step 1-2: 시간 정보 추출")
print("="*70)

# ============================================================
# 1. 통합 데이터 로드
# ============================================================
print("\n[1] 통합 데이터 로드...")
df = pd.read_csv('data/processed/integrated_jobs.csv')
print(f"  → {len(df):,}개")

# ============================================================
# 2. 마감일 파싱 → 문자열로 저장
# ============================================================
print("\n[2] 마감일 파싱...")

# datetime 변환은 계산용으로만 사용, 저장은 str
df['마감일_dt'] = pd.to_datetime(df['마감일'], errors='coerce')
has_deadline = df['마감일_dt'].notna().sum()
print(f"  마감일 있음: {has_deadline:,}개 ({has_deadline/len(df)*100:.1f}%)")

# ============================================================
# 3. 공고ID 기반 시간 보간 (원티드만)
# ============================================================
print("\n[3] 공고ID 기반 시간 보간 (원티드)...")

# 원티드만 추출
wanted_mask = df['출처'] == '원티드'
df_wanted = df[wanted_mask].copy()
df_wanted['공고ID_int'] = pd.to_numeric(df_wanted['공고ID'], errors='coerce')

# 학습 데이터: 마감일이 있는 것
ref = df_wanted[df_wanted['마감일_dt'].notna() & df_wanted['공고ID_int'].notna()].copy()
print(f"  학습 데이터: {len(ref):,}개")

# 선형 회귀 (초 단위로 계산)
X = ref['공고ID_int'].values
y_s = np.array([t.timestamp() for t in ref['마감일_dt']])  # .timestamp() → 초 단위, 확실함

slope, intercept = np.polyfit(X, y_s, 1)
r = np.corrcoef(X, y_s)[0, 1]
print(f"  상관계수: {r:.4f}")

# 테스트
test_id = int(X.mean())
test_ts = slope * test_id + intercept
print(f"  테스트 ID {test_id:,} → 추정: {pd.Timestamp(test_ts, unit='s').strftime('%Y-%m-%d')}")

# 모든 원티드 데이터에 대해 마감일 추정 (없는 것만)
mask_no_deadline = df_wanted['마감일_dt'].isna() & df_wanted['공고ID_int'].notna()

# 추정값 계산 → 문자열로 저장 (dtype 충돌 없음)
est_timestamps = slope * df_wanted.loc[mask_no_deadline, '공고ID_int'].values + intercept
est_dates = []
for ts in est_timestamps:
    try:
        dt = pd.Timestamp(ts, unit='s')
        if pd.Timestamp('2020-01-01') <= dt <= pd.Timestamp('2028-01-01'):
            est_dates.append(dt.strftime('%Y-%m-%d'))
        else:
            est_dates.append(None)
    except:
        est_dates.append(None)

# 원티드 인덱스에 게시일 저장
df.loc[wanted_mask, '마감일_추정'] = None  # 초기화

# 마감일 있는 것: 기존 마감일 사용
df.loc[wanted_mask & df['마감일_dt'].notna(), '마감일_추정'] = \
    df.loc[wanted_mask & df['마감일_dt'].notna(), '마감일']

# 마감일 없는 것: 추정값 사용
no_deadline_idx = df[wanted_mask & df['마감일_dt'].isna() & pd.to_numeric(df['공고ID'], errors='coerce').notna()].index
for i, (idx, est) in enumerate(zip(no_deadline_idx, est_dates)):
    df.loc[idx, '마감일_추정'] = est

valid_est = sum(1 for d in est_dates if d is not None)
print(f"  추정 완료: {valid_est:,}개 / {len(est_dates):,}개")

# 사람인/잡다임: 마감일이 있으면 그대로
other_mask = (~wanted_mask) & df['마감일_dt'].notna()
df.loc[other_mask, '마감일_추정'] = df.loc[other_mask, '마감일']

# ============================================================
# 4. 게시일 추정 = 마감일 - 30일
# ============================================================
print("\n[4] 게시일 추정 (마감일 - 30일)...")

df['마감일_추정_dt'] = pd.to_datetime(df['마감일_추정'], errors='coerce')
df['게시일_추정'] = (df['마감일_추정_dt'] - pd.Timedelta(days=30)).dt.strftime('%Y-%m-%d')
df.loc[df['마감일_추정_dt'].isna(), '게시일_추정'] = None

total_dated = df['게시일_추정'].notna().sum()
print(f"  최종 게시일 추정: {total_dated:,}개 ({total_dated/len(df)*100:.1f}%)")

# ============================================================
# 5. 년/월/분기 컬럼
# ============================================================
print("\n[5] 년/월/분기 컬럼 생성...")

게시일_dt = pd.to_datetime(df['게시일_추정'], errors='coerce')
df['게시년월'] = 게시일_dt.dt.to_period('M').astype(str).where(게시일_dt.notna(), '')
df['게시년도'] = 게시일_dt.dt.year
df['게시월']   = 게시일_dt.dt.month
df['게시분기'] = 게시일_dt.dt.quarter

# datetime 컬럼 제거 (저장 시 충돌 방지)
df = df.drop(columns=['마감일_dt', '마감일_추정_dt'], errors='ignore')

# ============================================================
# 6. 저장
# ============================================================
os.makedirs('data/processed', exist_ok=True)
output_file = 'data/processed/dated_jobs.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n[6] 저장: {output_file}")

# ============================================================
# 7. 월별 분포
# ============================================================
print("\n" + "="*70)
print("📊 월별 공고 수 (게시일 추정 기준)")
print("="*70)

valid_df = df[df['게시년월'].notna() & (df['게시년월'] != '') & (df['게시년월'] != 'NaT')]
monthly = valid_df['게시년월'].value_counts().sort_index()

print()
for month, count in monthly.items():
    bar = '█' * int(count / 50)
    print(f"  {month}: {count:5,}개 {bar}")

print(f"\n  총 시계열 데이터: {monthly.sum():,}개")

# ============================================================
# 8. 출처별 현황
# ============================================================
print("\n" + "="*70)
print("📊 출처별 시간 정보 보유 현황")
print("="*70)

for source in df['출처'].unique():
    sub = df[df['출처'] == source]
    has_date = sub['게시일_추정'].notna().sum()
    print(f"\n  {source}: {len(sub):,}개")
    print(f"    게시일 추정: {has_date:,}개 ({has_date/len(sub)*100:.1f}%)")

print("\n✅ Step 1-2 완료!")
print("\n📌 다음 단계: 03_text_preprocessing.py 실행")