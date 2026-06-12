#!/usr/bin/env python3
"""
[Step 1-1] 데이터 통합 (v2 - 원티드 2개 파일 통합)

원티드 데이터:
- wanted_past_1year.csv (7,165개, 과거 1년)
- _RUKDP9G.csv (9,674개, 최근)
→ 합치면 16,839개

전체:
- 원티드: 16,839개
- 사람인: 1,375개
- 잡다임: 11개
→ 총 18,225개

실행: python 01_integrate_data.py
"""

import pandas as pd
from datetime import datetime
import os
import sys

# 원본 데이터 파일 존재 확인
required_files = [
    'wanted_past_1year.csv',
    'wanted_recent.csv',
    'saramin_it_20260518.csv',
    'jobda_it_20260518_161026.csv',
]
missing = [f for f in required_files if not os.path.exists(f)]
if missing:
    print("❌ 원본 데이터 파일을 찾을 수 없습니다:")
    for f in missing:
        print(f"   - {f}")
    print("\n→ 위 파일들을 프로젝트 루트 폴더에 배치한 뒤 다시 실행하세요.")
    print("→ 분석 결과만 확인하려면 데이터 파이프라인 없이 서버를 바로 실행하세요:")
    print("   uvicorn server:app --reload --port 8000")
    sys.exit(1)

print("="*70)
print("📥 Step 1-1: 데이터 통합 (원티드 2개 파일 + 사람인 + 잡다임)")
print("="*70)

# ============================================================
# 1. 원티드 데이터 통합 (2개 파일)
# ============================================================
print("\n[1] 원티드 데이터 로드 (2개 파일)...")

# 1-1. 과거 1년 데이터
df_wanted_old = pd.read_csv('wanted_past_1year.csv')
print(f"  과거 1년 (wanted_past_1year.csv): {len(df_wanted_old):,}개")

# 1-2. 최근 데이터
df_wanted_new = pd.read_csv('wanted_recent.csv')
print(f"  최근 (wanted_recent.csv): {len(df_wanted_new):,}개")

# 1-3. 최근 데이터에 빈 컬럼 추가 (기존 데이터와 맞추기)
if '게시일' not in df_wanted_new.columns:
    df_wanted_new['게시일'] = ''
if '수정일' not in df_wanted_new.columns:
    df_wanted_new['수정일'] = ''
if '마감일' not in df_wanted_new.columns:
    df_wanted_new['마감일'] = ''

# 1-4. 통합
df_wanted = pd.concat([df_wanted_old, df_wanted_new], ignore_index=True)
print(f"  통합 전: {len(df_wanted):,}개")

# 1-5. 중복 제거 (공고ID 기준)
before = len(df_wanted)
df_wanted = df_wanted.drop_duplicates(subset=['공고ID'], keep='first')
after = len(df_wanted)
print(f"  중복 제거: {before - after}개")
print(f"  최종 원티드: {after:,}개")

# 표준 스키마로 변환
wanted_std = pd.DataFrame({
    '공고ID': df_wanted['공고ID'].astype(str),
    '출처': '원티드',
    '공고제목': df_wanted['공고제목'],
    '회사명': df_wanted['회사명'],
    '지역': df_wanted['지역'],
    '경력': df_wanted['경력'],
    '주요업무': df_wanted['주요업무'],
    '자격요건': df_wanted['자격요건'],
    '우대사항': df_wanted['우대사항'],
    '혜택복지': df_wanted['혜택복지'],
    '마감일': df_wanted['마감일'],
    '상태': df_wanted['상태'],
    '링크': df_wanted['링크'],
    '수집일시': df_wanted['수집일시'],
})

# ============================================================
# 2. 사람인 데이터 로드
# ============================================================
print("\n[2] 사람인 데이터 로드...")
df_saramin = pd.read_csv('saramin_it_20260518.csv')
print(f"  → {len(df_saramin):,}개")

saramin_std = pd.DataFrame({
    '공고ID': 'SR_' + df_saramin.index.astype(str),
    '출처': '사람인',
    '공고제목': df_saramin['공고제목'],
    '회사명': df_saramin['회사명'],
    '지역': df_saramin['지역'],
    '경력': df_saramin['경력'],
    '주요업무': '',
    '자격요건': '',
    '우대사항': '',
    '혜택복지': '',
    '마감일': df_saramin['마감일'],
    '상태': 'unknown',
    '링크': df_saramin['링크'],
    '수집일시': df_saramin['수집일시'],
})

# ============================================================
# 3. 잡다임 데이터 로드
# ============================================================
print("\n[3] 잡다임 데이터 로드...")
df_jobda = pd.read_csv('jobda_it_20260518_161026.csv')
print(f"  → {len(df_jobda):,}개")

jobda_std = pd.DataFrame({
    '공고ID': 'JD_' + df_jobda.index.astype(str),
    '출처': '잡다임',
    '공고제목': df_jobda['공고제목'],
    '회사명': df_jobda['회사명'],
    '지역': df_jobda.get('추가정보', '').astype(str).str.split('|').str[0].str.strip(),
    '경력': df_jobda.get('추가정보', '').astype(str).str.split('|').str[1].str.strip() if '추가정보' in df_jobda.columns else '',
    '주요업무': '',
    '자격요건': '',
    '우대사항': '',
    '혜택복지': '',
    '마감일': '',
    '상태': 'unknown',
    '링크': df_jobda['url'],
    '수집일시': '',
})

# ============================================================
# 4. 통합
# ============================================================
print("\n[4] 전체 통합...")
df_all = pd.concat([wanted_std, saramin_std, jobda_std], ignore_index=True)
print(f"  통합 전: {len(df_all):,}개")

# ============================================================
# 5. 중복 제거 (회사명 + 제목 기반)
# ============================================================
print("\n[5] 회사명+제목 중복 제거...")
df_all['중복키'] = df_all['회사명'].astype(str).str.strip() + '|' + df_all['공고제목'].astype(str).str.strip()
before = len(df_all)
df_all = df_all.drop_duplicates(subset=['중복키'], keep='first')
df_all = df_all.drop('중복키', axis=1)
after = len(df_all)
print(f"  중복 제거: {before - after}개")
print(f"  최종: {len(df_all):,}개")

# ============================================================
# 6. 결측치 처리
# ============================================================
df_all = df_all.fillna('')

# ============================================================
# 7. 저장
# ============================================================
output_dir = 'data/processed'
os.makedirs(output_dir, exist_ok=True)

output_file = f'{output_dir}/integrated_jobs.csv'
df_all.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n[6] 저장: {output_file}")

# ============================================================
# 8. 요약 통계
# ============================================================
print("\n" + "="*70)
print("📊 통합 데이터 요약")
print("="*70)

print(f"\n총 공고: {len(df_all):,}개")

print(f"\n[출처별]")
for source, count in df_all['출처'].value_counts().items():
    pct = count / len(df_all) * 100
    bar = '█' * int(pct / 2)
    print(f"  {source:8s}: {count:6,}개 ({pct:5.1f}%) {bar}")

print(f"\n[지역별 Top 10]")
for region, count in df_all['지역'].value_counts().head(10).items():
    print(f"  {region:15s}: {count:5,}개")

print(f"\n[고유 회사 수] {df_all['회사명'].nunique():,}개")

# 텍스트 데이터 보유 비율
has_업무 = (df_all['주요업무'].str.len() > 10).sum()
has_자격 = (df_all['자격요건'].str.len() > 10).sum()

print(f"\n[텍스트 데이터 보유]")
print(f"  주요업무: {has_업무:,}개 ({has_업무/len(df_all)*100:.1f}%)")
print(f"  자격요건: {has_자격:,}개 ({has_자격/len(df_all)*100:.1f}%)")

print("\n✅ Step 1-1 완료!")
print("\n📌 다음 단계: 02_extract_dates.py 실행")
