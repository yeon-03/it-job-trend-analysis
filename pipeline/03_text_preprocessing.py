#!/usr/bin/env python3
"""
[Step 1-3] 텍스트 전처리
1. 특수문자/이모지 정리
2. 기술 스택 키워드 추출
3. 직무 카테고리 분류
4. KoNLPy 형태소 분석 (선택)

실행: python 03_text_preprocessing.py
"""

import pandas as pd
import numpy as np
import re
import json
import os

print("="*70)
print("📝 Step 1-3: 텍스트 전처리")
print("="*70)

# ============================================================
# 1. 데이터 로드
# ============================================================
print("\n[1] 데이터 로드...")
df = pd.read_csv('data/processed/dated_jobs.csv')
print(f"  → {len(df):,}개")

# ============================================================
# 2. 기술 스택 키워드 사전
# ============================================================
print("\n[2] 기술 스택 키워드 사전 구축...")

TECH_KEYWORDS = {
    # 언어
    "언어": [
        "Python", "Java", "JavaScript", "TypeScript", "Kotlin", "Swift",
        "Go", "Golang", "Rust", "C\\+\\+", "C#", "Ruby", "Scala", "PHP",
        "R언어", "MATLAB"
    ],
    # 프레임워크/라이브러리
    "프레임워크": [
        "Spring", "Spring Boot", "Django", "FastAPI", "Flask", "NestJS",
        "Express", "React", "Vue", "Angular", "Next\\.js", "Nuxt",
        "Flutter", "React Native", "PyTorch", "TensorFlow", "Keras",
        "Pandas", "Scikit-learn", "LangChain"
    ],
    # 인프라/클라우드
    "인프라": [
        "AWS", "GCP", "Azure", "Docker", "Kubernetes", "K8s", "Terraform",
        "Jenkins", "GitHub Actions", "CI/CD", "Linux", "Nginx", "Ansible"
    ],
    # 데이터베이스
    "데이터베이스": [
        "MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch",
        "Oracle", "MariaDB", "DynamoDB", "Cassandra", "SQLite", "Kafka",
        "RabbitMQ", "Airflow"
    ],
    # AI/ML
    "AI_ML": [
        "LLM", "RAG", "GPT", "BERT", "Transformer", "딥러닝", "머신러닝",
        "자연어처리", "NLP", "컴퓨터비전", "Computer Vision", "MLOps",
        "Fine-tuning", "Embedding", "Vector DB", "FAISS", "ChromaDB"
    ],
    # 협업/방법론
    "협업": [
        "Git", "GitHub", "GitLab", "Jira", "Confluence", "Agile", "Scrum",
        "REST API", "GraphQL", "gRPC", "MSA", "마이크로서비스"
    ]
}

# 사전 저장
os.makedirs('data/analysis', exist_ok=True)
with open('data/analysis/tech_keywords.json', 'w', encoding='utf-8') as f:
    json.dump(TECH_KEYWORDS, f, ensure_ascii=False, indent=2)

total_keywords = sum(len(v) for v in TECH_KEYWORDS.values())
print(f"  총 키워드: {total_keywords}개")
for category, keywords in TECH_KEYWORDS.items():
    print(f"  {category}: {len(keywords)}개")

# ============================================================
# 3. 텍스트 정제 함수
# ============================================================
print("\n[3] 텍스트 정제...")

def clean_text(text):
    """특수문자/이모지/불필요한 문자 제거"""
    if pd.isna(text) or text == '':
        return ''
    text = str(text)
    # 이모지 제거
    text = re.sub(r'[^\w\s가-힣a-zA-Z0-9\.\+\#\-\/]', ' ', text)
    # 반복 공백 제거
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['주요업무_clean'] = df['주요업무'].apply(clean_text)
df['자격요건_clean'] = df['자격요건'].apply(clean_text)
df['우대사항_clean'] = df['우대사항'].apply(clean_text)

# 통합 텍스트 (분석용)
df['텍스트_통합'] = (
    df['공고제목'].fillna('') + ' ' +
    df['주요업무_clean'] + ' ' +
    df['자격요건_clean'] + ' ' +
    df['우대사항_clean']
)

print(f"  텍스트 정제 완료: {len(df):,}개")

# ============================================================
# 4. 기술 스택 추출
# ============================================================
print("\n[4] 기술 스택 추출...")

def extract_tech(text):
    """텍스트에서 기술 스택 추출"""
    found = {}
    text_lower = str(text).lower()

    for category, keywords in TECH_KEYWORDS.items():
        found[category] = []
        for kw in keywords:
            # 대소문자 무시 검색
            pattern = re.compile(re.escape(kw).replace('\\\\', '\\'), re.IGNORECASE)
            if pattern.search(text):
                # 정규화된 이름으로 저장
                clean_kw = kw.replace('\\+\\+', '++').replace('\\.', '.').replace('\\+', '+')
                found[category].append(clean_kw)

    return found

# 각 공고의 기술 스택 추출
print("  추출 중... (시간 걸릴 수 있음)")

tech_results = []
for idx, row in df.iterrows():
    tech = extract_tech(row['텍스트_통합'])
    # 전체 기술 목록 (카테고리 무관)
    all_tech = []
    for kws in tech.values():
        all_tech.extend(kws)
    tech_results.append({
        'idx': idx,
        'tech_all': ', '.join(sorted(set(all_tech))),
        'tech_언어': ', '.join(tech.get('언어', [])),
        'tech_프레임워크': ', '.join(tech.get('프레임워크', [])),
        'tech_인프라': ', '.join(tech.get('인프라', [])),
        'tech_DB': ', '.join(tech.get('데이터베이스', [])),
        'tech_AI': ', '.join(tech.get('AI_ML', [])),
    })
    if (idx + 1) % 2000 == 0:
        print(f"    {idx+1:,}/{len(df):,} 처리 중...")

tech_df = pd.DataFrame(tech_results).set_index('idx')
df = df.join(tech_df)

has_tech = (df['tech_all'] != '').sum()
print(f"  기술 스택 추출: {has_tech:,}개 ({has_tech/len(df)*100:.1f}%)")

# ============================================================
# 5. 직무 카테고리 분류
# ============================================================
print("\n[5] 직무 카테고리 분류...")

JOB_CATEGORIES = {
    '백엔드': ['백엔드', 'Backend', 'back-end', 'Spring', 'Django', 'FastAPI',
               'Node', 'API', '서버 개발', '서버개발'],
    '프론트엔드': ['프론트엔드', 'Frontend', 'front-end', 'React', 'Vue', 'Angular',
                  'Next', 'UI개발', 'UI 개발', '웹 퍼블'],
    '풀스택': ['풀스택', 'Fullstack', 'Full-stack', '풀 스택'],
    '모바일': ['iOS', 'Android', '안드로이드', '모바일', 'Flutter', 'React Native'],
    '데이터': ['데이터 엔지니어', '데이터엔지니어', 'Data Engineer', '데이터 분석',
               'Data Analyst', '데이터 사이언티스트', 'Data Scientist', 'Airflow', 'ETL'],
    'AI_ML': ['머신러닝', '딥러닝', 'AI', 'ML Engineer', 'MLOps', 'LLM',
              '인공지능', 'NLP', 'Computer Vision', '모델링'],
    'DevOps': ['DevOps', 'SRE', 'Platform', '인프라', 'Infrastructure',
               'Kubernetes', 'Docker', 'CI/CD', '클라우드'],
    '보안': ['보안', 'Security', '정보보호', '취약점', '침해'],
    'QA': ['QA', 'Quality', '테스트', 'Test Engineer'],
    'PM_기획': ['PM', 'Product Manager', 'PO', '기획', '프로덕트'],
}

def classify_job(title, text):
    """직무 카테고리 분류"""
    combined = str(title) + ' ' + str(text)
    
    for category, keywords in JOB_CATEGORIES.items():
        for kw in keywords:
            if re.search(re.escape(kw), combined, re.IGNORECASE):
                return category
    return '기타'

df['직무카테고리'] = df.apply(
    lambda row: classify_job(row['공고제목'], row['텍스트_통합']),
    axis=1
)

print(f"\n  [직무 카테고리 분포]")
category_counts = df['직무카테고리'].value_counts()
for cat, count in category_counts.items():
    pct = count / len(df) * 100
    bar = '█' * int(pct / 2)
    print(f"  {cat:12s}: {count:5,}개 ({pct:5.1f}%) {bar}")

# ============================================================
# 6. 회사 규모 추정
# ============================================================
print("\n[6] 회사 규모 추정...")

# 대기업 리스트
대기업_LIST = [
    '삼성', 'LG', 'SK', '현대', '롯데', '한화', '포스코', 'GS', 'CJ',
    '카카오', '네이버', '라인', '쿠팡', '배달의민족', '토스', '당근',
    '넥슨', 'NC소프트', '크래프톤', '펄어비스',
    '하나금융', '신한', 'KB', '우리은행', 'IBK',
    '코웨이', 'KT', 'SKT', 'LGU+',
]

공기업_LIST = ['공사', '공단', '진흥원', '연구원', '기관', '국가', '한국전력', 'ETRI']

def classify_company_size(company_name):
    company = str(company_name)
    for big in 대기업_LIST:
        if big in company:
            return '대기업'
    for pub in 공기업_LIST:
        if pub in company:
            return '공공기관'
    return '스타트업/중소'

df['회사규모'] = df['회사명'].apply(classify_company_size)

print(f"\n  [회사 규모 분포]")
size_counts = df['회사규모'].value_counts()
for size, count in size_counts.items():
    pct = count / len(df) * 100
    print(f"  {size:15s}: {count:5,}개 ({pct:5.1f}%)")

# ============================================================
# 7. 저장
# ============================================================
print("\n[7] 저장...")

output_file = 'data/processed/cleaned_jobs.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"  저장: {output_file}")

# 기술 스택 통계 저장
print("\n  [기술 스택 Top 30]")
from collections import Counter

all_techs = []
for tech_str in df['tech_all']:
    if tech_str:
        all_techs.extend([t.strip() for t in tech_str.split(',') if t.strip()])

tech_counter = Counter(all_techs)
top_techs = tech_counter.most_common(30)

tech_stats = pd.DataFrame(top_techs, columns=['기술', '빈도'])
tech_stats['비율(%)'] = (tech_stats['빈도'] / len(df) * 100).round(1)
tech_stats.to_csv('data/analysis/tech_frequency.csv', index=False, encoding='utf-8-sig')

for tech, count in top_techs[:20]:
    pct = count / len(df) * 100
    bar = '█' * int(pct / 2)
    print(f"  {tech:20s}: {count:5,}개 ({pct:5.1f}%) {bar}")

print("\n" + "="*70)
print("✅ Step 1-3 완료!")
print("="*70)
print(f"""
  산출물:
  - data/processed/cleaned_jobs.csv  (전처리 완료 데이터)
  - data/analysis/tech_keywords.json (키워드 사전)
  - data/analysis/tech_frequency.csv (기술 스택 빈도)
""")
print("📌 다음 단계: 04_eda.py 실행")
