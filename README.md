# IT 채용 트렌드 분석 시스템

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white"/>&nbsp;
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat&logo=FastAPI&logoColor=white"/>&nbsp;
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white"/>&nbsp;
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=HuggingFace&logoColor=black"/>&nbsp;
  <img src="https://img.shields.io/badge/ChromaDB-FF6719?style=flat&logo=data:image/svg+xml;base64,&logoColor=white"/>&nbsp;
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white"/>&nbsp;
  <img src="https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white"/>&nbsp;
  <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=JavaScript&logoColor=black"/>
</p>

> 한성대학교 AI응용학과 | 자연어처리 수업 팀 프로젝트 | 4조

<br>

## 📌 프로젝트 개요

### 소개

취업 준비생은 어떤 기술을 공부해야 하는지, 언제 지원하는 것이 유리한지, 스타트업과 대기업은 무엇을 다르게 요구하는지 —  
이 모든 정보가 수천 개의 채용 공고에 흩어져 있어 직접 분석하기 어렵습니다.

본 프로젝트는 **국내 IT 채용 공고 18,225건을 수집·분석**하여 기술 스택 트렌드, 직무별 채용 패턴, 기업 규모별 요구사항을 시각화하고,  
RAG 챗봇과 Fine-tuned LLM을 통해 **맞춤형 커리어 로드맵**을 생성하는 대화형 대시보드 시스템입니다.

- **기간**: 2026.05 ~ 2026.06
- **데이터**: 원티드 16,839건 · 사람인 1,375건 · 잡다임 11건 (총 **18,225건**)
- **분석 기간**: 2024.08 ~ 2026.03 (월별 트렌드)

<br>

## 👥 팀원 및 역할

### 역할

| 이름 | 역할 및 담당 업무 |
|------|------|
| **강연경** | - 채용 공고 데이터 수집 및 전처리 (`01~03`) <br> - 기술 스택 트렌드 분석 TF-IDF 기반 (`05_trend_analysis.py`) <br> - RAG 기반 채용 상담 챗봇 구현 (`rag_core.py`, `06_rag_chatbot.py`) <br> - BERT 직무 클러스터링 GPU 학습 (`07_bert_clustering.py`) |
| **우민하** | - 회사 규모별 기술 요구사항 비교 분석 (`company_size.py`) <br> - Qwen2.5-0.5B 기반 QLoRA 4-bit 파인튜닝 (`fine_tuning.py`) <br> - Fine-tuned LLM 로드맵 생성 모듈 (`roadmap_visualization.py`) |
| **김도헌** | - 최적 지원 시기 예측 모듈 (`04_eda.py`, `05_trend_analysis.py`) <br> - 클러스터 분석 및 학습 자료 추천 시스템 (`08_cluster_analysis.py`) <br> - 전체 파이프라인 통합 및 웹 UI 구현 (`server.py`, `static/`) |

<br>

## 🏗️ 시스템 아키텍처

```
[원티드 / 사람인 / 잡다임]
         │
         ▼
┌─────────────────────────────────────────────────┐
│           데이터 수집 & 전처리 파이프라인            │
│                                                  │
│  01_integrate  →  02_extract_dates               │
│       →  03_text_preprocessing  →  04_eda        │
│                                                  │
│  · 중복 제거 및 소스 통합 (18,225건)               │
│  · 날짜 정규화 (regex + heuristic)                │
│  · 기술 키워드 추출 및 직무 분류 (6개 카테고리)      │
└──────────────────┬──────────────────────────────┘
                   │  cleaned_jobs.csv
                   ▼
┌─────────────────────────────────────────────────┐
│                  분석 & 모델링                    │
│                                                  │
│  [트렌드 분석]          [BERT 클러스터링]           │
│  05_trend_analysis ──▶ 월별 기술 증감률            │
│  TF-IDF 기반           상승: Linux·FastAPI·Redis  │
│                        하락: Java·TypeScript·AWS  │
│                                                  │
│  [RAG 챗봇]            [BERT 클러스터링]           │
│  ChromaDB              jhgan/ko-sroberta          │
│  임베딩 검색   ──────▶  + KMeans (k=9)            │
│  채용 상담 응답          UMAP 2D 시각화             │
│                                                  │
│  [LLM 파인튜닝]                                   │
│  Qwen2.5-0.5B + QLoRA ──▶ 커리어 로드맵 생성      │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│              FastAPI + Web 대시보드               │
│                                                  │
│  /api/job-monthly     → 직무별 월별 공고 수        │
│  /api/tech-growth     → 기술 스택 증감률           │
│  /api/company-size    → 기업 규모별 분석            │
│  /api/chat (POST)     → RAG 챗봇 응답             │
│  /api/roadmap (POST)  → LLM 로드맵 생성           │
└─────────────────────────────────────────────────┘
```

<br>

## 📁 파일 구조

```
📦 project/
├── 📂 파이프라인
│   ├── 01_integrate_data_v2.py      # 데이터 통합 (원티드·사람인·잡다임 → 18,225건)
│   ├── 02_extract_dates_v5.py       # 날짜 정보 추출 및 정규화
│   ├── 03_text_preprocessing.py     # 불용어 제거, 기술 키워드 추출, 직무 분류
│   ├── 04_eda.py                    # 탐색적 데이터 분석 및 시각화
│   ├── 05_trend_analysis.py         # 기술 스택 월별 트렌드, 상승/하락 기술 분석
│   ├── 06_rag_chatbot.py            # ChromaDB 벡터 DB 구축
│   ├── 07_bert_clustering.py        # ko-sroberta 임베딩 + KMeans 클러스터링
│   └── 08_cluster_analysis.py       # 클러스터별 특성 분석 및 시각화
│
├── 📂 모델 & 서비스
│   ├── rag_core.py                  # RAG 핵심 로직 (검색 + 응답 생성)
│   ├── fine_tuning.py               # Qwen2.5-0.5B QLoRA 파인튜닝
│   ├── roadmap_visualization.py     # 로드맵 생성 엔진 (LLM + matplotlib)
│   ├── company_size.py              # 기업 규모 분류 로직
│   └── server.py                    # FastAPI 서버 (REST API + 정적 파일 서빙)
│
├── 📂 static/                       # 프론트엔드 (Vanilla JS)
│   ├── index.html
│   ├── script.js
│   └── styles.css
│
├── 📂 data/
│   ├── processed/                   # 전처리 CSV (gitignore - 용량 초과)
│   │   ├── integrated_jobs.csv      # 통합 원본 (18,225건)
│   │   ├── dated_jobs.csv           # 날짜 추출 완료
│   │   └── cleaned_jobs.csv         # 최종 전처리 완료
│   └── analysis/                    # 분석 결과 CSV
│       ├── job_monthly.csv          # 직무별 월별 공고 수
│       ├── tech_growth.csv          # 기술 스택 증감률
│       ├── tech_monthly_pct.csv     # 기술 월별 점유율
│       ├── cluster_final_summary.csv # 클러스터별 요약
│       └── ...
│
├── 📂 models/
│   ├── trend_summary.json           # 트렌드 분석 결과 요약
│   ├── vector_db/                   # ChromaDB (gitignore - 119MB)
│   └── roadmap_llm/adapter/         # QLoRA 어댑터 (가중치 gitignore)
│
├── 📂 outputs/
│   ├── visualizations/              # 분석 시각화 이미지 (11종)
│   └── roadmaps/                    # 생성된 커리어 로드맵 이미지
│
├── requirements.txt
└── README.md
```

<br>

## ▶️ How to run

### 0. 환경 준비

```bash
pip install -r requirements.txt
```

> `bitsandbytes`는 CUDA GPU 환경(Linux/Windows)에서만 정상 동작합니다.  
> CPU 환경에서는 `fine_tuning.py`와 로드맵 생성 기능을 제외한 모든 기능이 동작합니다.

### 1. 데이터 파이프라인 (전체 처음부터 실행 시)

원본 CSV 파일을 `data/processed/`에 배치 후 순서대로 실행합니다.

```bash
python 01_integrate_data_v2.py   # 데이터 통합
python 02_extract_dates_v5.py    # 날짜 추출
python 03_text_preprocessing.py  # 텍스트 전처리
python 04_eda.py                 # EDA 및 시각화
python 05_trend_analysis.py      # 트렌드 분석
```

### 2. RAG 벡터 DB 구축

```bash
python 06_rag_chatbot.py --build
```

### 3. BERT 클러스터링 (GPU 권장)

```bash
python 07_bert_clustering.py
python 08_cluster_analysis.py
```

### 4. LLM 파인튜닝 (GPU 필수)

```bash
python fine_tuning.py
```

### 5. 서버 실행

```bash
uvicorn server:app --reload --port 8000
```

브라우저 접속:
- **대시보드**: http://localhost:8000/
- **API 문서**: http://localhost:8000/docs

<br>

## 📊 분석 결과

### 기술 스택 트렌드 (2024.08 ~ 2026.03)

전체 상위 기술 스택 (전체 공고 대비 출현율):

| 순위 | 기술 | 출현율 |
|:----:|------|:------:|
| 1 | AWS | 35.0% |
| 2 | Java | 34.5% |
| 3 | Git | 31.7% |
| 4 | Python | 29.8% |
| 5 | React | 24.4% |
| 6 | CI/CD | 21.7% |
| 7 | TypeScript | 20.4% |
| 8 | Go | 20.2% |
| 9 | Spring | 18.6% |
| 10 | Docker | 17.2% |

### 상승 / 하락 기술

| 방향 | 기술 | 증감률 |
|:----:|------|:------:|
| 📈 상승 | Linux | +26.2% |
| 📈 상승 | FastAPI | +23.3% |
| 📈 상승 | Redis | +18.1% |
| 📈 상승 | Python | +7.5% |
| 📈 상승 | LLM | +4.7% |
| 📉 하락 | Java | -6.5% |
| 📉 하락 | TypeScript | -5.7% |
| 📉 하락 | AWS | -5.3% |

### 기업 규모별 차이

| 항목 | 스타트업/중소 | 대기업 |
|------|:------------:|:------:|
| 1위 기술 | AWS (30.5%) | Java (38.5%) |
| 2위 기술 | Java (29.0%) | AWS (25.4%) |
| 특징적 기술 | TypeScript (+12.2%p↑) | Java (+9.5%p↑) |
| 풀스택 선호도 | 2.1% | 0.3% |
| AI/ML 공고 비율 | 11.3% | 9.6% |
| DevOps 공고 비율 | 7.1% | 15.9% |

### 직무별 채용 피크 시기

| 직무 | 채용 피크 |
|------|:--------:|
| 백엔드 | 2025년 10월 |
| DevOps | 2025년 10월 |
| 데이터 | 2025년 10월 |
| 모바일 | 2025년 10월 |
| 프론트엔드 | 2025년 8월 |
| AI/ML | 2026년 3월 |

### BERT 클러스터링 결과

`jhgan/ko-sroberta-multitask` 임베딩 + KMeans(k=9) 클러스터링으로 도출된 직무 그룹:

| 클러스터 | 공고 수 | 주요 직무 | 핵심 기술 |
|:--------:|:-------:|----------|----------|
| 0 | 1,780 | 백엔드 | Java, AWS, Spring, Git, MySQL |
| 1 | 1,179 | 백엔드 | React, TypeScript, Java, Git, JavaScript |
| 2 | 868 | DevOps | AWS, CI/CD, Kubernetes, Linux, Python |
| 3 | 1,026 | 백엔드 | AWS, Java, React, TypeScript, Git |
| 4 | 906 | 백엔드 | Python, LLM, PyTorch, AWS, RAG |
| 5 | 492 | 백엔드 | Git, Kotlin, React, Swift, Java |
| 6 | 955 | 백엔드 | Python, Git, Linux, Java, C# |
| 7 | 597 | 데이터 | Python, AWS, Airflow, Java, GCP |
| 8 | 508 | DevOps | AWS, Python, GCP, Azure, Go |

<br>

## 📂 주요 데이터 파일

| 파일 | 내용 | 비고 |
|------|------|:----:|
| `data/processed/integrated_jobs.csv` | 통합 원본 채용 공고 | 18,225건 |
| `data/processed/cleaned_jobs.csv` | 전처리 완료 데이터 | gitignore |
| `data/analysis/job_monthly.csv` | 직무별 월별 공고 수 | 6개 직무 |
| `data/analysis/tech_growth.csv` | 기술 스택 증감률 | 상승/하락 분석 |
| `data/analysis/tech_monthly_pct.csv` | 기술 월별 점유율 | 시계열 |
| `data/analysis/cluster_final_summary.csv` | 클러스터별 요약 | k=9 |
| `models/trend_summary.json` | 트렌드 분석 결과 | 기업 규모별 |
| `models/roadmap_llm/adapter/` | QLoRA 어댑터 | Qwen2.5-0.5B 기반 |
| `outputs/visualizations/` | 분석 시각화 이미지 | 11종 PNG |
| `outputs/roadmaps/` | 생성된 로드맵 이미지 | 직무별 3종 |
