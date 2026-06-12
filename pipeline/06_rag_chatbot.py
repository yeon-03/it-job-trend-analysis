#!/usr/bin/env python3
"""
[Phase 3] RAG 기반 채용 상담 챗봇

실행:
  Step 1 (벡터 DB 구축): python 06_rag_chatbot.py --build
  Step 2 (챗봇 실행):    streamlit run 06_rag_chatbot.py

설치:
  pip install chromadb sentence-transformers streamlit
"""

import argparse
import os
import sys
import json
import pandas as pd

# ============================================================
# 인자 파싱
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--build', action='store_true', help='벡터 DB 구축')
args, unknown = parser.parse_known_args()

# ============================================================
# 설정
# ============================================================
VECTOR_DB_PATH = "models/vector_db"
COLLECTION_NAME = "job_postings"
EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"

os.makedirs(VECTOR_DB_PATH, exist_ok=True)
os.makedirs("models", exist_ok=True)

# ============================================================
# [BUILD 모드] 벡터 DB 구축
# ============================================================
def build_vector_db():
    print("="*70)
    print("🔨 벡터 DB 구축")
    print("="*70)

    print("\n[1] 데이터 로드...")
    df = pd.read_csv('data/processed/cleaned_jobs.csv')
    df_rag = df[
        (df['주요업무_clean'].astype(str).str.len() > 50) |
        (df['자격요건_clean'].astype(str).str.len() > 50)
    ].copy()
    print(f"  전체: {len(df):,}개")
    print(f"  RAG 대상: {len(df_rag):,}개")

    print("\n[2] 문서 생성...")
    documents, metadatas, ids = [], [], []

    for idx, row in df_rag.iterrows():
        doc_text = f"""[공고제목] {row['공고제목']}
[회사] {row['회사명']} ({row.get('회사규모', '')})
[지역] {row['지역']}
[직무] {row.get('직무카테고리', '')}
[기술스택] {row.get('tech_all', '')}
[주요업무] {str(row['주요업무_clean'])[:500]}
[자격요건] {str(row['자격요건_clean'])[:500]}
[우대사항] {str(row['우대사항_clean'])[:300]}"""

        metadata = {
            'title': str(row['공고제목'])[:100],
            'company': str(row['회사명'])[:50],
            'company_size': str(row.get('회사규모', '')),
            'region': str(row['지역'])[:30],
            'job_category': str(row.get('직무카테고리', '')),
            'tech_stack': str(row.get('tech_all', ''))[:200],
            'source': str(row['출처']),
            'year_month': str(row.get('게시년월', '')),
        }

        documents.append(doc_text)
        metadatas.append(metadata)
        ids.append(f"job_{idx}")

        if len(documents) % 1000 == 0:
            print(f"  {len(documents):,}개 처리 중...")

    print(f"  총 {len(documents):,}개 문서 생성")

    print("\n[3] ChromaDB 저장...")
    print(f"  임베딩 모델: {EMBEDDING_MODEL}")
    print("  (첫 실행 시 모델 다운로드 - 수분 소요)")

    try:
        import chromadb
        from chromadb.utils import embedding_functions

        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

        try:
            client.delete_collection(COLLECTION_NAME)
            print("  기존 컬렉션 삭제")
        except:
            pass

        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"}
        )

        BATCH_SIZE = 100
        for i in range(0, len(documents), BATCH_SIZE):
            collection.add(
                documents=documents[i:i+BATCH_SIZE],
                metadatas=metadatas[i:i+BATCH_SIZE],
                ids=ids[i:i+BATCH_SIZE]
            )
            if (i // BATCH_SIZE + 1) % 20 == 0:
                print(f"  {min(i+BATCH_SIZE, len(documents)):,}/{len(documents):,} 저장...")

        print(f"\n  ✅ 벡터 DB 구축 완료! 총 {collection.count():,}개")

    except ImportError as e:
        print(f"\n  ❌ 패키지 미설치: {e}")
        sys.exit(1)

    print("\n[4] 트렌드 데이터 저장...")
    try:
        tech_growth = pd.read_csv('data/analysis/tech_growth.csv')
        job_monthly = pd.read_csv('data/analysis/job_monthly.csv', index_col=0)

        trend_summary = {
            "rising_tech": tech_growth.nlargest(5, '증가율(%)')[
                ['기술', '증가율(%)', '전체평균(%)']].to_dict('records'),
            "falling_tech": tech_growth.nsmallest(3, '증가율(%)')[
                ['기술', '증가율(%)', '전체평균(%)']].to_dict('records'),
            "top_tech": tech_growth.nlargest(10, '전체평균(%)')[
                ['기술', '전체평균(%)']].to_dict('records'),
            "job_peak": {
                job: str(job_monthly[job].idxmax())
                for job in job_monthly.columns
            }
        }

        with open('models/trend_summary.json', 'w', encoding='utf-8') as f:
            json.dump(trend_summary, f, ensure_ascii=False, indent=2)
        print("  저장: models/trend_summary.json")
    except Exception as e:
        print(f"  ⚠️ 트렌드 데이터 오류: {e}")

    print("\n✅ 벡터 DB 구축 완료!")
    print("📌 다음 실행: streamlit run 06_rag_chatbot.py")


# ============================================================
# RAG 검색 함수
# ============================================================

def load_cluster_data():
    """BERT 클러스터 데이터 로드"""
    try:
        clusters = pd.read_csv('data/analysis/clusters.csv')
        summary = pd.read_csv('data/analysis/cluster_final_summary.csv')
        return clusters, summary
    except:
        return None, None


# 클러스터별 설명 (분석 결과 기반)
CLUSTER_INFO = {
    0: {'이름': '전통 백엔드',      '핵심기술': 'Java 64%, Spring 52%, AWS 57%',   '특징': '대기업/금융권에 많음'},
    1: {'이름': '프론트엔드',       '핵심기술': 'React 82%, TypeScript 65%, Git 53%', '특징': 'React 없으면 사실상 불가'},
    2: {'이름': 'DevOps/인프라',    '핵심기술': 'AWS 52%, Kubernetes 40%, Docker 27%', '특징': '클라우드 네이티브 중심'},
    3: {'이름': '풀스택형 백엔드',  '핵심기술': 'AWS 34%, Java 33%, React 29%',    '특징': '스타트업 선호'},
    4: {'이름': 'AI/ML',           '핵심기술': 'Python 59%, LLM 50%, AWS 26%',    '특징': '가장 빠르게 성장 중'},
    5: {'이름': '모바일',           '핵심기술': 'React 33%, Git 39%, Kotlin/Swift', '특징': '모바일 앱 개발'},
    6: {'이름': '시스템/범용',      '핵심기술': 'Python 32%, Java 17%, Linux 23%', '특징': '다양한 기술 혼합'},
    7: {'이름': '데이터 엔지니어링','핵심기술': 'Python 62%, AWS 45%, Airflow',    '특징': '데이터 파이프라인 전문'},
    8: {'이름': '클라우드',         '핵심기술': 'AWS 37%, Python 24%, GCP/Azure',  '특징': '멀티클라우드 수요'},
}

# 직무 → 관련 클러스터 매핑
JOB_TO_CLUSTERS = {
    '백엔드':    [0, 3],
    '프론트엔드': [1],
    'AI_ML':    [4],
    'DevOps':   [2, 8],
    '데이터':    [7],
    '모바일':    [5],
}

def load_vector_db():
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    except:
        return None


def get_rag_response(query: str, collection) -> str:
    trend_data = {}
    try:
        with open('models/trend_summary.json', 'r', encoding='utf-8') as f:
            trend_data = json.load(f)
    except:
        pass

    query_lower = query.lower()

    job_map = {
        '백엔드': ['백엔드', 'backend', '서버'],
        '프론트엔드': ['프론트엔드', 'frontend', '프론트', 'ui'],
        'AI_ML': ['ai', '인공지능', '머신러닝', '딥러닝', 'ml', 'llm'],
        'DevOps': ['devops', 'sre', '인프라', '클라우드'],
        '데이터': ['데이터', 'data'],
        '모바일': ['ios', 'android', '모바일', '앱'],
    }

    detected_job = None
    for job, keywords in job_map.items():
        if any(kw in query_lower for kw in keywords):
            detected_job = job
            break

    relevant_docs, relevant_meta = [], []
    if collection:
        try:
            results = collection.query(
                query_texts=[query], n_results=5,
                where={"job_category": detected_job} if detected_job else None
            )
            if results['documents']:
                relevant_docs = results['documents'][0]
                relevant_meta = results['metadatas'][0]
        except:
            try:
                results = collection.query(query_texts=[query], n_results=5)
                if results['documents']:
                    relevant_docs = results['documents'][0]
                    relevant_meta = results['metadatas'][0]
            except:
                pass

    parts = []

    if any(kw in query_lower for kw in ['공부', '배워', '기술', '스택', '필요', '준비']):
        job_tech = {
            '백엔드':     ('Java(48%), AWS(43%), Spring(32%), Python(32%)', 'Docker, Kubernetes, MySQL, Redis'),
            '프론트엔드': ('React(60%), TypeScript(42%), Git(30%)', 'Next.js, Vue, CSS/HTML'),
            'AI_ML':      ('Python(34%), PyTorch/TensorFlow, LLM(16%)', 'RAG, LangChain, Hugging Face'),
            'DevOps':     ('AWS(43%), Kubernetes(20%), Docker, Linux', 'Terraform, CI/CD, GitHub Actions'),
            '데이터':     ('Python(49%), AWS(27%), SQL, Airflow', 'Spark, Kafka, dbt'),
            '모바일':     ('Swift(iOS), Kotlin(Android), Git(20%)', 'Flutter, React Native'),
        }
        if detected_job and detected_job in job_tech:
            core, plus = job_tech[detected_job]
            parts.append(f"## {detected_job} 개발자 필수 기술스택\n")
            parts.append(f"**핵심 기술** (채용공고 기준 비율):\n{core}\n")
            parts.append(f"**추가 우대**:\n{plus}\n")

        # BERT 클러스터 기반 세분화 정보 추가
        if detected_job and detected_job in JOB_TO_CLUSTERS:
            related_clusters = JOB_TO_CLUSTERS[detected_job]
            parts.append(f"\n## BERT 분석 기반 세부 유형 (9,714개 공고 클러스터링 결과)\n")
            for c_id in related_clusters:
                info = CLUSTER_INFO[c_id]
                # 해당 클러스터 공고 수
                count_map = {0:1780, 1:1179, 2:868, 3:1026, 4:906, 5:492, 6:955, 7:597, 8:508}
                count = count_map.get(c_id, 0)
                parts.append(f"**[유형 {c_id+1}] {info['이름']}** ({count:,}개 공고)")
                parts.append(f"- 핵심 기술: {info['핵심기술']}")
                parts.append(f"- 특징: {info['특징']}\n")

        if trend_data.get('rising_tech'):
            parts.append("\n**지금 급상승 중인 기술** (원티드·사람인 9,714개 공고 기준 | 2025년 3분기→4분기 비교):")
            
            # 절대 수치 포함
            tech_detail = {
                'Linux':   ('183개→366개', '+26.2%'),
                'FastAPI': ('76개→150개',  '+23.3%'),
                'Redis':   ('179개→335개', '+18.1%'),
                'Python':  ('505개→866개', '+7.5%'),
                'LLM':     ('179개→300개', '+4.7%'),
                'Git':     ('574개→935개', '+2.5%'),
                'GCP':     ('221개→361개', '+2.2%'),
            }
            for item in trend_data['rising_tech'][:3]:
                tech = item['기술']
                if tech in tech_detail:
                    count_str, growth_str = tech_detail[tech]
                    parts.append(f"- **{tech}**: {count_str} 공고 ({growth_str})")
                else:
                    parts.append(f"- **{tech}** (+{item['증가율(%)']}%)")

    if any(kw in query_lower for kw in ['언제', '시기', '시즌', '지원']):
        parts.append("\n## 최적 지원 시기\n")
        parts.append("> 9,714개 채용공고 월별 분포 분석 기준 (2025.07~2026.03)\n")

        season_detail = {
            '백엔드':    ('10월', '8월', 581, 485, 19.8),
            '프론트엔드': ('8월',  '6월', 91,  67,  35.8),
            'AI_ML':    ('3월',  '1월', 109, 87,  25.3),
            'DevOps':   ('10월', '8월', 114, 68,  67.6),
            '데이터':    ('10월', '8월', 63,  51,  23.5),
            '모바일':    ('10월', '8월', 46,  30,  53.3),
        }

        if detected_job and detected_job in season_detail:
            peak, prep, peak_cnt, avg_cnt, diff_pct = season_detail[detected_job]
            parts.append(f"**{detected_job}** 채용 데이터:")
            parts.append(f"- 피크 시즌: **{peak}** ({peak_cnt}개 공고, 월평균 {avg_cnt}개 대비 **+{diff_pct}%**)")
            parts.append(f"- 준비 시작 권장: **{prep}부터** (서류·포트폴리오 완성 목표)")
            parts.append(f"\n> 피크 시즌에 채용 공고가 월평균보다 {diff_pct}% 많으므로,")
            parts.append(f"> 경쟁도 높아집니다. {prep}부터 미리 준비하는 것을 권장합니다.")
        else:
            parts.append("직무별 채용 피크 시즌 (월평균 대비 증가율):")
            parts.append("- **백엔드**: 10월 피크 (581개, +19.8%) → 8월부터 준비")
            parts.append("- **프론트엔드**: 8월 피크 (91개, +35.8%) → 6월부터 준비")
            parts.append("- **AI/ML**: 3월 피크 (109개, +25.3%) → 1월부터 준비")
            parts.append("- **DevOps**: 10월 피크 (114개, +67.6%) → 8월부터 준비")
            parts.append("- **데이터**: 10월 피크 (63개, +23.5%) → 8월부터 준비")
            parts.append("- **모바일**: 10월 피크 (46개, +53.3%) → 8월부터 준비")

    if any(kw in query_lower for kw in ['트렌드', '뜨는', '핫', '요즘', '최신']):
        parts.append("\n## 2025-2026 기술 트렌드\n")
        parts.append("> 원티드·사람인 채용공고 9,714개 분석 | 2025년 3분기(7~9월) vs 4분기(10~12월) 비교\n")
        parts.append("**급상승 기술:**")
        parts.append("- **Linux**: 183개 → 366개 공고 (+26.2%) — 클라우드 서버 필수화")
        parts.append("- **FastAPI**: 76개 → 150개 공고 (+23.3%) — Python 백엔드 대세")
        parts.append("- **Redis**: 179개 → 335개 공고 (+18.1%) — 실시간 처리 수요 증가")
        parts.append("- **LLM**: 179개 → 300개 공고 (+4.7%) — 전체 공고의 10% 요구\n")
        parts.append("**비중 감소 (여전히 중요):**")
        parts.append("- **Java**: 여전히 1위(30%)지만 -6.5% 감소 추세")
        parts.append("- **TypeScript**: -5.7% | **AWS**: -5.3% (포화 상태)")

    if any(kw in query_lower for kw in ['대기업', '스타트업', '규모', '차이']):
        parts.append("\n## 회사 규모별 기술 스택 차이\n")
        parts.append("**대기업**: Java 38.5%, Spring 22.5% 압도적 (안정적 기술 선호)")
        parts.append("**스타트업**: AWS 30.6%, React 22.5%, LLM 더 많이 요구 (최신 기술 적극 도입)")

    if relevant_docs:
        parts.append("\n---\n**관련 채용공고 샘플:**")
        for i, (doc, meta) in enumerate(zip(relevant_docs[:3], relevant_meta[:3]), 1):
            tech = meta.get('tech_stack', '')
            # nan 제거
            if tech and tech.lower() != 'nan':
                tech = tech[:80]
            else:
                tech = ''
            title = meta.get('title', '')
            company = meta.get('company', '')
            parts.append(f"\n**{i}. {title}** - {company}")
            if tech:
                parts.append(f"   기술: {tech}")

    if not parts:
        parts.append("## IT 채용 시장 핵심 정보\n")
        parts.append("**기술 Top 5**: Java(30%) > AWS(29.9%) > Python(25.5%) > React(21.2%) > Docker(14.7%)")
        parts.append("\n어떤 직무를 준비 중인지 알려주시면 더 구체적으로 안내해드릴게요!")

    # 모든 응답 끝에 출처 각주
    parts.append("\n\n---")
    parts.append("*데이터 출처: 원티드·사람인·잡다임 채용공고 9,714개 | 수집 기간: 2025.07~2026.03*")

    return "\n".join(parts)


# ============================================================
# [STREAMLIT 모드]
# ============================================================
def run_chatbot():
    import streamlit as st

    load_db_cached = st.cache_resource(load_vector_db)

    st.set_page_config(page_title="IT 채용 트렌드 챗봇", page_icon="💼", layout="wide")
    st.title("💼 IT 채용 트렌드 상담 챗봇")
    st.caption("9,714개 채용공고 데이터 기반")

    collection = load_db_cached()
    if collection:
        st.success(f"✅ 벡터 DB 로드 완료 ({collection.count():,}개 문서)")
    else:
        st.warning("⚠️ 벡터 DB 없음 - 먼저 `python 06_rag_chatbot.py --build` 실행하세요")

    with st.sidebar:
        st.header("📊 빠른 통계")
        st.metric("분석 공고 수", "9,714개")
        st.metric("분석 기간", "2025.07~2026.03")
        st.metric("수집 회사", "3,355개")
        st.divider()
        st.subheader("🔥 TOP 5 기술")
        for i, t in enumerate(['Java', 'AWS', 'Git', 'Python', 'React'], 1):
            st.write(f"{i}. {t}")
        st.divider()
        st.subheader("💬 예시 질문")
        for q in [
            "백엔드 개발자 뭘 공부해야 해?",
            "AI/ML 엔지니어 필요 기술은?",
            "요즘 가장 뜨는 기술은?",
            "백엔드 언제 지원해야 해?",
            "스타트업 vs 대기업 차이는?",
        ]:
            if st.button(q, use_container_width=True):
                st.session_state.pending_input = q

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "안녕하세요! IT 채용 트렌드 상담 챗봇입니다.\n\n**9,714개의 채용공고** 데이터를 기반으로 답변해드립니다.\n\n어떤 직무를 준비 중이신가요? 😊"
        }]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("예: 백엔드 개발자 준비 중인데 뭘 공부해야 해?")
    if "pending_input" in st.session_state and st.session_state.pending_input:
        user_input = st.session_state.pending_input
        st.session_state.pending_input = ""

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("채용공고 분석 중..."):
                response = get_rag_response(user_input, collection)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


# ============================================================
# 메인
# ============================================================
if args.build:
    build_vector_db()
else:
    run_chatbot()