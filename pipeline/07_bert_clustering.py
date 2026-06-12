#!/usr/bin/env python3
"""
[Phase 4] BERT 직무 클러스터링
- KoBERT/KcBERT 임베딩 생성
- K-means 클러스터링
- 클러스터 분석 + 시각화

GPU 서버에서 실행 (목요일 4:30~6:00)

설치:
  pip install torch transformers scikit-learn umap-learn matplotlib pandas

실행:
  python 07_bert_clustering.py
"""

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime

print("="*70)
print("🧠 Phase 4: BERT 직무 클러스터링")
print("="*70)

# ============================================================
# 0. GPU 환경 확인
# ============================================================
print("\n[0] 환경 확인...")
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  Device: {device}")

if device == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  ⚠️ GPU 없음 - CPU로 실행 (매우 느림)")
    print("  GPU 서버에서 실행하세요!")

# ============================================================
# 1. 데이터 로드
# ============================================================
print("\n[1] 데이터 로드...")
df = pd.read_csv('data/processed/cleaned_jobs.csv')

# 텍스트가 있는 데이터만 사용
df_bert = df[
    (df['주요업무_clean'].astype(str).str.len() > 30) |
    (df['자격요건_clean'].astype(str).str.len() > 30)
].copy().reset_index(drop=True)

print(f"  전체: {len(df):,}개")
print(f"  BERT 대상: {len(df_bert):,}개")

# 입력 텍스트 생성 (제목 + 주요업무 + 자격요건)
df_bert['bert_input'] = (
    df_bert['공고제목'].fillna('') + ' ' +
    df_bert['주요업무_clean'].fillna('').str[:200] + ' ' +
    df_bert['자격요건_clean'].fillna('').str[:200]
)

texts = df_bert['bert_input'].tolist()
print(f"  입력 텍스트 준비 완료")

# ============================================================
# 2. BERT 임베딩 생성
# ============================================================
print("\n[2] BERT 임베딩 생성...")
print("  모델: jhgan/ko-sroberta-multitask (한국어 특화)")

from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "jhgan/ko-sroberta-multitask"

print(f"  모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()

print(f"  모델 로드 완료!")

def mean_pooling(model_output, attention_mask):
    """Mean Pooling - 토큰 임베딩 평균"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

def get_embeddings(texts, batch_size=32):
    """텍스트 리스트 → 임베딩 배열"""
    all_embeddings = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)

        embeddings = mean_pooling(output, encoded['attention_mask'])

        # L2 정규화
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu().numpy())

        if (i // batch_size + 1) % 20 == 0:
            done = min(i + batch_size, total)
            pct = done / total * 100
            print(f"  {done:,}/{total:,} ({pct:.1f}%) 완료...")

    return np.vstack(all_embeddings)

start_time = time.time()
print(f"\n  임베딩 생성 시작...")
embeddings = get_embeddings(texts, batch_size=32)

elapsed = time.time() - start_time
print(f"\n  임베딩 완료!")
print(f"  shape: {embeddings.shape}")
print(f"  소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")

# 저장 (중간 저장 - 혹시 이후 실패해도 다시 안 해도 됨)
os.makedirs('models', exist_ok=True)
np.save('models/bert_embeddings.npy', embeddings)
print(f"  저장: models/bert_embeddings.npy")

# ============================================================
# 3. 최적 클러스터 수 결정 (Elbow + Silhouette)
# ============================================================
print("\n[3] 최적 클러스터 수 결정...")

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

# 클러스터 수 범위
K_RANGE = [5, 7, 8, 9, 10, 12, 15]

inertias = []
silhouettes = []

print(f"  k값별 Silhouette Score 계산 중...")
for k in K_RANGE:
    # MiniBatchKMeans: 빠른 대용량 처리
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
    labels = kmeans.fit_predict(embeddings)

    # Silhouette는 샘플링 (전체는 너무 느림)
    sample_idx = np.random.choice(len(embeddings), min(2000, len(embeddings)), replace=False)
    sil = silhouette_score(embeddings[sample_idx], labels[sample_idx])
    inertias.append(kmeans.inertia_)
    silhouettes.append(sil)

    print(f"  k={k:2d}: Silhouette={sil:.4f}, Inertia={kmeans.inertia_:.0f}")

# 최적 k 선택 (Silhouette 최대)
best_k_idx = np.argmax(silhouettes)
best_k = K_RANGE[best_k_idx]
print(f"\n  최적 클러스터 수: k={best_k} (Silhouette={silhouettes[best_k_idx]:.4f})")

# ============================================================
# 4. 최종 클러스터링
# ============================================================
print(f"\n[4] 최종 클러스터링 (k={best_k})...")

kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(embeddings)

df_bert['cluster'] = cluster_labels
print(f"  클러스터링 완료!")

# ============================================================
# 5. 클러스터별 분석
# ============================================================
print("\n[5] 클러스터별 분석...")

os.makedirs('data/analysis', exist_ok=True)
cluster_summary = []

print(f"\n{'클러스터':>5} {'공고수':>6} {'주요 직무':>12} {'주요 기술'}")
print("-"*70)

for c in range(best_k):
    cluster_df = df_bert[df_bert['cluster'] == c]
    count = len(cluster_df)

    # 주요 직무
    top_job = cluster_df['직무카테고리'].value_counts().index[0] if len(cluster_df) > 0 else '기타'
    job_pct = cluster_df['직무카테고리'].value_counts().iloc[0] / count * 100

    # 주요 기술 (상위 5개)
    from collections import Counter
    all_techs = []
    for tech_str in cluster_df['tech_all'].dropna():
        if tech_str and tech_str != 'nan':
            all_techs.extend([t.strip() for t in str(tech_str).split(',') if t.strip()])
    top_techs = [t for t, _ in Counter(all_techs).most_common(5)]

    # 주요 회사 규모
    top_size = cluster_df['회사규모'].value_counts().index[0] if len(cluster_df) > 0 else '기타'

    summary = {
        'cluster': c,
        'count': count,
        'top_job': top_job,
        'job_pct': round(job_pct, 1),
        'top_techs': ', '.join(top_techs),
        'top_size': top_size,
    }
    cluster_summary.append(summary)

    print(f"  C{c:02d}   {count:5,}개  {top_job:12s}  {', '.join(top_techs[:4])}")

# ============================================================
# 6. 저장
# ============================================================
print("\n[6] 결과 저장...")

# 클러스터 레이블 저장
df_bert[['공고ID', '공고제목', '회사명', '직무카테고리', '회사규모',
          'tech_all', '게시년월', 'cluster']].to_csv(
    'data/analysis/clusters.csv', index=False, encoding='utf-8-sig'
)
print(f"  저장: data/analysis/clusters.csv")

# 클러스터 요약 저장
pd.DataFrame(cluster_summary).to_csv(
    'data/analysis/cluster_summary.csv', index=False, encoding='utf-8-sig'
)
print(f"  저장: data/analysis/cluster_summary.csv")

# ============================================================
# 7. 시각화 (UMAP → 2D)
# ============================================================
print("\n[7] 시각화 (UMAP 차원 축소)...")
print("  (수분 소요될 수 있음)")

try:
    import umap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import platform
    if platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    # UMAP 차원 축소 (샘플링)
    sample_size = min(3000, len(embeddings))
    sample_idx = np.random.choice(len(embeddings), sample_size, replace=False)
    sample_emb = embeddings[sample_idx]
    sample_labels = cluster_labels[sample_idx]
    sample_jobs = df_bert['직무카테고리'].iloc[sample_idx].values

    print(f"  UMAP 실행 중 (샘플 {sample_size:,}개)...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
    embedding_2d = reducer.fit_transform(sample_emb)

    # 그래프 그리기
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 클러스터 기준
    colors1 = plt.cm.tab20(np.linspace(0, 1, best_k))
    for c in range(best_k):
        mask = sample_labels == c
        ax1.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                    c=[colors1[c]], label=f'C{c:02d}', alpha=0.5, s=10)
    ax1.set_title(f'BERT 임베딩 클러스터 (k={best_k})', fontsize=13, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)

    # 직무 카테고리 기준
    job_categories = list(set(sample_jobs))
    colors2 = plt.cm.Set1(np.linspace(0, 1, len(job_categories)))
    for job, color in zip(job_categories, colors2):
        mask = sample_jobs == job
        ax2.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                    c=[color], label=job, alpha=0.5, s=10)
    ax2.set_title('직무 카테고리 분포', fontsize=13, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)

    plt.suptitle('BERT 임베딩 2D 시각화 (UMAP)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs('outputs/visualizations', exist_ok=True)
    plt.savefig('outputs/visualizations/09_bert_umap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  저장: outputs/visualizations/09_bert_umap.png")

except ImportError:
    print("  ⚠️ umap-learn 미설치, 시각화 건너뜀")
    print("  pip install umap-learn")

# ============================================================
# 8. 완료 요약
# ============================================================
total_elapsed = time.time() - start_time
print("\n" + "="*70)
print("✅ Phase 4 완료!")
print("="*70)
print(f"""
  소요 시간: {total_elapsed/60:.1f}분
  임베딩: {embeddings.shape} 저장 완료
  클러스터 수: {best_k}개

  산출물:
  - models/bert_embeddings.npy          (BERT 임베딩)
  - data/analysis/clusters.csv          (클러스터 레이블)
  - data/analysis/cluster_summary.csv   (클러스터 요약)
  - outputs/visualizations/09_bert_umap.png (시각화)
""")
print("📌 다음: GPU 없이 클러스터 분석 가능!")
print("  python 08_cluster_analysis.py")
