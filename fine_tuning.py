#!/usr/bin/env python3
"""
[2번 담당] fine_tuning.py
company_comparison 결과를 활용한 파인튜닝 데이터 생성 +
Llama-3.2-Korean-GGACHI-1B-Instruct-v1 QLoRA 파인튜닝

파이프라인
----------
1. CompanyComparison metrics + 원시 채용공고 → 파인튜닝 데이터셋 생성
2. QLoRA (4-bit) 로 학습
3. 학습된 adapter 저장 → 3번 담당자가 사용할 수 있는 로드맵 추론 스크립트 포함

실행: python fine_tuning.py
     또는 train.py 에서 자동 호출

환경 요구사항
-----------
- CUDA 12.1 + RTX 4060 (8 GB VRAM)
- pip install transformers datasets peft trl bitsandbytes accelerate
"""

from __future__ import annotations

import os
import sys
import json
import random
import re
import warnings
from pathlib import Path
from typing import Optional, List

import pandas as pd

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════
# 설정
# ════════════════════════════════════════════════════════════════════════════

MODEL_ID  = "Qwen/Qwen2.5-0.5B-Instruct"
OUT_DATA  = "data/finetune"
OUT_MODEL = "models/roadmap_llm"
OUT_METRICS = "outputs/metrics"
ADAPTER_SAVE = os.path.join(OUT_MODEL, "adapter")

# RTX 4060 8GB VRAM 기준 안전한 배치 설정
TRAIN_CFG = {
    "max_seq_length":      1024,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,   # 유효 배치 = 16
    "num_train_epochs":    3,
    "learning_rate":       2e-4,
    "warmup_ratio":        0.05,
    "lr_scheduler_type":   "cosine",
    "logging_steps":       10,
    "save_steps":          50,
    "fp16":                True,        # CUDA 12.1, RTX 4060
    "optim":               "paged_adamw_8bit",
    "gradient_checkpointing": True,
}

LORA_CFG = {
    "r":            16,
    "lora_alpha":   32,
    "lora_dropout": 0.05,
    "bias":         "none",
    "task_type":    "CAUSAL_LM",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
}


# ════════════════════════════════════════════════════════════════════════════
# 데이터셋 생성기
# ════════════════════════════════════════════════════════════════════════════

class RoadmapDatasetBuilder:
    """
    채용공고 데이터 + CompanyComparison metrics →
    로드맵 생성용 instruction-tuning 데이터셋 빌더.

    생성되는 샘플 유형
    ------------------
    A. 규모/직무 기반 로드맵 질문 → 단계별 학습 로드맵 답변
    B. 기술 선택 질문 → 시장 데이터 기반 우선순위 답변
    C. 비교 질문 → 스타트업 vs 대기업 요구사항 비교 답변
    """

    SYSTEM_PROMPT = (
        "당신은 IT 채용 시장 데이터를 기반으로 개발자 학습 로드맵을 제시하는 전문 어시스턴트입니다. "
        "실제 채용공고 822개를 분석한 데이터를 바탕으로 구체적이고 실용적인 조언을 제공합니다."
    )

    # 로드맵 템플릿: {직무} → 단계별 기술
    ROADMAP_TEMPLATES: dict[str, list[dict]] = {
        "백엔드": [
            {"단계": 1, "주제": "프로그래밍 기초",      "기술": ["Python 또는 Java", "자료구조/알고리즘", "Git"]},
            {"단계": 2, "주제": "웹 프레임워크",         "기술": ["Spring Boot (Java)", "FastAPI/Django (Python)"]},
            {"단계": 3, "주제": "데이터베이스",           "기술": ["MySQL", "PostgreSQL", "Redis"]},
            {"단계": 4, "주제": "인프라/클라우드",        "기술": ["Docker", "AWS 기초", "CI/CD"]},
            {"단계": 5, "주제": "심화",                   "기술": ["Kubernetes", "마이크로서비스", "시스템 설계"]},
        ],
        "프론트엔드": [
            {"단계": 1, "주제": "웹 기초",               "기술": ["HTML/CSS", "JavaScript ES6+", "Git"]},
            {"단계": 2, "주제": "React 생태계",           "기술": ["React", "TypeScript", "상태관리(Zustand/Redux)"]},
            {"단계": 3, "주제": "빌드/툴링",              "기술": ["Webpack/Vite", "npm/yarn", "ESLint"]},
            {"단계": 4, "주제": "성능 최적화",            "기술": ["Next.js (SSR/SSG)", "Web Vitals", "번들 최적화"]},
            {"단계": 5, "주제": "심화",                   "기술": ["테스트(Jest/Cypress)", "디자인 시스템", "접근성"]},
        ],
        "AI_ML": [
            {"단계": 1, "주제": "Python/수학 기초",      "기술": ["Python", "NumPy/Pandas", "선형대수/통계"]},
            {"단계": 2, "주제": "ML 기초",               "기술": ["Scikit-learn", "데이터 전처리", "모델 평가"]},
            {"단계": 3, "주제": "딥러닝",                 "기술": ["PyTorch", "신경망 아키텍처", "GPU 활용"]},
            {"단계": 4, "주제": "LLM/생성형 AI",         "기술": ["Hugging Face", "Fine-tuning", "RAG", "LangChain"]},
            {"단계": 5, "주제": "MLOps",                  "기술": ["MLflow", "Docker", "Kubeflow", "AWS SageMaker"]},
        ],
        "DevOps": [
            {"단계": 1, "주제": "Linux/네트워크 기초",   "기술": ["Linux 명령어", "TCP/IP", "Shell Script"]},
            {"단계": 2, "주제": "컨테이너",               "기술": ["Docker", "Docker Compose"]},
            {"단계": 3, "주제": "CI/CD",                  "기술": ["GitHub Actions", "Jenkins", "ArgoCD"]},
            {"단계": 4, "주제": "클라우드",               "기술": ["AWS (EC2/EKS/RDS)", "Terraform"]},
            {"단계": 5, "주제": "오케스트레이션",         "기술": ["Kubernetes", "모니터링(Prometheus/Grafana)"]},
        ],
        "풀스택": [
            {"단계": 1, "주제": "기초",                   "기술": ["JavaScript/TypeScript", "HTML/CSS", "Git"]},
            {"단계": 2, "주제": "프론트엔드",             "기술": ["React", "Next.js", "상태관리"]},
            {"단계": 3, "주제": "백엔드",                 "기술": ["Node.js/Express 또는 FastAPI", "REST API", "인증"]},
            {"단계": 4, "주제": "데이터베이스/인프라",    "기술": ["PostgreSQL", "Redis", "Docker", "배포"]},
            {"단계": 5, "주제": "심화",                   "기술": ["시스템 설계", "성능 최적화", "테스트"]},
        ],
    }

    # 유형A 질문 템플릿 (직무/회사 슬롯)
    ROADMAP_Q_TEMPLATES = [
        "{job} 개발자로 {company}에 취업하려면 어떤 순서로 공부해야 하나요?",
        "{company} {job} 포지션 준비를 위한 단계별 학습 로드맵을 알려주세요.",
        "취업 준비생인데 {job} 분야 {company} 취업을 목표로 하고 있어요. 로드맵 추천해 주세요.",
        "{job} 개발자 {company} 취업 준비, 어디서부터 시작해야 하나요?",
        "비전공자인데 {job} 개발자로 {company}에 취업할 수 있을까요? 공부 순서 알려주세요.",
        "{company} {job} 신입 공채 준비 중인데 1년 안에 가능한 로드맵 있을까요?",
        "현재 {job} 공부 중인데 {company} 취업을 위해 핵심만 뽑으면 뭘 먼저 해야 하나요?",
        "{job} 분야로 이직을 고려 중이에요. {company} 기준으로 필수 기술이 뭔가요?",
        "{company}에서 {job} 개발자로 일하려면 어떤 기술 스택을 알아야 하나요?",
        "{job} 개발자 취업 준비를 6개월 안에 끝내야 한다면 {company} 기준으로 어떻게 할까요?",
    ]

    # 유형B 질문 템플릿 (기술 선택)
    TECH_CHOICE_Q_TEMPLATES = [
        "{size}에 취업하고 싶은데, 어떤 언어와 프레임워크를 먼저 배워야 할까요?",
        "{size} 개발자 채용 시장에서 가장 많이 요구하는 기술이 뭔가요?",
        "{size} 취업을 위해 기술 스택을 선택해야 하는데 뭘 골라야 할까요?",
        "요즘 {size} 채용공고에서 자주 보이는 기술 스택이 뭔지 알 수 있을까요?",
        "{size} 신입 개발자 공고에서 공통적으로 요구하는 기술은 무엇인가요?",
    ]

    # 유형C 비교 질문 템플릿
    COMPARISON_Q_TEMPLATES = [
        "스타트업과 대기업의 기술 요구사항 차이가 궁금해요.",
        "{job} 개발자인데 스타트업이랑 대기업 중 어디를 먼저 목표로 해야 할까요?",
        "스타트업 개발자와 대기업 개발자가 사용하는 기술 스택이 많이 다른가요?",
        "취업 목표를 스타트업으로 할지 대기업으로 할지 모르겠어요. 기술 준비 방향이 다른가요?",
        "대기업 공고랑 스타트업 공고를 비교해봤는데 요구사항이 많이 달라요. 어떻게 준비해야 하죠?",
        "스타트업 경력으로 대기업으로 이직할 때 추가로 준비해야 할 기술이 있나요?",
    ]

    def __init__(
        self,
        cleaned_csv:  str = "data/processed/cleaned_jobs.csv",
        jsonl_path:   str = "data/raw/wanted_nlp_data.jsonl",
        metrics_path: str = "outputs/metrics/company_comparison_metrics.json",
    ):
        self.cleaned_csv  = cleaned_csv
        self.jsonl_path   = jsonl_path
        self.metrics_path = metrics_path
        self.df:          Optional[pd.DataFrame] = None
        self.metrics:     dict = {}
        self.samples:     List[dict] = []
        self._jsonl_data: List[dict] = []

    # ── 로드 ──────────────────────────────────────────────────────────────

    def load(self) -> "RoadmapDatasetBuilder":
        print("\n[1] 데이터 로드...")

        if os.path.exists(self.cleaned_csv):
            self.df = pd.read_csv(self.cleaned_csv)
            print(f"  CSV: {len(self.df):,}개")
        else:
            print(f"  ⚠ CSV 없음: {self.cleaned_csv}")

        if os.path.exists(self.jsonl_path):
            with open(self.jsonl_path, encoding='utf-8') as f:
                self._jsonl_data = [json.loads(l) for l in f if l.strip()]
            print(f"  JSONL: {len(self._jsonl_data):,}개")

        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, encoding='utf-8') as f:
                self.metrics = json.load(f)
            print(f"  metrics 로드: {self.metrics_path}")
        else:
            print(f"  ⚠ metrics 없음 ({self.metrics_path}) — 기본값 사용")
        return self

    # ── 헬퍼 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _format_roadmap(job: str, size: str, roadmap: List[dict]) -> str:
        lines = [f"**{job} 개발자 학습 로드맵** ({size} 취업 목표)\n"]
        for step in roadmap:
            techs = ", ".join(step["기술"])
            lines.append(f"📌 {step['단계']}단계 — {step['주제']}")
            lines.append(f"   필수 기술: {techs}\n")
        lines.append(
            "💡 팁: 단계별로 프로젝트를 완성하며 GitHub에 포트폴리오를 쌓으세요. "
            "각 단계는 2~4주를 권장합니다."
        )
        return "\n".join(lines)

    def _top5_str(self, size: str) -> str:
        top = (self.metrics.get("top_techs_per_size") or {}).get(size, [])
        return ", ".join(f"{t['tech']}({t['ratio']}%)" for t in top[:5]) if top else "데이터 없음"

    def _top3_lang(self, size: str) -> List[tuple]:
        data = (self.metrics.get("tech_by_size") or {}).get(size, {}).get("언어", {})
        return sorted(data.items(), key=lambda x: x[1], reverse=True)[:3]

    def _top3_fw(self, size: str) -> List[tuple]:
        data = (self.metrics.get("tech_by_size") or {}).get(size, {}).get("프레임워크", {})
        return sorted(data.items(), key=lambda x: x[1], reverse=True)[:3]

    def _fullstack_pref(self, size: str) -> float:
        return (self.metrics.get("fullstack_preference") or {}).get(size, 0.0)

    def _make_sample(self, instruction: str, response: str) -> dict:
        return {
            "conversations": [
                {"role": "system",    "content": self.SYSTEM_PROMPT},
                {"role": "user",      "content": instruction},
                {"role": "assistant", "content": response},
            ]
        }

    # ════════════════════════════════════════════════════════════════════
    # 유형 A — 로드맵 질문 (직무 × 규모 × 질문 템플릿 전체 조합)
    # ════════════════════════════════════════════════════════════════════

    def _build_roadmap_samples(self) -> List[dict]:
        job_size_pairs = [
            ("백엔드",     "스타트업/중소", "스타트업"),
            ("백엔드",     "대기업",        "대기업"),
            ("프론트엔드", "스타트업/중소", "스타트업"),
            ("프론트엔드", "대기업",        "대기업"),
            ("AI_ML",      "스타트업/중소", "AI 스타트업"),
            ("AI_ML",      "대기업",        "대기업"),
            ("DevOps",     "스타트업/중소", "스타트업"),
            ("DevOps",     "대기업",        "대기업"),
            ("풀스택",     "스타트업/중소", "스타트업"),
            ("풀스택",     "대기업",        "대기업"),
        ]
        samples = []
        for job, size_key, size_display in job_size_pairs:
            if job not in self.ROADMAP_TEMPLATES:
                continue
            roadmap = self.ROADMAP_TEMPLATES[job]
            top5    = self._top5_str(size_key)
            for q_tmpl in self.ROADMAP_Q_TEMPLATES:
                q = q_tmpl.format(job=job, company=size_display)
                a = (
                    self._format_roadmap(job, size_display, roadmap)
                    + f"\n\n📊 채용 시장 데이터 (원티드 822개 공고 분석):\n"
                    + f"   {size_key} TOP5 기술: {top5}"
                )
                samples.append(self._make_sample(q, a))
        return samples   # 10쌍 × 10템플릿 = 100개

    # ════════════════════════════════════════════════════════════════════
    # 유형 B — 기술 선택 질문 (규모 × 템플릿)
    # ════════════════════════════════════════════════════════════════════

    def _build_tech_choice_samples(self) -> List[dict]:
        samples = []
        for size_key in ["스타트업/중소", "대기업"]:
            top_lang = self._top3_lang(size_key)
            top_fw   = self._top3_fw(size_key)
            if not top_lang:
                continue
            lang_str = ", ".join(f"{t}({r}%)" for t, r in top_lang)
            fw_str   = ", ".join(f"{t}({r}%)" for t, r in top_fw)
            size_disp = "스타트업" if "스타트업" in size_key else "대기업"

            for q_tmpl in self.TECH_CHOICE_Q_TEMPLATES:
                q = q_tmpl.format(size=size_disp)
                a = (
                    f"채용공고 분석 결과, **{size_key}**에서 가장 많이 요구되는 기술입니다.\n\n"
                    f"📌 **프로그래밍 언어** (공고 내 등장 비율)\n   {lang_str}\n\n"
                    f"📌 **프레임워크/라이브러리** (등장 비율)\n   {fw_str}\n\n"
                    f"✅ 추천 학습 순서:\n"
                    f"   1. {top_lang[0][0]} 언어 기초 숙달 (문법 + 자료구조)\n"
                    f"   2. {top_fw[0][0]} 프레임워크로 실습 프로젝트 1개 완성\n"
                    f"   3. Docker + CI/CD 기본 배포 경험\n"
                    f"   4. AWS 기초 (EC2, S3, RDS) 실습\n\n"
                    f"💡 {size_disp}은 {'즉시 투입 가능한 실무 능력' if '스타트업' in size_key else 'CS 기초 + 코딩테스트 역량'}을 중시합니다."
                )
                samples.append(self._make_sample(q, a))
        return samples   # 2규모 × 5템플릿 = 10개

    # ════════════════════════════════════════════════════════════════════
    # 유형 C — 스타트업 vs 대기업 비교 질문
    # ════════════════════════════════════════════════════════════════════

    def _build_comparison_samples(self) -> List[dict]:
        distinctive  = self.metrics.get("distinctive_techs") or {}
        startup_list = distinctive.get("스타트업/중소_우세", [])[:5]
        corp_list    = distinctive.get("대기업_우세", [])[:5]
        fs_startup   = self._fullstack_pref("스타트업/중소")
        fs_corp      = self._fullstack_pref("대기업")

        startup_str = ", ".join(
            f"{d['tech']}(+{d['diff']}%p)" for d in startup_list
        ) or "데이터 수집 중"
        corp_str = ", ".join(
            f"{d['tech']}(+{d['diff']}%p)" for d in corp_list
        ) or "데이터 수집 중"

        base_answer = (
            "채용공고 데이터 분석 결과입니다.\n\n"
            f"🚀 **스타트업/중소기업** 특화 기술 (대기업 대비)\n"
            f"   {startup_str}\n"
            f"   풀스택 비율: {fs_startup}% — 한 사람이 넓은 범위 담당\n\n"
            f"🏢 **대기업** 특화 기술 (스타트업 대비)\n"
            f"   {corp_str}\n"
            f"   풀스택 비율: {fs_corp}% — 역할 분업 명확, 전문성 심화 요구\n\n"
            "✅ **준비 전략 차이**:\n"
            "   스타트업 → 다양한 스택 경험 + 빠른 프로토타이핑 능력\n"
            "   대기업   → 특정 스택 깊이 + CS 기초 + 코딩테스트 필수"
        )

        samples = []
        for i, q_tmpl in enumerate(self.COMPARISON_Q_TEMPLATES):
            # 직무 슬롯이 있는 템플릿에는 직무 채워넣기
            jobs = ["백엔드", "프론트엔드", "AI_ML", "DevOps", "풀스택", "백엔드"]
            q = q_tmpl.format(job=jobs[i % len(jobs)]) if "{job}" in q_tmpl else q_tmpl
            samples.append(self._make_sample(q, base_answer))
        return samples   # 6개

    # ════════════════════════════════════════════════════════════════════
    # 유형 D — 실제 채용공고(CSV) 전체 활용 → 포지션 분석 + 준비 전략
    # ════════════════════════════════════════════════════════════════════

    def _build_jobposting_samples(self) -> List[dict]:
        """cleaned_jobs.csv 전체에서 유효 행을 모두 샘플로 변환."""
        if self.df is None:
            return []

        q_templates = [
            "다음은 {size} {job} 채용공고 자격요건입니다.\n\n[자격요건]\n{req}\n\n이 포지션 합격을 위한 준비 전략을 알려주세요.",
            "아래 {size} {job} 공고를 보고 있는데, 어떤 기술부터 준비해야 할지 모르겠어요.\n\n[자격요건]\n{req}",
            "{size} {job} 포지션에 지원하려고 합니다. 자격요건을 분석해서 학습 우선순위를 알려주세요.\n\n[자격요건]\n{req}",
            "이 {size} {job} 공고에 6개월 안에 합격하려면 뭘 먼저 해야 할까요?\n\n[자격요건]\n{req}",
        ]

        df_valid = self.df[
            self.df['자격요건_clean'].notna() &
            self.df['회사규모'].isin(["스타트업/중소", "대기업"]) &
            self.df['직무카테고리'].notna() &
            self.df['tech_all'].notna()
        ].copy()

        samples = []
        for idx, (_, row) in enumerate(df_valid.iterrows()):
            job  = str(row.get('직무카테고리', '개발'))
            size = str(row.get('회사규모', '회사'))
            req  = str(row.get('자격요건_clean', ''))[:400].strip()
            techs_raw = str(row.get('tech_all', ''))
            techs = [t.strip() for t in techs_raw.split(',')
                     if t.strip() and t.strip().lower() != 'nan'][:6]

            if not req or len(req) < 20 or not techs:
                continue

            tech_str  = ", ".join(techs)
            size_tip  = (
                "즉시 투입 가능한 실무 경험과 포트폴리오를 강조하세요."
                if "스타트업" in size else
                "코딩테스트 대비, CS 기초(OS·네트워크·DB), 팀 협업 경험이 중요합니다."
            )

            a_lines = [
                f"이 **{size} {job}** 포지션 분석 결과입니다.\n",
                f"📌 핵심 요구 기술: {tech_str}\n",
                "✅ 우선순위별 학습 전략:",
            ]
            for i, tech in enumerate(techs[:4], 1):
                a_lines.append(f"   {i}. **{tech}** — 공식 문서 + 실습 프로젝트 1개 완성")
            a_lines += [
                "",
                "📂 포트폴리오 구성 팁:",
                f"   - 위 기술 스택을 활용한 미니 프로젝트 GitHub 업로드",
                f"   - README에 기술 선택 이유와 트러블슈팅 경험 명시",
                "",
                f"💡 {size} 지원 포인트: {size_tip}",
            ]

            # 질문 템플릿 순환 (다양성 확보)
            q_tmpl = q_templates[idx % len(q_templates)]
            q = q_tmpl.format(size=size, job=job, req=req)
            samples.append(self._make_sample(q, "\n".join(a_lines)))

        return samples

    # ════════════════════════════════════════════════════════════════════
    # 유형 E — JSONL(원티드 822개) 전체 활용 → 주요업무 기반 QA
    # ════════════════════════════════════════════════════════════════════

    def _build_jsonl_samples(self) -> List[dict]:
        """wanted_nlp_data.jsonl 전체를 '이 회사에서 하는 일 / 필요한 기술' QA로 변환."""
        if not self._jsonl_data:
            return []

        q_templates = [
            "원티드에 올라온 '{title}' 포지션에 지원하려면 어떤 준비를 해야 할까요?",
            "'{title}' 직무의 주요 업무를 보고 싶어요. 이 포지션에 필요한 핵심 역량이 뭔가요?",
            "다음 채용공고 주요업무를 분석해서 취업 준비 방향을 알려주세요.\n\n[주요업무]\n{duties}",
            "'{title}' 공고를 보고 있는데, 자격요건에 나온 기술들을 어떻게 준비해야 하나요?\n\n[자격요건]\n{req}",
            "이런 업무를 하는 회사에 취업하고 싶어요. 필요한 기술 스택이 뭔가요?\n\n[주요업무]\n{duties}",
        ]

        samples = []
        for idx, row in enumerate(self._jsonl_data):
            title  = str(row.get('공고제목', '개발자'))[:60]
            duties = str(row.get('주요업무_clean') or row.get('주요업무', ''))[:300].strip()
            req    = str(row.get('자격요건_clean') or row.get('자격요건', ''))[:300].strip()

            if not duties and not req:
                continue

            # 우대사항도 활용
            pref   = str(row.get('우대사항_clean') or row.get('우대사항', ''))[:200].strip()
            region = str(row.get('지역', '서울'))
            search = str(row.get('검색어', '개발'))   # 프론트엔드/백엔드/etc

            a = (
                f"**{title}** 포지션 분석입니다.\n\n"
                f"📋 **주요 업무 요약**\n{duties[:200] if duties else '정보 없음'}\n\n"
                f"📌 **필수 자격요건**\n{req[:200] if req else '정보 없음'}\n\n"
            )
            if pref:
                a += f"⭐ **우대사항** (있으면 플러스)\n{pref[:150]}\n\n"
            a += (
                f"✅ **준비 가이드**\n"
                f"   1. 자격요건에 나온 기술 스택 우선 학습\n"
                f"   2. 주요업무와 유사한 미니 프로젝트 1~2개 제작\n"
                f"   3. 우대사항 기술은 여유 있을 때 추가 학습\n"
                f"   4. {region} 기반 회사이므로 현지 취업 네트워크도 활용하세요.\n\n"
                f"💡 **{search}** 분야 공고 기준으로, 해당 도메인의 "
                f"오픈소스 기여나 개인 프로젝트가 있으면 강점이 됩니다."
            )

            q_tmpl = q_templates[idx % len(q_templates)]
            q = q_tmpl.format(title=title, duties=duties[:200], req=req[:200])
            samples.append(self._make_sample(q, a))

        return samples   # 최대 822개

    # ── 공통 포맷터 ───────────────────────────────────────────────────

    def _make_sample(self, instruction: str, response: str) -> dict:
        return {
            "conversations": [
                {"role": "system",    "content": self.SYSTEM_PROMPT},
                {"role": "user",      "content": instruction},
                {"role": "assistant", "content": response},
            ]
        }

    # ── 전체 빌드 ─────────────────────────────────────────────────────

    def build(self) -> List[dict]:
        print("\n[2] 파인튜닝 데이터셋 생성...")
        all_samples: List[dict] = []

        a = self._build_roadmap_samples()
        print(f"  유형A (로드맵):      {len(a):4d}개  (직무×규모×질문템플릿 전체 조합)")
        all_samples.extend(a)

        b = self._build_tech_choice_samples()
        print(f"  유형B (기술선택):    {len(b):4d}개  (규모×질문템플릿)")
        all_samples.extend(b)

        c = self._build_comparison_samples()
        print(f"  유형C (비교):        {len(c):4d}개  (스타트업 vs 대기업)")
        all_samples.extend(c)

        d = self._build_jobposting_samples()
        print(f"  유형D (공고기반CSV): {len(d):4d}개  (cleaned_jobs 전체 활용)")
        all_samples.extend(d)

        e = self._build_jsonl_samples()
        print(f"  유형E (JSONL전체):   {len(e):4d}개  (wanted_nlp_data 822개)")
        all_samples.extend(e)

        random.shuffle(all_samples)
        self.samples = all_samples
        print(f"\n  ✅ 총 {len(all_samples):,}개 샘플 생성 완료")
        return all_samples

    def save(self) -> str:
        """JSONL 형식으로 저장. 경로 반환."""
        os.makedirs(OUT_DATA, exist_ok=True)
        path = os.path.join(OUT_DATA, "roadmap_train.jsonl")
        with open(path, 'w', encoding='utf-8') as f:
            for s in self.samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  ✅ 데이터셋 저장: {path} ({len(self.samples)}개)")
        return path


# ════════════════════════════════════════════════════════════════════════════
# QLoRA 파인튜너
# ════════════════════════════════════════════════════════════════════════════

class QLoRAFineTuner:
    """
    Llama-3.2-Korean-GGACHI-1B-Instruct-v1 QLoRA 파인튜너.

    사용법
    ------
    >>> ft = QLoRAFineTuner("data/finetune/roadmap_train.jsonl")
    >>> ft.setup()
    >>> ft.train()
    >>> ft.save()
    >>> result = ft.infer("백엔드 개발자 스타트업 취업 로드맵 알려줘")
    """

    def __init__(self, dataset_path: str = "data/finetune/roadmap_train.jsonl"):
        self.dataset_path = dataset_path
        self.model        = None
        self.tokenizer    = None
        self.trainer      = None

    # ── 의존성 검사 ──────────────────────────────────────────────────────

    @staticmethod
    def check_dependencies() -> bool:
        missing = []
        for pkg in ["transformers", "datasets", "peft", "trl", "bitsandbytes", "accelerate"]:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        if missing:
            print(f"\n⚠ 미설치 패키지: {', '.join(missing)}")
            print("  설치 명령: pip install " + " ".join(missing))
            return False
        return True

    # ── 모델 & 토크나이저 로드 ───────────────────────────────────────────

    def setup(self) -> "QLoRAFineTuner":
        if not self.check_dependencies():
            raise EnvironmentError("필수 패키지를 먼저 설치하세요.")

        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                   BitsAndBytesConfig)
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        print(f"\n[3] 모델 로드: {MODEL_ID}")
        print(f"  CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # 4-bit 양자화 설정 (RTX 4060 8GB 기준)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        base_model = prepare_model_for_kbit_training(base_model)

        lora_config = LoraConfig(**LORA_CFG)
        self.model  = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()
        print("  모델 로드 완료 ✅")
        return self

    # ── 데이터셋 포맷 ────────────────────────────────────────────────────

    def _format_conversation(self, sample: dict) -> str:
        """
        Llama-3 Instruct 형식:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>...<|eot_id|>
        <|start_header_id|>user<|end_header_id|>...<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>...
        """
        convs = sample.get("conversations", [])
        text  = "<|begin_of_text|>"
        for turn in convs:
            role    = turn["role"]
            content = turn["content"]
            text += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>\n"
        text += "<|end_of_text|>"
        return text

    def _load_dataset(self):
        from datasets import Dataset

        raw = []
        with open(self.dataset_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    raw.append(json.loads(line))

        texts = [self._format_conversation(s) for s in raw]

        def tokenize(examples):
            tokens = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=TRAIN_CFG["max_seq_length"],
                padding="max_length",
            )
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens

        dataset = Dataset.from_dict({"text": texts})
        dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
        split   = dataset.train_test_split(test_size=0.1, seed=42)
        print(f"  train: {len(split['train'])}개 / eval: {len(split['test'])}개")
        return split

    # ── 학습 ─────────────────────────────────────────────────────────────

    def train(self) -> "QLoRAFineTuner":
        from transformers import TrainingArguments, DataCollatorForLanguageModeling
        from trl import SFTTrainer

        print(f"\n[4] 파인튜닝 시작...")
        dataset = self._load_dataset()

        training_args = TrainingArguments(
            output_dir=OUT_MODEL,
            **{k: v for k, v in TRAIN_CFG.items()
               if k not in ("max_seq_length",)},
            report_to="none",
            remove_unused_columns=False,
        )

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            args=training_args,
            max_seq_length=TRAIN_CFG["max_seq_length"],
            dataset_text_field="",           # 이미 토크나이즈됨
            packing=False,
        )

        print("  학습 시작 (RTX 4060 기준 약 20~40분 예상)...")
        self.trainer.train()
        print("  ✅ 학습 완료")
        return self

    # ── 저장 ─────────────────────────────────────────────────────────────

    def save(self) -> str:
        """LoRA adapter만 저장 (경량, ~50MB)."""
        os.makedirs(ADAPTER_SAVE, exist_ok=True)
        self.model.save_pretrained(ADAPTER_SAVE)
        self.tokenizer.save_pretrained(ADAPTER_SAVE)

        # 3번 담당자용 메타데이터
        meta = {
            "base_model":   MODEL_ID,
            "adapter_path": ADAPTER_SAVE,
            "max_seq_length": TRAIN_CFG["max_seq_length"],
            "usage": (
                "from peft import PeftModel; from transformers import AutoModelForCausalLM, AutoTokenizer; "
                f"tok = AutoTokenizer.from_pretrained('{ADAPTER_SAVE}'); "
                f"model = AutoModelForCausalLM.from_pretrained('{MODEL_ID}'); "
                f"model = PeftModel.from_pretrained(model, '{ADAPTER_SAVE}')"
            )
        }
        meta_path = os.path.join(OUT_METRICS, "finetuning_meta.json")
        os.makedirs(OUT_METRICS, exist_ok=True)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"  ✅ Adapter 저장: {ADAPTER_SAVE}")
        print(f"  ✅ 메타데이터:   {meta_path}")
        return ADAPTER_SAVE

    # ── 개인 프로필 → 구조화 프롬프트 빌더 ─────────────────────────────

    @staticmethod
    def build_profile_prompt(profile: dict) -> str:
        """
        UI 드롭박스에서 넘어온 개인정보 딕셔너리 → 모델 입력 프롬프트 생성.

        profile 예시
        ------------
        {
            "job":        "백엔드",          # 목표 직무
            "company":    "스타트업",         # 목표 회사 규모
            "experience": "비전공자",         # 경력/배경
            "current_stack": ["Python"],     # 현재 알고 있는 기술
            "goal_period": "6개월",          # 목표 기간
        }
        """
        job        = profile.get("job", "백엔드")
        company    = profile.get("company", "스타트업")
        experience = profile.get("experience", "비전공자")
        stacks     = profile.get("current_stack", [])
        period     = profile.get("goal_period", "6개월")
        stack_str  = ", ".join(stacks) if stacks else "없음"

        return (
            f"저는 {experience} 배경을 가진 취업 준비생입니다.\n"
            f"목표: {company} {job} 개발자 취업 ({period} 내)\n"
            f"현재 알고 있는 기술: {stack_str}\n\n"
            f"위 정보를 바탕으로 저에게 맞는 단계별 학습 로드맵을 JSON 형식으로 출력해주세요.\n\n"
            "반드시 아래 JSON 형식만 출력하세요 (다른 텍스트 없이):\n"
            "{\n"
            '  "title": "로드맵 제목",\n'
            '  "summary": "한 줄 요약",\n'
            '  "total_weeks": 숫자,\n'
            '  "steps": [\n'
            "    {\n"
            '      "step": 1,\n'
            '      "title": "단계 제목",\n'
            '      "duration": "2주",\n'
            '      "techs": ["기술1", "기술2"],\n'
            '      "description": "이 단계에서 할 일 설명",\n'
            '      "resources": ["추천 학습 자료1", "자료2"],\n'
            '      "milestone": "완료 기준"\n'
            "    }\n"
            "  ],\n"
            '  "portfolio_tip": "포트폴리오 조언",\n'
            '  "market_insight": "시장 데이터 기반 인사이트"\n'
            "}"
        )

    # ── 추론: 개인 프로필 → 구조화 로드맵 JSON ──────────────────────────

    def infer(self, question: str, max_new_tokens: int = 600) -> str:
        """단순 텍스트 질문 → 텍스트 응답 (하위 호환용)."""
        import torch
        prompt = (
            "<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"{RoadmapDatasetBuilder.SYSTEM_PROMPT}<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return generated.strip()

    def infer_roadmap(self, profile: dict, max_new_tokens: int = 900) -> dict:
        """
        개인 프로필 딕셔너리 → 구조화된 로드맵 딕셔너리 반환.

        Parameters
        ----------
        profile : dict
            UI 드롭박스에서 받은 개인정보
            {
                "job":           "백엔드" | "프론트엔드" | "AI_ML" | "DevOps" | "풀스택",
                "company":       "스타트업" | "대기업",
                "experience":    "비전공자" | "전공자" | "1~2년차" | "3년차 이상",
                "current_stack": ["Python", "Django"],   # 리스트
                "goal_period":   "3개월" | "6개월" | "1년",
            }

        Returns
        -------
        dict  (파싱 실패 시 fallback dict 반환, 예외 없음)
        """
        import torch
        import re

        user_msg = self.build_profile_prompt(profile)
        prompt   = (
            "<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"{RoadmapDatasetBuilder.SYSTEM_PROMPT}<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,     # JSON이 깨지지 않게 낮게
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        raw = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        # ── JSON 파싱 시도 ───────────────────────────────────────────
        # 모델이 마크다운 코드블록 안에 넣는 경우 처리
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # ── 파싱 실패 → 규칙 기반 fallback ──────────────────────────
        return self._fallback_roadmap(profile)

    def _fallback_roadmap(self, profile: dict) -> dict:
        """
        모델 JSON 파싱 실패 시 규칙 기반으로 로드맵 딕셔너리 생성.
        완전히 동일한 스키마를 보장 → 시각화 코드가 항상 동작.
        """
        job     = profile.get("job", "백엔드")
        company = profile.get("company", "스타트업")
        period  = profile.get("goal_period", "6개월")

        # ROADMAP_TEMPLATES에서 해당 직무 데이터 가져오기
        templates = RoadmapDatasetBuilder.ROADMAP_TEMPLATES
        raw_steps = templates.get(job, templates["백엔드"])

        period_map = {"3개월": 12, "6개월": 24, "1년": 48}
        total_weeks = period_map.get(period, 24)
        weeks_per_step = max(1, total_weeks // len(raw_steps))

        steps = []
        resources_db = {
            "Python":       ["점프 투 파이썬 (무료 온라인)", "Python 공식 문서"],
            "Java":         ["점프 투 자바", "백기선 Java 강의"],
            "Spring Boot":  ["스프링 공식 가이드", "인프런 스프링부트 강의"],
            "React":        ["React 공식 문서 (react.dev)", "모던 리액트 딥다이브"],
            "Docker":       ["Docker 공식 Docs", "subicura 블로그 Docker 입문"],
            "AWS":          ["AWS 무료 티어 실습", "AWS 공식 Getting Started"],
            "MySQL":        ["MySQL 공식 문서", "생활코딩 MySQL"],
            "TypeScript":   ["TypeScript 공식 핸드북", "타입스크립트 교과서"],
            "PyTorch":      ["PyTorch 공식 튜토리얼", "딥러닝 with PyTorch (한빛)"],
            "Kubernetes":   ["Kubernetes 공식 Docs", "CNCF 한국어 자료"],
        }
        default_res = ["공식 문서 정독", "YouTube 입문 강의", "개인 실습 프로젝트"]

        for i, raw in enumerate(raw_steps, 1):
            techs = [t.split("(")[0].strip() for t in raw["기술"]]
            res   = []
            for t in techs[:2]:
                res.extend(resources_db.get(t, default_res[:1]))
            if not res:
                res = default_res

            steps.append({
                "step":        i,
                "title":       raw["주제"],
                "duration":    f"{weeks_per_step}주",
                "techs":       techs,
                "description": f"{raw['주제']} 단계에서는 {', '.join(techs[:3])}을 집중 학습합니다.",
                "resources":   list(dict.fromkeys(res))[:3],
                "milestone":   f"{techs[0]} 기반 미니 프로젝트 1개 완성",
            })

        size_insight = (
            "스타트업은 즉시 투입 가능한 실무 경험과 빠른 배포 능력을 중시합니다."
            if "스타트업" in company else
            "대기업은 CS 기초, 코딩테스트, 체계적 아키텍처 설계 능력을 중시합니다."
        )

        return {
            "title":          f"{job} 개발자 {period} 로드맵 ({company} 목표)",
            "summary":        f"{company} {job} 개발자 취업을 위한 {period} 학습 커리큘럼",
            "total_weeks":    total_weeks,
            "steps":          steps,
            "portfolio_tip":  (
                "각 단계 완료 시 GitHub에 커밋 기록을 남기고, "
                "README에 기술 선택 이유와 트러블슈팅 경험을 상세히 작성하세요."
            ),
            "market_insight": size_insight,
        }

    # ── 저장된 adapter 로드 ──────────────────────────────────────────────

    @classmethod
    def load_for_inference(cls, adapter_path: str = ADAPTER_SAVE) -> "QLoRAFineTuner":
        """
        학습 없이 추론만 할 때 사용 (3번 담당자 호출용).
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        print(f"[추론 모드] adapter 로드: {adapter_path}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        base      = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=bnb_config, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, adapter_path)

        ft = cls()
        ft.tokenizer = tokenizer
        ft.model     = model
        return ft


# ════════════════════════════════════════════════════════════════════════════
# 전체 파이프라인
# ════════════════════════════════════════════════════════════════════════════

class FineTuningPipeline:
    """
    데이터 생성 + 파인튜닝을 한 번에 실행하는 파사드 클래스.

    사용법
    ------
    >>> pipeline = FineTuningPipeline()
    >>> pipeline.run()
    """

    def __init__(
        self,
        cleaned_csv:  str = "data/processed/cleaned_jobs.csv",
        jsonl_path:   str = "data/raw/wanted_nlp_data.jsonl",
        metrics_path: str = "outputs/metrics/company_comparison_metrics.json",
        skip_train:   bool = False,
    ):
        self.cleaned_csv  = cleaned_csv
        self.jsonl_path   = jsonl_path
        self.metrics_path = metrics_path
        self.skip_train   = skip_train

    def run(self) -> dict:
        print("=" * 70)
        print("🤖 파인튜닝 파이프라인")
        print("=" * 70)

        # ── 1. 데이터셋 생성 ──────────────────────────────────────
        builder = RoadmapDatasetBuilder(
            cleaned_csv  = self.cleaned_csv,
            jsonl_path   = self.jsonl_path,
            metrics_path = self.metrics_path,
        )
        builder.load().build()
        dataset_path = builder.save()

        if self.skip_train:
            print("\n  [skip_train=True] 학습 건너뜀")
            return {"dataset_path": dataset_path, "adapter_path": None}

        # ── 2. QLoRA 파인튜닝 ─────────────────────────────────────
        ft = QLoRAFineTuner(dataset_path)
        ft.setup()
        ft.train()
        adapter_path = ft.save()

        # ── 3. 샘플 추론 ──────────────────────────────────────────
        print("\n[5] 추론 테스트...")
        test_q = "백엔드 개발자로 스타트업에 취업하고 싶어요. 어떻게 공부해야 하나요?"
        answer = ft.infer(test_q)
        print(f"\nQ: {test_q}")
        print(f"A: {answer[:300]}...")

        return {
            "dataset_path": dataset_path,
            "adapter_path": adapter_path,
        }


# ════════════════════════════════════════════════════════════════════════════
# 직접 실행
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="로드맵 LLM 파인튜닝")
    parser.add_argument("--csv",          default="./cleaned_jobs.csv")
    parser.add_argument("--jsonl",        default="./wanted_nlp_data.jsonl")
    parser.add_argument("--metrics",      default="outputs/metrics/company_comparison_metrics.json")
    parser.add_argument("--skip-train",   action="store_true", help="데이터 생성만, 학습 건너뜀")
    parser.add_argument("--data-only",    action="store_true", help="--skip-train 의 별칭")
    args = parser.parse_args()

    pipeline = FineTuningPipeline(
        cleaned_csv  = args.csv,
        jsonl_path   = args.jsonl,
        metrics_path = args.metrics,
        skip_train   = args.skip_train or args.data_only,
    )
    pipeline.run()