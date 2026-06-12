#!/usr/bin/env python3
"""
[2번 담당] roadmap_visualizer.py
개인 프로필(드롭박스 input) → 모델 추론 → 타임라인형 로드맵 시각화

사용법
------
# 직접 실행 (모델 없이 fallback으로 테스트)
python roadmap_visualizer.py

# 코드에서 호출
from roadmap_visualizer import RoadmapVisualizer
from fine_tuning import QLoRAFineTuner

ft = QLoRAFineTuner.load_for_inference("models/roadmap_llm/adapter")
profile = {
    "job":           "백엔드",
    "company":       "스타트업",
    "experience":    "비전공자",
    "current_stack": ["Python"],
    "goal_period":   "6개월",
}
roadmap_data = ft.infer_roadmap(profile)
viz = RoadmapVisualizer()
path = viz.render(roadmap_data, profile, save_path="outputs/my_roadmap.png")
"""

from __future__ import annotations

import os
import platform
import textwrap
from typing import Optional, List

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm


# ── 한글 폰트 ─────────────────────────────────────────────────────────────
def _setup_font() -> None:
    if platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif platform.system() == 'Darwin':
        plt.rcParams['font.family'] = 'AppleGothic'
    else:
        candidates = ['NanumGothic', 'NanumBarunGothic', 'UnDotum', 'DejaVu Sans']
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = next((c for c in candidates if c in available), None)
        if chosen:
            plt.rcParams['font.family'] = chosen
    plt.rcParams['axes.unicode_minus'] = False


_setup_font()

# ── 색상 팔레트 ───────────────────────────────────────────────────────────
PALETTES = {
    "스타트업": [
        "#FF6B6B", "#FF9F43", "#FECA57", "#48DBFB",
        "#FF9FF3", "#54A0FF", "#5F27CD", "#00D2D3",
    ],
    "대기업": [
        "#2C3E7A", "#3A5BA0", "#4A7DC4", "#5A9FE8",
        "#2D9CDB", "#27AE60", "#1ABC9C", "#8E44AD",
    ],
}


# ════════════════════════════════════════════════════════════════════════════
# 메인 시각화 클래스 (중앙 타임라인 레이아웃)
# ════════════════════════════════════════════════════════════════════════════

class RoadmapVisualizer:
    """
    단정하고 모던한 수직 중앙 타임라인 렌더링.
    """

    def __init__(self, output_dir: str = "outputs/roadmaps"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ── 공개 API ─────────────────────────────────────────────────────────

    def render(
            self,
            roadmap: dict,
            profile: Optional[dict] = None,
            save_path: Optional[str] = None,
            show: bool = False,
    ) -> str:
        steps = roadmap.get("steps", [])
        n = len(steps)
        company = (profile or {}).get("company", "스타트업")
        palette = PALETTES.get(company, PALETTES["스타트업"])

        # 레이아웃 수치 계산
        fig_width = 14
        y_start = 3.5
        y_step = 3.2
        fig_height = max(12, y_start + n * y_step + 4.5)

        fig = plt.figure(figsize=(fig_width, fig_height), facecolor='#F4F5F7')
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 12)
        ax.set_ylim(0, fig_height)
        ax.axis('off')
        ax.invert_yaxis()

        # ── 상단 헤더 ───────────────────────────────────────────────
        self._draw_header(ax, roadmap, palette[0])

        # ── 중앙 타임라인 선 (스파인) ───────────────────────────────
        end_y = y_start + (n - 1) * y_step
        self._draw_spine(ax, y_start, end_y)

        # ── 노드 및 정보 카드 생성 ──────────────────────────────────
        self._draw_nodes(ax, steps, y_start, y_step, palette)

        # ── 하단 인사이트 ────────────────────────────────────────────
        footer_y = end_y + 2.5
        self._draw_footer(ax, roadmap, footer_y)

        # ── 저장 ─────────────────────────────────────────────────────
        if save_path is None:
            job = (profile or {}).get("job", "roadmap")
            save_path = os.path.join(self.output_dir, f"roadmap_{job}.png")

        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#F4F5F7')
        if show:
            plt.show()
        plt.close(fig)
        print(f"  ✅ 로드맵 저장: {save_path}")
        return save_path

    # ── 내부 드로잉 함수들 ───────────────────────────────────────────────

    def _draw_header(self, ax: plt.Axes, roadmap: dict, color: str) -> None:
        title = roadmap.get("title", "개발자 학습 로드맵")
        summary = roadmap.get("summary", "")
        weeks = roadmap.get("total_weeks", 0)

        ax.text(6, 0.8, "ROAD MAP", ha='center', va='center',
                fontsize=28, fontweight='900', color=color, alpha=0.9)
        ax.text(6, 1.5, title, ha='center', va='center',
                fontsize=16, fontweight='bold', color='#333333')
        ax.text(6, 2.0, summary, ha='center', va='center',
                fontsize=11, color='#666666')

        if weeks:
            ax.add_patch(mpatches.FancyBboxPatch(
                (10.0, 0.5), 1.5, 0.6,
                boxstyle="round,pad=0.1",
                facecolor=color, edgecolor='none', alpha=0.15
            ))
            ax.text(10.75, 0.8, f"총 {weeks}주", ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')

    def _draw_spine(self, ax: plt.Axes, start_y: float, end_y: float) -> None:
        # 굵고 연한 배경 선 + 얇고 진한 중앙 점선
        ax.plot([6, 6], [start_y, end_y], color='#EAEAEA', linewidth=10, zorder=1)
        ax.plot([6, 6], [start_y, end_y], color='#BDBDBD', linewidth=2, linestyle='--', zorder=2)

    def _draw_nodes(
            self, ax: plt.Axes, steps: list[dict], y_start: float, y_step: float, palette: list[str]
    ) -> None:

        card_w = 4.8
        card_h = 2.6

        for i, step in enumerate(steps):
            y = y_start + i * y_step
            color = palette[i % len(palette)]

            # 좌/우 교차 배치
            if i % 2 == 0:
                cx = 2.9
            else:
                cx = 9.1

            # 연결선
            ax.plot([6, cx], [y, y], color=color, linewidth=2, linestyle='--', alpha=0.6, zorder=3)

            # 원형 노드 (스텝 번호)
            ax.add_patch(plt.Circle((6, y), 0.45, color=color, zorder=5))
            ax.add_patch(plt.Circle((6, y), 0.45, fill=False, edgecolor='white', linewidth=3, zorder=6))
            ax.text(6, y, str(step.get("step", i + 1)), ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white', zorder=7)

            # 정보 카드 (배경 화이트 박스)
            box_x = cx - (card_w / 2)
            box_y = y - (card_h / 2)
            ax.add_patch(mpatches.FancyBboxPatch(
                (box_x, box_y), card_w, card_h,
                boxstyle="round,pad=0.1",
                facecolor='white', edgecolor=color, linewidth=1.5, alpha=0.95, zorder=4
            ))

            # 단계 제목
            ax.text(cx, y - 0.75, step.get('title', ''),
                    ha='center', va='center', fontsize=12, fontweight='bold', color=color, zorder=7)

            # 기간
            duration = step.get('duration', '')
            if duration:
                ax.text(cx, y - 0.45, duration,
                        ha='center', va='center', fontsize=9, color='#7F8C8D', zorder=7)

            # 기술 뱃지
            techs = step.get("techs", [])
            badge_w = 1.1
            badge_h = 0.35
            badge_gap = 0.1
            num_techs = min(len(techs), 3)

            if num_techs > 0:
                total_w = num_techs * badge_w + (num_techs - 1) * badge_gap
                start_x = cx - (total_w / 2)

                for j, tech in enumerate(techs[:3]):
                    bx = start_x + j * (badge_w + badge_gap)
                    ax.add_patch(mpatches.FancyBboxPatch(
                        (bx, y - 0.15), badge_w, badge_h,
                        boxstyle="round,pad=0.05",
                        facecolor=color, edgecolor='none', alpha=0.15, zorder=5
                    ))
                    ax.text(bx + badge_w / 2, y - 0.15 + badge_h / 2, tech,
                            ha='center', va='center', fontsize=8, color=color, fontweight='bold', zorder=7)

            # 설명 (줄바꿈 처리)
            desc = step.get("description", "")
            wrapped = textwrap.fill(desc, width=28)
            ax.text(cx, y + 0.45, wrapped,
                    ha='center', va='top', fontsize=9.5, color='#34495E', linespacing=1.5, zorder=7)

            # 마일스톤
            milestone = step.get("milestone", "")
            if milestone:
                ax.text(cx, y + 1.05, f"목표: {milestone}",
                        ha='center', va='top', fontsize=8.5, color='#27AE60', fontweight='bold', zorder=7)

    def _draw_footer(self, ax: plt.Axes, roadmap: dict, y_pos: float) -> None:
        tip = roadmap.get("portfolio_tip", "")
        insight = roadmap.get("market_insight", "")

        box_w = 5.2
        box_h = 1.6

        if tip:
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.5, y_pos), box_w, box_h,
                boxstyle="round,pad=0.1",
                facecolor='white', edgecolor='#2980B9', linewidth=1.2, zorder=3
            ))
            ax.text(3.1, y_pos + 0.4, "포트폴리오 팁",
                    ha='center', va='center', fontsize=11, fontweight='bold', color='#2980B9', zorder=4)
            ax.text(3.1, y_pos + 0.9, textwrap.fill(tip, width=35),
                    ha='center', va='center', fontsize=9, color='#333333', linespacing=1.5, zorder=4)

        if insight:
            ax.add_patch(mpatches.FancyBboxPatch(
                (6.3, y_pos), box_w, box_h,
                boxstyle="round,pad=0.1",
                facecolor='white', edgecolor='#27AE60', linewidth=1.2, zorder=3
            ))
            ax.text(8.9, y_pos + 0.4, "시장 인사이트",
                    ha='center', va='center', fontsize=11, fontweight='bold', color='#27AE60', zorder=4)
            ax.text(8.9, y_pos + 0.9, textwrap.fill(insight, width=35),
                    ha='center', va='center', fontsize=9, color='#333333', linespacing=1.5, zorder=4)


# ════════════════════════════════════════════════════════════════════════════
# 3번 담당자(UI)가 호출할 통합 인터페이스
# ════════════════════════════════════════════════════════════════════════════

class RoadmapEngine:
    """
    UI 담당자가 드롭박스 값을 넘기면 로드맵 이미지 + 텍스트 설명을 반환하는 파사드.
    """

    DROPDOWN_OPTIONS = {
        "job": ["백엔드", "프론트엔드", "AI_ML", "DevOps", "풀스택"],
        "company": ["스타트업", "대기업"],
        "experience": ["비전공자", "전공자(재학/졸업)", "1~2년차", "3년차 이상"],
        "current_stack": [
            "없음", "Python", "Java", "JavaScript", "TypeScript",
            "C++", "Kotlin", "Go", "React", "Spring Boot",
            "Django", "FastAPI", "Docker", "AWS",
        ],
        "goal_period": ["3개월", "6개월", "1년"],
    }

    def __init__(
            self,
            adapter_path: Optional[str] = None,
            output_dir: str = "outputs/roadmaps",
    ):
        self.adapter_path = adapter_path
        self.visualizer = RoadmapVisualizer(output_dir)
        self._ft_model = None  # 지연 로드

    def _get_model(self):
        """모델 지연 로드 (첫 generate 호출 시)."""
        if self._ft_model is None:
            if self.adapter_path and os.path.exists(self.adapter_path):
                # 실제 모델이 있을 경우 불러오기 (fine_tuning.py 필요)
                try:
                    from fine_tuning import QLoRAFineTuner
                    self._ft_model = QLoRAFineTuner.load_for_inference(self.adapter_path)
                except ImportError:
                    self._ft_model = _FallbackModel()
            else:
                # adapter 없으면 fallback-only 더미 사용
                self._ft_model = _FallbackModel()
        return self._ft_model

    def generate(
            self,
            profile: dict,
            save_path: Optional[str] = None,
            show: bool = False,
    ) -> dict:
        """
        profile dict → 로드맵 이미지 생성 + 결과 반환.
        """
        model = self._get_model()
        roadmap = model.infer_roadmap(profile)
        image_path = self.visualizer.render(roadmap, profile, save_path, show)

        summaries = [
            f"{s['step']}단계 [{s['title']}] — "
            f"{', '.join(s.get('techs', [])[:3])} ({s.get('duration', '')})"
            for s in roadmap.get("steps", [])
        ]

        return {
            "image_path": image_path,
            "roadmap": roadmap,
            "step_summaries": summaries,
            "title": roadmap.get("title", ""),
            "market_insight": roadmap.get("market_insight", ""),
            "portfolio_tip": roadmap.get("portfolio_tip", ""),
        }


class _FallbackModel:
    """adapter 없을 때 프로필에 맞춰 직관적인 데이터를 뱉어주는 더미 모델."""

    def infer_roadmap(self, profile: dict) -> dict:
        job = profile.get("job", "개발자")
        company = profile.get("company", "스타트업")

        # 직무에 따라 다른 로드맵 생성
        if job == "프론트엔드":
            steps = [
                {"step": 1, "title": "웹 기본기 마스터", "duration": "4주", "techs": ["HTML/CSS", "JS"],
                 "description": "웹 브라우저의 동작 원리와 자바스크립트 핵심 문법을 익힙니다.", "milestone": "정적 웹페이지 클론 코딩"},
                {"step": 2, "title": "React 생태계", "duration": "4주", "techs": ["React", "Router"],
                 "description": "컴포넌트 기반 UI 설계와 SPA(Single Page Application) 개발을 학습합니다.",
                 "milestone": "React 기반 Todo 앱"},
                {"step": 3, "title": "상태 관리 체득", "duration": "4주", "techs": ["Redux", "Zustand"],
                 "description": "복잡한 앱의 전역 상태를 관리하는 모던 기법을 다룹니다.", "milestone": "장바구니 기능 구현"},
                {"step": 4, "title": "타입 안정성", "duration": "4주", "techs": ["TypeScript"],
                 "description": "대규모 협업을 위한 정적 타입 시스템을 프로젝트에 도입합니다.", "milestone": "기존 프로젝트 TS 마이그레이션"},
                {"step": 5, "title": "성능 최적화", "duration": "4주", "techs": ["Next.js", "Vite"],
                 "description": "SSR/SSG 개념과 번들링 최적화로 초기 렌더링 속도를 개선합니다.", "milestone": "Lighthouse 점수 90점 달성"},
                {"step": 6, "title": "최종 배포", "duration": "4주", "techs": ["Vercel", "Git"],
                 "description": "지속적 통합/배포(CI/CD)를 구축하고 포트폴리오를 완성합니다.", "milestone": "이력서 첨부용 프로젝트 배포"}
            ]
            market = "대기업 프론트엔드는 React와 TypeScript 조합을 필수 자격 요건으로 삼는 비율이 80% 이상입니다."

        elif job == "AI_ML":
            steps = [
                {"step": 1, "title": "파이썬 및 수학 기본", "duration": "4주", "techs": ["Python", "Numpy"],
                 "description": "AI 구현을 위한 파이썬 심화 문법과 선형대수학을 학습합니다.", "milestone": "데이터 처리 스크립트 작성"},
                {"step": 2, "title": "데이터 분석", "duration": "4주", "techs": ["Pandas", "EDA"],
                 "description": "정형 데이터를 분석하고 시각화하여 인사이트를 도출합니다.", "milestone": "Kaggle 타이타닉 생존자 예측"},
                {"step": 3, "title": "머신러닝 기초", "duration": "4주", "techs": ["Scikit-learn"],
                 "description": "회귀, 분류, 군집화 등 전통적인 머신러닝 알고리즘을 배웁니다.", "milestone": "머신러닝 기반 분류기 완성"},
                {"step": 4, "title": "딥러닝 프레임워크", "duration": "4주", "techs": ["PyTorch", "CUDA"],
                 "description": "신경망 기초와 PyTorch를 활용한 딥러닝 모델링을 시작합니다.", "milestone": "이미지 분류 모델 훈련"},
                {"step": 5, "title": "자연어/비전 심화", "duration": "4주", "techs": ["Transformers", "CNN"],
                 "description": "HuggingFace 생태계와 최신 모델 아키텍처를 실습합니다.", "milestone": "텍스트 요약 봇 구현"},
                {"step": 6, "title": "모델 배포 (MLOps)", "duration": "4주", "techs": ["FastAPI", "Docker"],
                 "description": "학습된 AI 모델을 실제 API 서버로 서빙하는 방법을 익힙니다.", "milestone": "AI API 서비스 배포"}
            ]
            market = "스타트업 AI 직군은 연구뿐만 아니라 모델을 직접 서버에 올릴 수 있는 엔지니어링 역량을 크게 우대합니다."

        else:
            steps = [
                {"step": 1, "title": "기초 언어 마스터", "duration": "4주", "techs": ["Python", "Git"],
                 "description": "개발의 기본이 되는 언어 문법과 버전 관리 시스템을 학습합니다.", "milestone": "간단한 터미널 게임 제작"},
                {"step": 2, "title": "웹 프레임워크", "duration": "4주", "techs": ["FastAPI", "Django"],
                 "description": "백엔드 API 서버를 구축하는 방법을 배웁니다.", "milestone": "RESTful API 설계 및 구현"},
                {"step": 3, "title": "데이터베이스 설계", "duration": "4주", "techs": ["MySQL", "Redis"],
                 "description": "데이터 모델링과 쿼리 최적화 기법을 집중 학습합니다.", "milestone": "게시판 ERD 설계 및 연동"},
                {"step": 4, "title": "클라우드 인프라", "duration": "4주", "techs": ["AWS EC2", "Linux"],
                 "description": "만든 서버를 실제 클라우드 환경에 배포하고 관리하는 법을 배웁니다.", "milestone": "클라우드 환경에 서버 배포"},
                {"step": 5, "title": "가상화 및 컨테이너", "duration": "4주", "techs": ["Docker", "Nginx"],
                 "description": "환경에 구애받지 않는 안정적인 배포 파이프라인을 구축합니다.", "milestone": "Docker 기반 무중단 배포"},
                {"step": 6, "title": "최종 프로젝트", "duration": "4주", "techs": ["포트폴리오", "최적화"],
                 "description": "지금까지 배운 기술을 모두 통합하여 실무 수준의 프로젝트를 완성합니다.", "milestone": "이력서 첨부용 프로젝트 완성"}
            ]
            market = "최근 스타트업 백엔드 채용 시장에서는 Docker를 활용한 배포 경험과 FastAPI의 수요가 급격히 증가하고 있습니다."

        return {
            "title": f"{company} {job} 6개월 완성 로드맵",
            "summary": "채용 시장 트렌드 분석 기반 최적화 커리큘럼",
            "total_weeks": 24,
            "steps": steps,
            "portfolio_tip": "기술 선택 이유를 반드시 README에 적어주세요. 단순 클론 코딩보다 문제 해결 과정을 중시합니다.",
            "market_insight": market
        }


# ════════════════════════════════════════════════════════════════════════════
# 직접 실행 (테스트)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("🗺  로드맵 시각화 테스트 (중앙 타임라인 레이아웃)")
    print("=" * 60)

    # ── 3개의 테스트 케이스 ────────────────────────────────────────
    test_cases = [
        {
            "job": "백엔드",
            "company": "스타트업",
            "experience": "비전공자",
            "current_stack": ["Python"],
            "goal_period": "6개월",
        },
        {
            "job": "프론트엔드",
            "company": "대기업",
            "experience": "전공자(재학/졸업)",
            "current_stack": ["HTML/CSS", "JavaScript"],
            "goal_period": "1년",
        },
        {
            "job": "AI_ML",
            "company": "스타트업",
            "experience": "1~2년차",
            "current_stack": ["Python", "PyTorch"],
            "goal_period": "6개월",
        },
    ]

    engine = RoadmapEngine(adapter_path=None, output_dir="outputs/roadmaps")

    for profile in test_cases:
        print(f"\n▶ {profile['job']} / {profile['company']} / {profile['goal_period']}")
        result = engine.generate(profile)
        print(f"  이미지: {result['image_path']}")
        print(f"  제목:   {result['title']}")
        for s in result["step_summaries"]:
            print(f"    {s}")

    print("\n드롭박스 선택지 (UI 담당자 참고):")
    for k, v in RoadmapEngine.DROPDOWN_OPTIONS.items():
        print(f"  {k}: {v}")