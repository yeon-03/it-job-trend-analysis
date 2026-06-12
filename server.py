from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import importlib.util
from pydantic import BaseModel
from core.roadmap_visualization import RoadmapEngine
import os

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

spec = importlib.util.spec_from_file_location("ragbot", "core/rag_core.py")
ragbot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ragbot)

collection = ragbot.load_vector_db()

roadmap_engine = RoadmapEngine(adapter_path="models/roadmap_llm/adapter")

@app.get("/api/job-monthly")
def job_monthly():
    df = pd.read_csv("data/analysis/job_monthly.csv")
    return df.to_dict(orient="records")

@app.get("/api/cluster_final_summary")
def cluster_summary():
    df = pd.read_csv("data/analysis/cluster_final_summary.csv")
    return df.to_dict(orient="records")

@app.get("/api/tech-growth")
def tech_growth():
    df = pd.read_csv("data/analysis/tech_growth.csv")
    return df.to_dict(orient="records")

@app.get("/api/tech-monthly")
def tech_monthly():
    df = pd.read_csv("data/analysis/tech_monthly_pct.csv")
    return df.to_dict(orient="records")

@app.get("/api/trend-summary")
def trend_summary():
    with open("models/trend_summary.json", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/company-size")
def company_summary():
    with open("data/analysis/company_size.json", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/job-by-size")
def job_by_size():
    df = pd.read_csv("data/analysis/job_category_by_size.csv", encoding="utf-8-sig")
    return df.to_dict(orient="records")

@app.get("/api/tech-by-size")
def tech_by_size():
    df = pd.read_csv("data/analysis/tech_ratio_by_size.csv", encoding="utf-8-sig")
    return df.to_dict(orient="records")
    

@app.get("/api/job-growth")
def job_growth():
    df = pd.read_csv("data/analysis/job_monthly.csv", encoding="utf-8-sig")
    df = df.rename(columns={df.columns[0]: "month"})
    jobs = ['백엔드', '프론트엔드', 'AI_ML', 'DevOps', '데이터', '모바일']
    first, last = df.iloc[0], df.iloc[-1]
    result = []
    for job in jobs:
        f, l = int(first[job]), int(last[job])
        growth = round((l - f) / f * 100, 1) if f else 0
        result.append({"분야": job, "증감률(%)": growth, "공고수": l})
    return result


    
class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
def chat(req: ChatRequest):
    answer = ragbot.get_rag_response(req.message, collection)
    return {"answer": answer}


class RoadmapRequest(BaseModel):
    job: str
    company: str
    experience: str
    current_stack: list[str]
    goal_period: str

@app.post("/api/roadmap")
def roadmap(req: RoadmapRequest):
    result = roadmap_engine.generate({
        "job": req.job,
        "company": req.company,
        "experience": req.experience,
        "current_stack": req.current_stack,
        "goal_period": req.goal_period,
    })
    # 파일 경로를 브라우저용 URL로 변환
    img_url = "/roadmaps/" + os.path.basename(result["image_path"])
    return {
        "image_url": img_url,
        "title": result["title"],
        "step_summaries": result["step_summaries"],
        "market_insight": result["market_insight"],
        "portfolio_tip": result["portfolio_tip"],
    }

app.mount("/roadmaps", StaticFiles(directory="outputs/roadmaps"), name="roadmaps")
app.mount("/", StaticFiles(directory="static", html=True), name="static")