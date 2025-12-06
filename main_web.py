# -*- coding: utf-8 -*-
"""
LLM 프롬프트 방화벽 웹 서버

FastAPI를 사용하여 REST API 및 웹 UI를 제공합니다.
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

# 공유 컴포넌트 및 유틸리티 임포트
from prompt_firewall.core.firewall import firewall_pipeline
from prompt_firewall.utils.components import rewriter
from prompt_firewall.utils.utils import setup_logging, log_result

# ===== FastAPI 앱 초기화 =====
app = FastAPI(title="LLM Prompt Firewall")

# 템플릿 디렉토리 설정
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ===== 요청 모델 정의 =====
class AnalyzeRequest(BaseModel):
    """프롬프트 분석 요청 모델"""
    prompt: str

# ===== 서버 시작 이벤트 =====
@app.on_event("startup")
def startup_event():
    """
    서버 시작 시 실행
    
    로깅 설정을 초기화합니다.
    """
    setup_logging()
    print("Server startup complete. Logging is configured.")

# ===== API 엔드포인트 =====
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    메인 HTML 페이지 제공
    
    웹 브라우저에서 프롬프트 분석 인터페이스를 제공합니다.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/analyze")
async def analyze_prompt(request: AnalyzeRequest):
    """
    프롬프트 분석 API 엔드포인트
    
    POST 요청으로 프롬프트를 받아 방화벽 파이프라인을 실행하고 결과를 반환합니다.
    
    Args:
        request: 분석할 프롬프트를 포함한 요청 객체
    
    Returns:
        분석 결과 JSON 객체
    
    Raises:
        HTTPException: Rewriter 컴포넌트를 사용할 수 없는 경우 503 에러 반환
    """
    # Rewriter 컴포넌트 확인
    if not rewriter:
        raise HTTPException(status_code=503, detail="Rewriter component is not available.")
    
    # 파이프라인 실행
    result = firewall_pipeline(request.prompt, rewriter)
    
    # 결과 로깅
    log_result(result)
    
    return result

if __name__ == "__main__":
    # 개발 서버 실행
    # 명령어: uvicorn main_web:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
