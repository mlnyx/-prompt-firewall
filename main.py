import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

# --- 1. S1, S2 모듈 임포트 ---
# (이 파일들이 main.py와 같은 폴더에 있어야 함)
try:
    from stage1_filter import Stage1Filter
    from stage2_scorer import Stage2Scorer
except ImportError:
    print("오류: 'stage1_filter.py' 또는 'stage2_scorer.py'를 찾을 수 없습니다.")
    exit()

# --- 2. FastAPI 앱 및 모델 로드 ---
# (앱 실행 시 S1, S2 모델을 '미리' 로드하여 메모리에 상주)
print("FastAPI: 애플리케이션을 시작합니다...")
app = FastAPI(title="LLM Prompt Firewall")

print("FastAPI: S1(규칙 필터)을 로드합니다...")
s1_filter = Stage1Filter()

print("FastAPI: S2(ML 스코어러)를 로드합니다...")
s2_scorer = Stage2Scorer()
print("FastAPI: 모든 모듈 로드 완료. 서버 준비 완료.")

# --- 3. API 요청/응답 모델 정의 ---
class PromptRequest(BaseModel):
    """
    /check_prompt 엔드포인트가 받을 요청 바디
    """
    text: str

class PromptResponse(BaseModel):
    """
    /check_prompt 엔드포인트가 반환할 응답 바디
    """
    decision: str       # 최종 결정 (ALLOW, BLOCK, REWRITE)
    stage: int          # 결정이 내려진 단계 (1 또는 2)
    risk_score: float   # S2가 계산한 위험 점수 (S1 결정 시 0.0 또는 1.0)
    rule_id: str | None = None # S1에서 매치된 규칙 ID

# --- 4. FastAPI 엔드포인트 정의 ---
@app.get("/")
def read_root():
    """
    서버가 살아있는지 확인하는 루트 엔드포인트
    """
    return {"status": "LLM Prompt Firewall API is running."}

@app.post("/check_prompt", response_model=PromptResponse)
async def check_prompt_pipeline(request: PromptRequest):
    """
    메인 파이프라인: S1(규칙) -> S2(ML)
    """
    start_time = time.time()
    
    if not request.text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    # === 1단계: S1(규칙 필터) 실행 ===
    try:
        s1_decision, s1_rule_id, _ = s1_filter.filter_text(request.text)
    except Exception as e:
        print(f"Error during S1 processing: {e}")
        raise HTTPException(status_code=500, detail="Internal error in Stage 1")

    # S1이 '즉시 허용' 또는 '즉시 차단'한 경우, S2 없이 바로 반환
    if s1_decision == "ALLOW":
        return PromptResponse(
            decision="ALLOW", 
            stage=1, 
            risk_score=0.0, 
            rule_id=s1_rule_id
        )
    if s1_decision == "BLOCK":
        return PromptResponse(
            decision="BLOCK", 
            stage=1, 
            risk_score=1.0, 
            rule_id=s1_rule_id
        )

    # === 2단계: S2(ML 스코어러) 실행 ===
    # (S1이 'ESCALATE'한 경우에만 S2 실행)
    if s1_decision == "ESCALATE":
        if not s2_scorer.models_loaded:
            print("Warning: S2 models not loaded, returning REWRITE as fallback.")
            return PromptResponse(
                decision="REWRITE", # 모델 로드 실패 시 보수적으로 REWRITE
                stage=2,
                risk_score=0.5,
                rule_id=s1_rule_id
            )
            
        try:
            s2_decision, s2_risk_score = s2_scorer.predict(request.text)
        except Exception as e:
            print(f"Error during S2 processing: {e}")
            raise HTTPException(status_code=500, detail="Internal error in Stage 2")
        
        # S2의 최종 결정 반환 (ALLOW, BLOCK, REWRITE)
        return PromptResponse(
            decision=s2_decision, 
            stage=2, 
            risk_score=s2_risk_score, 
            rule_id=s1_rule_id # S1에서 ESCALATE된 규칙 ID (e.g., N/A_DEFAULT)
        )
    
    # (S1이 ALLOW/BLOCK/ESCALATE 외의 값을 반환한 경우 - 예외 처리)
    raise HTTPException(status_code=500, detail=f"Invalid decision from Stage 1: {s1_decision}")


# --- 5. (선택) uvicorn으로 서버 실행 ---
if __name__ == "__main__":
    print("--- FastAPI 서버를 'main:app'으로 실행합니다 (http://127.0.0.1:8000) ---")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)