# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Firewall components
from stage1_filter import Stage1Filter
from stage2_scorer import Stage2Scorer
from config import Decision

# --- Application Setup ---
app = FastAPI(
    title="Prompt Firewall API",
    description="A two-stage firewall to detect and mitigate prompt injection attacks.",
    version="1.0.0",
)

# --- Data Models ---
class PromptRequest(BaseModel):
    text: str

class FirewallResponse(BaseModel):
    decision: str
    stage: int
    rule_id: Optional[str] = None
    risk_score: Optional[float] = None
    message: str

# --- Firewall Initialization ---
# Singleton instances to avoid reloading models on every request
s1_filter = Stage1Filter()
s2_scorer = Stage2Scorer()

print("API is ready to accept requests.")

# --- API Endpoints ---
@app.post("/check-prompt", response_model=FirewallResponse)
async def check_prompt(request: PromptRequest):
    """
    Analyzes a user-provided prompt through the two-stage firewall.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    # --- Stage 1: Rule-based Filter ---
    s1_decision, s1_rule_id, s1_message = s1_filter.filter_text(request.text)

    if s1_decision != Decision.ESCALATE:
        # Stage 1 made a final decision (ALLOW or BLOCK)
        return FirewallResponse(
            decision=s1_decision,
            stage=1,
            rule_id=s1_rule_id,
            message=s1_message,
        )

    # --- Stage 2: ML-based Scorer ---
    if not s2_scorer.models_loaded:
        # If S2 models are not available, return a default safe decision
        return FirewallResponse(
            decision=Decision.REWRITE,
            stage=2,
            message="Stage 1 escalated, but Stage 2 models are not available. Defaulting to REWRITE.",
        )
        
    s2_decision, s2_risk_score = s2_scorer.predict(request.text)
    
    return FirewallResponse(
        decision=s2_decision,
        stage=2,
        risk_score=round(s2_risk_score, 4) if s2_risk_score is not None else None,
        message=f"Evaluated by Stage 2 scorer. Risk score: {s2_risk_score:.4f}",
    )

@app.get("/")
async def root():
    return {"message": "Prompt Firewall API is running. Use the /docs endpoint for documentation."}

# To run the server, use the following command in your terminal:
# uvicorn main:app --reload
