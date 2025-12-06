# -*- coding: utf-8 -*-
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

# Import shared components and utilities
from firewall import firewall_pipeline
from components import rewriter
from utils import setup_logging, log_result

# --- FastAPI App Initialization ---
app = FastAPI(title="LLM Prompt Firewall")
templates = Jinja2Templates(directory="templates")

# --- Request Models ---
class AnalyzeRequest(BaseModel):
    prompt: str

# --- Server Startup Logic ---
@app.on_event("startup")
def startup_event():
    """Initializes logging on server startup."""
    setup_logging()
    print("Server startup complete. Logging is configured.")

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/analyze")
async def analyze_prompt(request: AnalyzeRequest):
    """
    Receives a prompt, runs it through the firewall pipeline, logs the result,
    and returns the analysis as JSON.
    """
    if not rewriter:
        raise HTTPException(status_code=503, detail="Rewriter component is not available.")
    
    # Execute the pipeline
    result = firewall_pipeline(request.prompt, rewriter)
    
    # Log the result
    log_result(result)
    
    return result

if __name__ == "__main__":
    # Recommended command for development: `uvicorn server:app --reload`
    uvicorn.run(app, host="0.0.0.0", port=8000)
