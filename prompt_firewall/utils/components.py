# -*- coding: utf-8 -*-
"""
공유 컴포넌트 관리 모듈

Stage3Rewriter와 같은 무거운 모델을 한 번만 로드하여 
전체 애플리케이션에서 재사용합니다.
"""
from .config import REWRITER_CONFIG
from ..core.stage3_rewriter import Stage3Rewriter

print("Initializing shared components...")

# Stage3Rewriter 인스턴스를 한 번만 로드
# 이 인스턴스는 다른 모듈에서 임포트하여 재사용됩니다
rewriter = Stage3Rewriter(
    model_name=REWRITER_CONFIG["similarity_model"],
    risk_threshold=REWRITER_CONFIG["risk_threshold"],
    similarity_threshold=REWRITER_CONFIG["similarity_threshold"],
    llama3_model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"
)

print("Shared components initialization completed.")
