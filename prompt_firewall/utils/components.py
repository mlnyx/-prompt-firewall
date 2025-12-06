# -*- coding: utf-8 -*-
"""
Stage3Rewriter 모델과 같이 무거운 공유 컴포넌트를 초기화하고 관리합니다.
앱 전체에서 무거운 모델이 여러 번 로드되는 것을 방지합니다.
"""
from .config import REWRITER_CONFIG
from ..core.stage3_rewriter import Stage3Rewriter

print("공유 컴포넌트를 초기화합니다...")

# Stage3Rewriter 인스턴스를 앱 시작 시 한 번만 로드하여 다른 모듈에서 가져와 사용할 수 있도록 합니다.
rewriter = Stage3Rewriter(
    model_name=REWRITER_CONFIG["similarity_model"],
    risk_threshold=REWRITER_CONFIG["risk_threshold"],
    similarity_threshold=REWRITER_CONFIG["similarity_threshold"]
)

print("공유 컴포넌트 초기화 완료.")
