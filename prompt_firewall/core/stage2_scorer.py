# -*- coding: utf-8 -*-
"""
Stage 2: ML 기반 위험도 스코어러

사전정의된 위험 키워드들을 바탕으로 프롬프트의 위험도를 0.0~1.0 사이의 점수로 평가.
실제 운영 환경에서는 딥러닝 모델 앙상블로 대체 가능.
"""

def predict(text: str) -> float:
    """
    입력 텍스트의 위험도를 점수로 평가
    
    점수 범위:
    - 0.0 ~ 0.25: 낮음 (ALLOW)
    - 0.25 ~ 0.60: 중간 (그레이 영역 -> Stage 3로 이관)
    - 0.60 ~ 1.0: 높음 (BLOCK)
    
    Args:
        text: 평가할 입력 텍스트
    
    Returns:
        위험도 점수 (0.0 ~ 1.0)
    """
    text_lower = text.lower()
    
    # ===== 높은 위험도 키워드 -> 점수 0.95 (확정 차단) =====
    high_risk_words = ["secret", "api key", "ignore all previous instructions"]
    if any(word in text_lower for word in high_risk_words):
        print(f"[Stage 2] 높은 위험도 키워드 감지. 점수: 0.95")
        return 0.95

    # ===== 중간 위험도 키워드 -> 점수 0.45 (그레이 영역) =====
    medium_risk_words = ["execute a command", "delete file", "run script", "install"]
    if any(word in text_lower for word in medium_risk_words):
        print(f"[Stage 2] 중간 위험도 키워드 감지. 점수: 0.45")
        return 0.45

    # ===== 낮은 위험도 키워드 -> 점수 0.10 (안전) =====
    low_risk_words = ["history", "explain", "what is", "how does", "eiffel tower", "explain", "reason"]
    if any(word in text_lower for word in low_risk_words):
        print(f"[Stage 2] 낮은 위험도 키워드 감지. 점수: 0.10")
        return 0.10

    # ===== 기본값: 그레이 영역 (확실하지 않은 경우) =====
    print(f"[Stage 2] 특정 키워드를 찾을 수 없음. 그레이 영역 점수: 0.30")
    return 0.30
