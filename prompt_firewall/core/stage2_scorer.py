# -*- coding: utf-8 -*-

# This is a functional placeholder for Stage 2.
# A real implementation would load and run actual ML models.

def predict(text: str) -> float:
    """
    Analyzes the text and returns a risk score between 0.0 and 1.0.
    """
    text_lower = text.lower()
    
    # High-risk keywords -> High score (definitely BLOCK)
    high_risk_words = ["secret", "api key", "ignore all previous instructions"]
    if any(word in text_lower for word in high_risk_words):
        print(f"[Debug Stage 2] High-risk keyword found. Score: 0.95")
        return 0.95

    # Medium-risk keywords -> Gray Area (needs Stage 3)
    medium_risk_words = ["execute a command", "delete file", "run script", "install"]
    if any(word in text_lower for word in medium_risk_words):
        print(f"[Debug Stage 2] Medium-risk keyword found. Score: 0.45")
        return 0.45

    # Low-risk keywords -> Low score (likely ALLOW)
    low_risk_words = ["history", "explain", "what is", "how does", "에펠탑", "설명", "원인", "무엇인가요"]
    if any(word in text_lower for word in low_risk_words):
        print(f"[Debug Stage 2] Low-risk keyword found. Score: 0.10")
        return 0.10

    # Default for unknown patterns
    print(f"[Debug Stage 2] No specific keywords found. Defaulting to gray area. Score: 0.30")
    return 0.30
