# -*- coding: utf-8 -*-
"""
Stage 1: 규칙 기반 필터

블랙리스트와 화이트리스트 패턴을 이용한 빠른 위협 탐지.
공격 패턴이 명확하면 즉시 차단, 안전한 패턴이면 통과, 애매한 경우는 Stage 2로 이관.
"""
import re

def check(text: str) -> str:
    """
    입력 텍스트를 규칙 기반으로 검사
    
    검사 순서:
    1. 블랙리스트 패턴 매칭 (악의적 키워드/구문)
    2. 화이트리스트 패턴 매칭 (안전한 질문)
    3. 둘 다 아니면 Stage 2로 이관
    
    Args:
        text: 검사할 입력 텍스트
    
    Returns:
        "BLOCK": 악의적인 패턴 감지
        "ALLOW": 안전한 패턴 감지
        "ESCALATE": 명확한 판단 불가능 -> Stage 2로 이관
    """
    
    # ===== 블랙리스트: 명확히 악의적인 패턴 =====
    blacklist_patterns = [
        # 명령어 주입 시도
        r'\b(rm -rf|execute|eval|os\.system|subprocess)\b',
        # SQL 주입 시도
        r'\b(SELECT\s.*FROM\s.*|INSERT\s.*INTO\s.*|DROP\s.*TABLE)\b',
        # XSS 시도
        r'<script.*?>',
        # 민감 정보 요청
        r'password'
    ]

    for pattern in blacklist_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            print(f"[Stage 1] 블랙리스트 패턴 감지: '{pattern}'")
            return "BLOCK"

    # ===== 화이트리스트: 일반적으로 안전한 패턴 =====
    whitelist_patterns = [
        # 정보 요청 질문
        r'^\s*(what is|who is|tell me about|explain)\s'
    ]

    for pattern in whitelist_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            print(f"[Stage 1] 화이트리스트 패턴 감지: '{pattern}'")
            return "ALLOW"

    # ===== 기본값: 명확한 판단 불가능 -> Stage 2로 이관 =====
    print("[Stage 1] 특정 패턴을 찾을 수 없음. Stage 2로 이관합니다.")
    return "ESCALATE"
