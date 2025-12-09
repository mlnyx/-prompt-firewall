# -*- coding: utf-8 -*-
"""
Stage 1: 규칙 기반 필터

블랙리스트와 화이트리스트 패턴을 이용한 빠른 위협 탐지.
공격 패턴이 명확하면 즉시 차단, 안전한 패턴이면 통과, 애매한 경우는 Stage 2로 이관.
"""
import re
import yaml
import os
from typing import Dict, List, Any, Tuple

from ..utils.config import STAGE1_RULES_PATH, Decision

# ===== 전역 변수: 컴파일된 규칙 캐시 =====
_compiled_whitelist_rules = []
_compiled_blacklist_rules = []
_rules_loaded = False

def _load_and_compile_rules():
    """YAML 규칙 파일을 로드하고 정규식을 컴파일합니다."""
    global _compiled_whitelist_rules, _compiled_blacklist_rules, _rules_loaded
    
    if _rules_loaded:
        return
    
    try:
        if not os.path.exists(STAGE1_RULES_PATH):
            print(f"[Stage 1] Warning: 규칙 파일을 찾을 수 없습니다: {STAGE1_RULES_PATH}")
            print(f"[Stage 1] 기본 패턴으로 계속 진행합니다.")
            _rules_loaded = True
            return
        
        with open(STAGE1_RULES_PATH, 'r', encoding='utf-8') as f:
            rules_data = yaml.safe_load(f)
        
        if not rules_data:
            print(f"[Stage 1] Warning: 규칙 파일이 비어있습니다: {STAGE1_RULES_PATH}")
            _rules_loaded = True
            return
        
        # 화이트리스트 규칙 컴파일
        for rule in rules_data.get('whitelist', []):
            try:
                compiled_rule = {
                    'id': rule.get('id'),
                    'pattern': re.compile(rule['pattern']),
                    'action': rule.get('action', 'allow'),
                    'message': rule.get('message', 'Whitelist matched')
                }
                _compiled_whitelist_rules.append(compiled_rule)
            except Exception as e:
                print(f"[Stage 1] Warning: 화이트리스트 규칙 컴파일 실패 ({rule.get('id')}): {e}")
        
        # 블랙리스트 규칙 컴파일
        for rule in rules_data.get('blacklist', []):
            try:
                compiled_rule = {
                    'id': rule.get('id'),
                    'pattern': re.compile(rule['pattern']),
                    'action': rule.get('action', 'block'),
                    'message': rule.get('message', 'Blacklist matched')
                }
                _compiled_blacklist_rules.append(compiled_rule)
            except Exception as e:
                print(f"[Stage 1] Warning: 블랙리스트 규칙 컴파일 실패 ({rule.get('id')}): {e}")
        
        print(f"[Stage 1] 규칙 로드 완료: 화이트리스트 {len(_compiled_whitelist_rules)}, 블랙리스트 {len(_compiled_blacklist_rules)}")
        _rules_loaded = True
        
    except Exception as e:
        print(f"[Stage 1] Error: 규칙 파일 로드 실패: {e}")
        _rules_loaded = True

class Stage1Filter:
    """규칙 기반 프롬프트 필터"""
    
    def __init__(self):
        _load_and_compile_rules()
    
    def filter_text(self, text: str) -> Tuple[str, str, str]:
        """
        입력 텍스트를 규칙 기반으로 검사
        
        검사 순서:
        1. 블랙리스트 패턴 매칭 (악의적 키워드/구문) → BLOCK
        2. 화이트리스트 패턴 매칭 (안전한 질문) → ALLOW
        3. 둘 다 아니면 Stage 2로 이관 → ESCALATE
        
        Args:
            text: 검사할 입력 텍스트
        
        Returns:
            (decision, rule_id, message): (결정, 규칙ID, 메시지)
        """
        # ===== 블랙리스트 검사 (YAML 규칙 사용) =====
        if _compiled_blacklist_rules:
            for rule in _compiled_blacklist_rules:
                if rule['pattern'].search(text):
                    print(f"[Stage 1] 블랙리스트 매칭: {rule['id']} - {rule['message']}")
                    return (Decision.BLOCK, rule['id'], rule['message'])
        
        # ===== 화이트리스트 검사 (YAML 규칙 사용) =====
        if _compiled_whitelist_rules:
            for rule in _compiled_whitelist_rules:
                if rule['pattern'].search(text):
                    print(f"[Stage 1] 화이트리스트 매칭: {rule['id']} - {rule['message']}")
                    return (Decision.ALLOW, rule['id'], rule['message'])
        
        # ===== 기본값: Stage 2로 이관 (Zero-Trust) =====
        return (Decision.ESCALATE, None, "No rule matched, escalating to Stage 2")

def check(text: str) -> str:
    """전역 함수 버전 (호환성용)"""
    filter_obj = Stage1Filter()
    decision, _, _ = filter_obj.filter_text(text)
    return decision
