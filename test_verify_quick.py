#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
빠른 검증 스크립트 (Import + 기본 동작 확인)
서버 없이 코드 문법과 기본 기능만 검증합니다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("="*60)
print("1. 핵심 Import 검증")
print("="*60)

try:
    print("[1-1] Config 모듈...", end=" ")
    from prompt_firewall.utils.config import Decision, STAGE1_RULES_PATH, ASYMMETRIC_WEIGHTS
    print("✓")
    print(f"  - STAGE1_RULES_PATH: {STAGE1_RULES_PATH}")
    print(f"  - 비대칭 가중치 모델: {list(ASYMMETRIC_WEIGHTS.keys())}")
    print(f"  - Decision: {Decision.ALLOW}, {Decision.BLOCK}, {Decision.ESCALATE}, {Decision.REWRITE}")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

try:
    print("\n[1-2] Stage 1 Filter...", end=" ")
    from prompt_firewall.core.stage1_filter import Stage1Filter
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

try:
    print("[1-3] Stage 2 Scorer...", end=" ")
    from prompt_firewall.core.stage2_scorer import Stage2Scorer
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

try:
    print("[1-4] Stage 3 Rewriter...", end=" ")
    from prompt_firewall.core.stage3_rewriter import Stage3Rewriter
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

try:
    print("[1-5] Tester Framework...", end=" ")
    from tester_framework.core import Population, Seed
    from tester_framework.runners import Stage1LocalRunner, Stage2LocalRunner, Stage3LocalRunner
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

try:
    print("[1-6] Evaluate 모듈...", end=" ")
    from evaluate import run_stage1, run_stage2, run_stage3
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

print("\n" + "="*60)
print("2. 기본 기능 검증 (서버 없음)")
print("="*60)

try:
    print("\n[2-1] Stage 1 필터 테스트...", end=" ")
    s1 = Stage1Filter()
    decision, rule_id, msg = s1.filter_text("안녕하세요")
    assert decision in [Decision.ALLOW, Decision.BLOCK, Decision.ESCALATE]
    print(f"✓ (결정: {decision})")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

try:
    print("[2-2] Stage 2 스코어러 테스트...", end=" ")
    s2 = Stage2Scorer()
    decision, risk_score = s2.predict("안녕하세요")
    assert 0.0 <= risk_score <= 1.0
    assert decision in [Decision.ALLOW, Decision.BLOCK, Decision.ESCALATE]
    print(f"✓ (위험도: {risk_score:.2f}, 결정: {decision})")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

try:
    print("[2-3] Stage 3 리라이터 초기화...", end=" ")
    s3 = Stage3Rewriter(
        stage1_filter=Stage1Filter(),
        stage2_scorer=Stage2Scorer(),
        use_local_llm=False,  # Llama 3 안 쓰고 API만 사용
        llama3_model_id="meta-llama/Llama-3-8B-Instruct"
    )
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

print("\n" + "="*60)
print("3. 설정 값 검증")
print("="*60)

try:
    print("\n[3-1] STAGE1_RULES_PATH 존재 여부...", end=" ")
    import os
    if os.path.exists(STAGE1_RULES_PATH):
        print(f"✓ ({STAGE1_RULES_PATH})")
    else:
        print(f"⚠ 경로 없음 (폴백 모드 사용: {STAGE1_RULES_PATH})")
except Exception as e:
    print(f"✗ {e}")

try:
    print("[3-2] 모델 경로 설정...", end=" ")
    from prompt_firewall.utils.config import PROTECTAI_PATH, SENTINEL_PATH
    has_protectai = os.path.exists(PROTECTAI_PATH)
    has_sentinel = os.path.exists(SENTINEL_PATH)
    print(f"✓")
    print(f"  - ProtectAI: {'✓' if has_protectai else '⚠'} ({PROTECTAI_PATH})")
    print(f"  - Sentinel:  {'✓' if has_sentinel else '⚠'} ({SENTINEL_PATH})")
except Exception as e:
    print(f"✗ {e}")

try:
    print("\n[3-3] 비대칭 가중치 설정...", end=" ")
    print("✓")
    for model, (w_low, w_high) in ASYMMETRIC_WEIGHTS.items():
        print(f"  - {model:<12}: Low={w_low:.1f}, High={w_high:.1f}")
except Exception as e:
    print(f"✗ {e}")

print("\n" + "="*60)
print("✓ 모든 검증 통과!")
print("="*60)
print("\n다음 단계:")
print("1. CLI 테스트: python main_cli.py '테스트 프롬프트'")
print("2. 평가 실행: python evaluate.py --use-llm (필요시)")
print("3. 웹 서버: python main_web.py")
