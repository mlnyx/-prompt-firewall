# -*- coding: utf-8 -*-
"""
유틸리티 함수 모듈

로깅, 결과 기록, CLI 출력 포맷팅 등 공통 기능을 담당합니다.
"""
import os
import csv
from .config import LOG_FILE

def setup_logging():
    """
    로그 파일 초기화
    
    CSV 로그 파일이 없으면 생성하고 헤더 행을 추가합니다.
    """
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # CSV 헤더 정의
            writer.writerow([
                "timestamp", "user_prompt", "stage1_result", "stage2_score",
                "stage3_result", "final_decision", "final_output"
            ])

def log_result(log_data):
    """
    분석 결과를 CSV 파일에 기록
    
    Args:
        log_data: 방화벽 파이프라인의 분석 결과 딕셔너리
    """
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            log_data.get("timestamp"),
            log_data.get("user_prompt"),
            log_data.get("stage1_result"),
            f'{log_data.get("stage2_score", -1):.4f}',  # 점수를 4자리 소수점까지 표시
            log_data.get("stage3_result"),
            log_data.get("final_decision"),
            log_data.get("final_output")
        ])

def format_cli_output(log_data):
    """
    CLI 출력용으로 분석 결과를 포맷팅하여 출력
    
    최종 결정에 따라 색상 구분:
    - ALLOW: 초록색
    - BLOCK: 빨간색
    - REWRITTEN_AND_ALLOWED: 노란색
    
    Args:
        log_data: 방화벽 파이프라인의 분석 결과 딕셔너리
    """
    # 결정 결과별 색상 정의
    decision_colors = {
        "ALLOW": "\033[92m",  # 초록색
        "BLOCK": "\033[91m",  # 빨간색
        "REWRITTEN_AND_ALLOWED": "\033[93m",  # 노란색
    }
    ENDC = "\033[0m"  # 색상 리셋
    color = decision_colors.get(log_data['final_decision'], "")
    
    # 최종 결정을 한글로 변환
    decision_kr = {
        "ALLOW": "허용",
        "BLOCK": "차단",
        "REWRITTEN_AND_ALLOWED": "재작성 후 허용"
    }
    decision_text = decision_kr.get(log_data['final_decision'], log_data['final_decision'])

    # ===== 헤더 =====
    print("\n" + "="*80)
    print("LLM 프롬프트 방화벽 분석 보고서".center(80))
    print("="*80)
    
    # ===== 기본 정보 =====
    print(f"\n[기본 정보]")
    print(f"분석 시간:   {log_data['timestamp']}")
    print(f"입력 프롬프트: \"{log_data['user_prompt']}\"")
    print()
    
    # ===== 모델 및 설정 정보 =====
    from .config import STAGE2_ALLOW_THRESHOLD, STAGE2_BLOCK_THRESHOLD, REWRITER_CONFIG
    print(f"[모델 및 설정 정보]")
    print(f"Stage 1 (규칙 필터):     패턴 매칭 기반")
    print(f"Stage 2 (ML 스코어):     4개 모델 (ProtectAI, Sentinels, etc.)")
    print(f"  - 허용 임계값:        < {STAGE2_ALLOW_THRESHOLD}")
    print(f"  - 그레이 영역:        {STAGE2_ALLOW_THRESHOLD} ~ {STAGE2_BLOCK_THRESHOLD}")
    print(f"  - 차단 임계값:        >= {STAGE2_BLOCK_THRESHOLD}")
    print(f"Stage 3 (LLM 재작성):    Llama 3 8B Instruct")
    print(f"  - 재작성 임계값:      < {REWRITER_CONFIG['risk_threshold']}")
    print(f"  - 의미 유사도 임계값:  >= {REWRITER_CONFIG['similarity_threshold']}")
    print()
    
    # ===== 분석 흐름 =====
    print(f"[분석 흐름]")
    for i, trace in enumerate(log_data.get("flow_trace", []), 1):
        # 영어 용어를 한글로 변환
        translated_trace = trace
        
        # 주요 용어 변환
        translations = {
            "Rule Filter": "규칙 필터",
            "ML Scorer": "ML 스코어러",
            "LLM Rewriting": "LLM 재작성",
            "ESCALATE": "다음 단계로 이관",
            "passing to Stage 2": "→ Stage 2로",
            "passing to Stage 3": "→ Stage 3으로",
            "Made final decision": "최종 결정",
            "below allow threshold": "허용 임계값 이하",
            "in gray area": "그레이 영역",
            "above block threshold": "차단 임계값 이상",
            "decision:": "결정:",
            "Rewriting attempt": "재작성 시도",
            "Rewriting Successful (REWRITTEN)": "재작성 성공",
            "Rewriting Failed (REWRITE_FAILED)": "재작성 실패",
        }
        
        for eng, kor in translations.items():
            translated_trace = translated_trace.replace(eng, kor)
        
        print(f"  {i}. {translated_trace}")
    print()
    print("-" * 80)
    
    # ===== 상세 점수 =====
    print(f"\n[상세 분석 결과]")
    print(f"Stage 1 결과:      {log_data.get('stage1_result', 'N/A')}")
    if log_data.get('stage2_score', -1) >= 0:
        score = log_data.get('stage2_score', 0)
        risk_level = "낮음" if score < STAGE2_ALLOW_THRESHOLD else \
                     "중간" if score < STAGE2_BLOCK_THRESHOLD else "높음"
        print(f"Stage 2 점수:      {score:.4f} ({risk_level})")
    print(f"Stage 3 결과:      {log_data.get('stage3_result', 'N/A')}")
    print()
    
    # ===== 최종 판정 =====
    print("="*80)
    print(f"\n[최종 판정]")
    print(f"결정: {color}{decision_text}{ENDC} ({log_data['final_decision']})")
    print(f"최종 출력:\n  \"{log_data['final_output']}\"")
    print("\n" + "="*80)
    print(f"전체 로그 저장: {LOG_FILE}\n")
