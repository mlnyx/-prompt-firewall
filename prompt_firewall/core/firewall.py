# -*- coding: utf-8 -*-
"""
방화벽 파이프라인 모듈

3단계 방화벽 파이프라인을 조율하고 전체 분석 플로우를 관리합니다.
"""
from datetime import datetime
import unicodedata

# 각 단계별 필터 및 스코어러 모듈
from . import stage1_filter
from . import stage2_scorer
from .stage3_rewriter import Stage3Rewriter, SAFE_SUMMARY_MSG

# 임계값 및 설정 값
from ..utils.config import STAGE2_ALLOW_THRESHOLD, STAGE2_BLOCK_THRESHOLD

def _preprocess_nfkc(text: str) -> str:
    """
    Pre-process: NFKC 정규화
    
    유니코드 호환 정규화(NFKC)를 적용하여 다양한 입력 형식을 통일합니다.
    예: ２０１９년 → 2019년, ℓ → l
    
    Args:
        text: 입력 텍스트
    
    Returns:
        정규화된 텍스트
    """
    return unicodedata.normalize('NFKC', text)

def firewall_pipeline(prompt: str, rewriter: Stage3Rewriter):
    """
    LLM 프롬프트 방화벽 파이프라인 실행
    
    Pre-process: NFKC 정규화
    Stage 1: 규칙 기반 필터 (블랙리스트/화이트리스트)
    Stage 2: ML 기반 위험도 스코어러 (확률 모델)
    Stage 3: LLM 기반 안전 재작성 (그레이 영역 처리)
    
    Args:
        prompt: 분석할 사용자 입력 프롬프트
        rewriter: Stage3Rewriter 인스턴스
    
    Returns:
        분석 결과를 담은 딕셔너리
    """
    # ========== Pre-process: NFKC 정규화 ==========
    normalized_prompt = _preprocess_nfkc(prompt)
    
    # 로그 데이터 구조 초기화
    log_data = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "user_prompt": prompt,  # 원본 입력 기록
        "flow_trace": [],  # 분석 흐름 추적
        "stage1_result": "N/A",
        "stage2_score": -1.0,
        "stage3_result": "N/A",
        "final_decision": "N/A",
        "final_output": ""
    }
    
    # 정규화된 프롬프트와 원본이 다르면 기록
    if normalized_prompt != prompt:
        log_data["flow_trace"].append(f"Pre-process (NFKC Normalization): '{prompt}' -> '{normalized_prompt}'")

    # ========== Stage 1: 규칙 기반 필터 ==========
    s1_result = stage1_filter.check(normalized_prompt)
    log_data["stage1_result"] = s1_result

    # Stage 1에서 최종 결정이 나면 즉시 반환
    if s1_result != "ESCALATE":
        log_data["final_decision"] = s1_result
        log_data["final_output"] = prompt if s1_result == "ALLOW" else SAFE_SUMMARY_MSG
        log_data["flow_trace"].append(f"Stage 1 (Rule Filter): Made final decision -> {s1_result}")
        return log_data
    
    log_data["flow_trace"].append(f"Stage 1 (Rule Filter): ESCALATE (passing to Stage 2)")

    # ========== Stage 2: ML 스코어러 ==========
    s2_score = stage2_scorer.predict(normalized_prompt)
    log_data["stage2_score"] = s2_score

    # 허용 임계값보다 낮으면 ALLOW
    if s2_score < STAGE2_ALLOW_THRESHOLD:
        log_data["final_decision"] = "ALLOW"
        log_data["final_output"] = prompt
        log_data["flow_trace"].append(f"Stage 2 (ML Scorer): Score ({s2_score:.4f}) below allow threshold ({STAGE2_ALLOW_THRESHOLD}), decision: ALLOW")
    # 차단 임계값 이상이면 BLOCK
    elif s2_score >= STAGE2_BLOCK_THRESHOLD:
        log_data["final_decision"] = "BLOCK"
        log_data["final_output"] = SAFE_SUMMARY_MSG
        log_data["flow_trace"].append(f"Stage 2 (ML Scorer): Score ({s2_score:.4f}) above block threshold ({STAGE2_BLOCK_THRESHOLD}), decision: BLOCK")
    # 그 사이는 그레이 영역 -> Stage 3으로 이관
    else:
        log_data["flow_trace"].append(f"Stage 2 (ML Scorer): Score ({s2_score:.4f}) in gray area, passing to Stage 3")
        
        # ========== Stage 3: LLM 기반 안전 재작성 ==========
        rewritten_text = rewriter.rewrite(normalized_prompt)
        
        # 재작성 성공 여부 판정
        if rewritten_text != SAFE_SUMMARY_MSG:
            stage3_decision = "Rewriting Successful (REWRITTEN)"
            log_data["final_decision"] = "REWRITTEN_AND_ALLOWED"
            log_data["final_output"] = rewritten_text
        else:
            stage3_decision = "Rewriting Failed (REWRITE_FAILED)"
            log_data["final_decision"] = "BLOCK"
            log_data["final_output"] = SAFE_SUMMARY_MSG

        log_data["stage3_result"] = stage3_decision
        log_data["flow_trace"].append(f"Stage 3 (LLM Rewriting): Rewriting attempt -> {stage3_decision}")
            
    return log_data

