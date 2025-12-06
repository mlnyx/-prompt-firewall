# -*- coding: utf-8 -*-
from datetime import datetime

# 방화벽 단계별 모듈
from . import stage1_filter
from . import stage2_scorer
from .stage3_rewriter import Stage3Rewriter, SAFE_SUMMARY_MSG

# 중앙 설정값
from ..utils.config import STAGE2_ALLOW_THRESHOLD, STAGE2_BLOCK_THRESHOLD

def firewall_pipeline(prompt: str, rewriter: Stage3Rewriter):
    """
    1단계부터 3단계까지 전체 방화벽 파이프라인을 실행하고 처리 흐름을 추적합니다.
    """
    log_data = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "user_prompt": prompt,
        "flow_trace": [],
        "stage1_result": "N/A",
        "stage2_score": -1.0,
        "stage3_result": "N/A",
        "final_decision": "N/A",
        "final_output": ""
    }

    # --- 1단계: 규칙 기반 필터 ---
    s1_result = stage1_filter.check(prompt)
    log_data["stage1_result"] = s1_result

    if s1_result != "ESCALATE":
        log_data["final_decision"] = s1_result
        log_data["final_output"] = prompt if s1_result == "ALLOW" else SAFE_SUMMARY_MSG
        log_data["flow_trace"].append(f"1단계(규칙 필터): 최종 판정을 내렸습니다 -> {s1_result}")
        return log_data
    
    log_data["flow_trace"].append(f"1단계(규칙 필터): 'ESCALATE' (2단계로 이관)")

    # --- 2단계: ML 스코어러 ---
    s2_score = stage2_scorer.predict(prompt)
    log_data["stage2_score"] = s2_score

    if s2_score < STAGE2_ALLOW_THRESHOLD:
        log_data["final_decision"] = "ALLOW"
        log_data["final_output"] = prompt
        log_data["flow_trace"].append(f"2단계(ML 스코어러): 점수({s2_score:.4f})가 허용 임계값({STAGE2_ALLOW_THRESHOLD})보다 낮아 'ALLOW' 판정을 내렸습니다.")
    elif s2_score >= STAGE2_BLOCK_THRESHOLD:
        log_data["final_decision"] = "BLOCK"
        log_data["final_output"] = SAFE_SUMMARY_MSG
        log_data["flow_trace"].append(f"2단계(ML 스코어러): 점수({s2_score:.4f})가 차단 임계값({STAGE2_BLOCK_THRESHOLD})보다 높아 'BLOCK' 판정을 내렸습니다.")
    else: # 회색지대 (Gray Area)
        log_data["flow_trace"].append(f"2단계(ML 스코어러): 점수({s2_score:.4f})가 회색지대에 해당하여 3단계로 이관합니다.")
        
        # --- 3단계: 안전 재작성 ---
        rewritten_text = rewriter.rewrite(prompt)
        
        if rewritten_text != SAFE_SUMMARY_MSG:
            stage3_decision = "재작성 성공 (REWRITTEN)"
            log_data["final_decision"] = "REWRITTEN_AND_ALLOWED"
            log_data["final_output"] = rewritten_text
        else:
            stage3_decision = "재작성 실패 (REWRITE_FAILED)"
            log_data["final_decision"] = "BLOCK"
            log_data["final_output"] = SAFE_SUMMARY_MSG

        log_data["stage3_result"] = stage3_decision
        log_data["flow_trace"].append(f"3단계(LLM 정화): 재작성을 시도했습니다 -> {stage3_decision}")
            
    return log_data

