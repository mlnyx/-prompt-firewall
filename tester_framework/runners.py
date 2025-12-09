# tester_framework/runners.py
from abc import ABC, abstractmethod
from .core import Seed
from prompt_firewall.utils.config import Decision

# 테스트할 필터/스코어러/리라이터 로직을 임포트합니다.
# 의존성 주입을 위해 try-except 블록을 유지합니다.
try:
    from prompt_firewall.core.stage1_filter import Stage1Filter
except ImportError:
    Stage1Filter = None
    print("Warning: 'stage1_filter' not found. Stage1LocalRunner will not work.")

try:
    from prompt_firewall.core.stage2_scorer import Stage2Scorer
except ImportError:
    Stage2Scorer = None
    print("Warning: 'stage2_scorer' not found. Stage2LocalRunner will not work.")

try:
    from prompt_firewall.core.stage3_rewriter import Stage3Rewriter
except ImportError:
    Stage3Rewriter = None
    print("Warning: 'stage3_rewriter' not found. Stage3LocalRunner will not work.")


class IFilterRunner(ABC):
    """테스트 실행기 인터페이스 (전략 패턴)"""
    @abstractmethod
    def run(self, seed: Seed) -> Seed:
        """주어진 Seed를 테스트하고, 결과가 업데이트된 Seed 객체를 반환합니다."""
        pass

class Stage1LocalRunner(IFilterRunner):
    """Stage 1 필터를 로컬에서 실행하는 실행기"""
    def __init__(self):
        if Stage1Filter:
            self.filter_instance = Stage1Filter()
        else:
            self.filter_instance = None

    def run(self, seed: Seed) -> Seed:
        """Stage1Filter의 filter_text()를 호출하고 S1 결과를 Seed에 저장합니다."""
        if not self.filter_instance:
            seed.error = "Stage1Filter is not available."
            seed.s1_decision = Decision.ERROR_S1
            return seed
            
        try:
            decision, rule_id, msg = self.filter_instance.filter_text(seed.data)
            seed.s1_decision = decision
            seed.s1_rule_id = rule_id
            seed.s1_message = msg
        except Exception as e:
            seed.error = str(e)
            seed.s1_decision = Decision.ERROR_S1
            
        return seed
            
        return seed
    
class Stage2LocalRunner(IFilterRunner):
    """Stage 2 스코어러를 로컬에서 실행하는 실행기"""
    def __init__(self):
        if Stage2Scorer:
            self.scorer_instance = Stage2Scorer()
        else:
            self.scorer_instance = None

    def run(self, seed: Seed) -> Seed:
        """Stage2Scorer의 predict()를 호출하고 S2 결과를 Seed에 저장합니다."""
        if not self.scorer_instance:
            seed.error = "Stage2Scorer is not available."
            seed.s2_decision = Decision.ERROR_S2
            return seed

        try:
            # S1 결정이 ESCALATE인 경우에만 S2 실행
            if seed.s1_decision == Decision.ESCALATE:
                decision, risk_score = self.scorer_instance.predict(seed.data)
                seed.s2_decision = decision
                seed.s2_risk_score = risk_score
            else:
                # S2를 실행할 필요가 없는 경우는 N/A로 명확히 표시
                seed.s2_decision = Decision.NOT_APPLICABLE
                seed.s2_risk_score = None
        except Exception as e:
            seed.error = str(e)
            seed.s2_decision = Decision.ERROR_S2
            
        return seed


class Stage3LocalRunner(IFilterRunner):
    """Stage 3 리라이터를 로컬에서 실행하는 실행기"""
    def __init__(self, use_local_llm: bool = True, llama3_model_id: str = "meta-llama/Llama-3-8B-Instruct"):
        if Stage3Rewriter:
            try:
                self.rewriter_instance = Stage3Rewriter(
                    stage1_filter=Stage1Filter() if Stage1Filter else None,
                    stage2_scorer=Stage2Scorer() if Stage2Scorer else None,
                    use_local_llm=use_local_llm,
                    llama3_model_id=llama3_model_id
                )
            except Exception as e:
                print(f"Warning: Failed to initialize Stage3Rewriter: {e}")
                self.rewriter_instance = None
        else:
            self.rewriter_instance = None

    def run(self, seed: Seed) -> Seed:
        """Stage3Rewriter의 rewrite()를 호출하고 S3 결과를 Seed에 저장합니다.
        
        팀원 사양: rewrite()는 dict 반환
        {
            "rewrite": 재작성된_프롬프트,
            "sim_score": 유사도_점수,
            "safe_score": 안전성_점수,
            "contains_danger": 위험_키워드_포함_여부,
            "final_decision": "pass" or "fail",
            "reason": 상세_사유
        }
        """
        if not self.rewriter_instance:
            seed.error = "Stage3Rewriter is not available."
            seed.s3_decision = Decision.ERROR_S3
            return seed

        try:
            # S2 결정이 REWRITE인 경우에만 S3 실행
            if hasattr(seed, 's2_decision') and seed.s2_decision == Decision.REWRITE:
                result = self.rewriter_instance.rewrite(seed.data)
                
                # 결과 dict에서 정보 추출
                final_decision = result.get("final_decision", "fail")
                rewritten_text = result.get("rewrite", "")
                sim_score = result.get("sim_score", 0.0)
                safe_score = result.get("safe_score", 0.0)
                contains_danger = result.get("contains_danger", False)
                reason = result.get("reason", "")
                
                # S3 결정: "pass"면 ALLOW, "fail"이면 BLOCK
                seed.s3_decision = Decision.ALLOW if final_decision == "pass" else Decision.BLOCK
                seed.s3_rewritten_prompt = rewritten_text
                seed.s3_confidence = sim_score  # 유사도를 신뢰도로 사용
                seed.s3_safe_score = safe_score
                seed.s3_similarity = sim_score
                seed.s3_contains_danger = contains_danger
                seed.s3_final_decision = final_decision
                seed.s3_reason = reason  # 상세 사유
            else:
                # S3를 실행할 필요가 없는 경우
                seed.s3_decision = Decision.NOT_APPLICABLE
                seed.s3_rewritten_prompt = None
                seed.s3_confidence = None
                seed.s3_safe_score = None
                seed.s3_reason = "Stage 2에서 REWRITE 판정 없음"
        except Exception as e:
            seed.error = str(e)
            seed.s3_decision = Decision.ERROR_S3
            seed.s3_reason = f"오류: {str(e)}"
            
        return seed