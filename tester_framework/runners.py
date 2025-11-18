# tester_framework/runners.py
from abc import ABC, abstractmethod
from .core import Seed
from config import Decision

# 테스트할 필터/스코어러 로직을 임포트합니다.
# 의존성 주입을 위해 try-except 블록을 유지합니다.
try:
    from stage1_filter import Stage1Filter
except ImportError:
    Stage1Filter = None
    print("Warning: 'stage1_filter' not found. Stage1LocalRunner will not work.")

try:
    from stage2_scorer import Stage2Scorer
except ImportError:
    Stage2Scorer = None
    print("Warning: 'stage2_scorer' not found. Stage2LocalRunner will not work.")


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