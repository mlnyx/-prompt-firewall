# tester_framework/orchestrator.py
from .core import Population, Seed
from .runners import IFilterRunner # Runner 의존성 주입
from typing import List

class Tester:
    """
    테스트 오케스트레이터
    Population과 Runner를 조합하여 테스트를 실행합니다.
    """
    def __init__(self, population: Population, runner: IFilterRunner):
        """
        테스트할 Population과 실행 전략(Runner)을 세팅한다. 
        """
        self.population = population
        self.runner = runner
        self.results: List[Seed] = []
        self.errors: List[Seed] = [] # 실행 중 에러가 발생한 시드

    def run_all(self):
        """
        Population의 모든 Seed에 대해 테스트를 실행합니다.
        """
        total = len(self.population)
        print(f"Running test for {total} seeds with {self.runner.__class__.__name__}...")
        
        for i, seed in enumerate(self.population):
            # 1. Runner에게 Seed 실행을 위임
            updated_seed = self.runner.run(seed)
            if updated_seed.error:
                self.errors.append(updated_seed)
            # 2. 결과 저장
            else : 
                self.results.append(updated_seed)
            
            # 진행 상황을 한 줄에 업데이트
            print(f"Processed {i + 1}/{total} seeds...", end='\r')
        
        print(f"\nTesting complete.") # Final newline after progress bar
        if self.errors:
            print(f"Total Errors: {len(self.errors)}")
        return self.results