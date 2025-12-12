import os
from collections import Counter
from typing import List, Dict, Any

# 리팩토링된 모듈 및 설정 임포트
from prompt_firewall.utils.config import (
    TEST_DATA_PATH,
    Decision,
    ASYMMETRIC_WEIGHTS,
    THRESHOLD_LOW,
    THRESHOLD_HIGH,
)
from tester_framework.core import Population, Seed
from tester_framework.runners import Stage1LocalRunner, Stage2LocalRunner, Stage3LocalRunner
from tester_framework.orchestrator import Tester

def display_config(runner_s2: Stage2LocalRunner):
    """Stage 2 Scorer의 설정을 터미널에 출력합니다."""
    if runner_s2.scorer_instance and runner_s2.scorer_instance.models_loaded:
        scorer = runner_s2.scorer_instance
        loaded_count = sum(1 for v in scorer.model_load_status.values() if v)
        print(f"[Stage 2] {loaded_count}/4 models loaded | Thresholds: {THRESHOLD_LOW} ~ {THRESHOLD_HIGH}")

def process_results(results: List[Seed], stage: str) -> Dict[str, int]:
    """테스트 결과를 요약합니다."""
    summary = Counter()
    decision_attr = f"s{stage}_decision"
    
    for seed in results:
        decision = getattr(seed, decision_attr)
        if decision:
            summary[f"{decision}_{seed.label}"] += 1
            
    summary["Total Seeds"] = len(results)
    return dict(summary)

def print_summary(stage_name: str, summary: Dict[str, int]):
    """결과 요약을 형식에 맞게 출력합니다."""
    print(f"\n[Summary S{stage_name}]")
    for key, value in sorted(summary.items()):
        if key != "Total Seeds":
            print(f"  {key}: {value}")
    print(f"  Total: {summary.get('Total Seeds', 0)}")

def run_stage1(population: Population) -> List[Seed]:
    """Stage 1 테스트를 실행합니다."""
    print("[Stage 1] 실행 중...")
    runner_s1 = Stage1LocalRunner()
    tester_s1 = Tester(population, runner_s1)
    return tester_s1.run_all()

def run_stage2(escalated_seeds: List[Seed]) -> List[Seed]:
    """Stage 2 테스트를 실행합니다."""
    if not escalated_seeds:
        return []
    
    print(f"[Stage 2] 실행 중 ({len(escalated_seeds)} seeds)...")
    escalated_population = Population(seeds=escalated_seeds)
    runner_s2 = Stage2LocalRunner()
    display_config(runner_s2)
    
    tester_s2 = Tester(escalated_population, runner_s2)
    return tester_s2.run_all()

def run_stage3(escalated_seeds: List[Seed], use_local_llm: bool = True) -> List[Seed]:
    """Stage 3 테스트를 실행합니다."""
    if not escalated_seeds:
        return []
    
    print(f"[Stage 3] 실행 중 ({len(escalated_seeds)} seeds) - LLM 사용: {use_local_llm}")
    escalated_population = Population(seeds=escalated_seeds)
    runner_s3 = Stage3LocalRunner(use_local_llm=use_local_llm)
    
    tester_s3 = Tester(escalated_population, runner_s3)
    return tester_s3.run_all()

def main():
    """메인 평가 파이프라인"""
    import argparse
    
    # 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser(description="3단계 프롬프트 방화벽 평가")
    parser.add_argument("--no-llm", action="store_true", help="Stage 3에서 LLM 사용 안함 (테스트용)")
    args = parser.parse_args()
    
    # 1. 데이터 로드
    population = Population()
    population.create_population_from_file(TEST_DATA_PATH)

    if not population.seeds:
        print(f"테스트할 데이터가 없습니다. '{TEST_DATA_PATH}' 파일을 확인하세요.")
        return

    # 2. Stage 1 실행 및 결과 처리
    results_s1 = run_stage1(population)
    summary_s1 = process_results(results_s1, stage='1')
    
    # 3. Stage 2 실행 및 결과 처리
    escalated_seeds_s2 = [seed for seed in results_s1 if seed.s1_decision == Decision.ESCALATE]
    results_s2 = run_stage2(escalated_seeds_s2)
    summary_s2 = process_results(results_s2, stage='2')


    #stage2에서 REWRITE인 것들을 추출 stage2_rewrites.txt로 저장
    stage2_rewrites = [seed for seed in results_s2 if seed.s2_decision == Decision.REWRITE]
    Population(seeds=stage2_rewrites).save_to_csv("stage2_rewrites.txt")

    # # 4. Stage 3 실행 및 결과 처리 (선택사항: REWRITE만 처리)
    escalated_seeds_s3 = [seed for seed in results_s2 if hasattr(seed, 's2_decision') and seed.s2_decision == Decision.REWRITE]
    results_s3 = run_stage3(escalated_seeds_s3, use_local_llm=not args.no_llm) if escalated_seeds_s3 else []
    summary_s3 = process_results(results_s3, stage='3') if results_s3 else {}

    # 5. 최종 결과 요약 출력
    print_summary("S1", summary_s1)
    if summary_s2:
        print_summary("S2", summary_s2)
    if summary_s3:
        print_summary("S3", summary_s3)

if __name__ == "__main__":
    main()