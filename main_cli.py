# -*- coding: utf-8 -*-
"""
LLM 프롬프트 방화벽 CLI 진입점

커맨드라인 인터페이스를 통해 프롬프트 분석 기능을 제공합니다.
"""
import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

# 공유 컴포넌트 및 유틸리티 임포트
from prompt_firewall.utils.components import rewriter
from prompt_firewall.core.firewall import firewall_pipeline
from prompt_firewall.utils.utils import setup_logging, log_result, format_cli_output

def main():
    """
    CLI 메인 함수
    
    명령줄 인자를 파싱하고 방화벽 파이프라인을 실행합니다.
    """
    # 인자 파서 설정
    parser = argparse.ArgumentParser(
        description="LLM 프롬프트 방화벽 CLI - 프롬프트의 보안 위험도를 분석합니다."
    )
    parser.add_argument(
        "prompt", 
        type=str, 
        help="분석할 사용자 프롬프트"
    )
    args = parser.parse_args()

    try:
        # 로깅 초기화
        setup_logging()

        # 파이프라인 실행
        print("\n프롬프트 분석 중...\n")
        result = firewall_pipeline(args.prompt, rewriter)

        # 결과 로깅 및 출력
        log_result(result)
        format_cli_output(result)

    except FileNotFoundError as e:
        print(f"\n[오류] 필요한 파일을 찾을 수 없습니다: {e}")
        print("모든 종속성이 설치되었는지 확인하세요.\n")
    except Exception as e:
        print(f"\n[치명적 오류] 예상치 못한 오류가 발생했습니다: {e}\n")

if __name__ == "__main__":
    main()