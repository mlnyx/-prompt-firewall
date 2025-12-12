# -*- coding: utf-8 -*-
"""
중앙 설정 파일

모든 모듈에서 사용하는 공통 설정값을 정의합니다.
"""
import os
from typing import Dict, Tuple

# ===== 경로 설정 =====
# __file__ 기준 3단계 상위 디렉토리 계산
_file_based_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Colab 중복 경로 수정: 마지막 -prompt-firewall까지만 유지
# 예: /content/-prompt-firewall/-prompt-firewall/-prompt-firewall → /content/-prompt-firewall
parts = _file_based_root.split(os.sep)
if "-prompt-firewall" in parts:
    # 첫 번째 -prompt-firewall까지만 경로 구성
    first_idx = parts.index("-prompt-firewall")
    PROJECT_ROOT = os.sep.join(parts[:first_idx + 1])
else:
    PROJECT_ROOT = _file_based_root

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# ===== 일반 설정 =====
# 분석 결과를 저장할 로그 파일명
LOG_FILE = 'firewall_log.csv'

# 테스트 데이터 경로
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.csv")

# ===== Stage 1: 규칙 기반 필터 설정 =====
STAGE1_RULES_PATH = os.path.join(PROJECT_ROOT, "stage1_rules.yaml")

# ===== Stage 2: ML 스코어러 모델 경로 =====
# ONNX 모델 경로
PROTECTAI_PATH = os.path.join(MODEL_DIR, "protectai-deberta-v3-base")
SENTINEL_PATH = os.path.join(MODEL_DIR, "prompt-injection-sentinel")

# Hugging Face 모델 ID (온라인 다운로드)
PIGUARD_ID = "leolee99/PIGuard"
TESTSAVANTAI_ID = "testsavantai/prompt-injection-defender-base-v0"

# 모델별 비대칭 가중치
# W_Low: 모델이 '안전'하다고 판단했을 때의 가중치
# W_High: 모델이 '공격'이라고 판단했을 때의 가중치
ASYMMETRIC_WEIGHTS: Dict[str, Tuple[float, float]] = {
    "protectai": (0.8, 0.2),
    "sentinel":  (0.3, 0.7),
    "piguard":   (0.6, 0.6),
    "savantai":  (0.6, 0.8),
}

# 최종 점수 임계값
THRESHOLD_LOW = 0.25   # 이 값 미만은 ALLOW
THRESHOLD_HIGH = 0.60  # 이 값 이상은 BLOCK

class Decision:
    """결정 유형"""
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    ESCALATE = "ESCALATE"
    REWRITE = "REWRITE"
    ERROR_S1 = "S1_ERROR"
    ERROR_S2 = "S2_ERROR"
    ERROR_S3 = "S3_ERROR"
    NOT_APPLICABLE = "N/A"

# ===== Stage 2: ML 스코어러 임계값 =====
# 이 값들은 V2.0 스펙을 기반으로 설정되었습니다

# 허용 임계값: 점수 < 0.25면 ALLOW
STAGE2_ALLOW_THRESHOLD = 0.25

# 차단 임계값: 점수 >= 0.60이면 BLOCK
# 0.25 <= 점수 < 0.60이면 그레이 영역 (Stage 3으로 이관)
STAGE2_BLOCK_THRESHOLD = 0.60

# ===== Stage 3: 재작성 모듈 설정 =====
# Stage3Rewriter 컴포넌트의 설정값
REWRITER_CONFIG = {
    # 의미 유사도 측정에 사용할 모델
    "similarity_model": "all-MiniLM-L6-v2",
    # 재작성된 텍스트의 안전성 검사 임계값
    # Stage 3은 이미 그레이 영역 프롬프트를 받으므로 0.50으로 설정 (원래 0.25)
    "risk_threshold": 0.50,
    # 원본과 재작성 텍스트의 의미적 유사도 임계값
    "similarity_threshold": 0.85,
}