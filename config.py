# config.py
import os
from typing import Dict, Tuple

# --- 경로 설정 ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- 데이터 및 규칙 경로 ---
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.csv")
STAGE1_RULES_PATH = os.path.join(PROJECT_ROOT, "stage1_rules.yaml")
EVALUATION_OUTPUT_DIR = os.path.join(DATA_DIR, "evaluation_results")

# --- 공통 상수 ---
# 결정(Decision) 유형
class Decision:
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    ESCALATE = "ESCALATE"
    REWRITE = "REWRITE"
    ERROR_S1 = "S1_ERROR"
    ERROR_S2 = "S2_ERROR"
    NOT_APPLICABLE = "N/A"

# 데이터 레이블 유형
class Label:
    BENIGN = "benign"
    MALICIOUS = "jailbreak"

# --- Stage 2 Scorer 설정 ---
# 모델 경로
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
PROTECTAI_PATH = os.path.join(MODEL_DIR, "protectai-deberta-v3-base")
SENTINEL_PATH = os.path.join(MODEL_DIR, "prompt-injection-sentinel")

# Hugging Face 모델 ID
PIGUARD_ID = "leolee99/PIGuard"
TESTSAVANTAI_ID = "testsavantai/prompt-injection-defender-base-v0"

# 모델별 비대칭 가중치 (W_Low, W_High)
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
# 사이 값은 REWRITE
