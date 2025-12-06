# -*- coding: utf-8 -*-
"""
중앙 설정 파일

모든 모듈에서 사용하는 공통 설정값을 정의합니다.
"""

# ===== 일반 설정 =====
# 분석 결과를 저장할 로그 파일명
LOG_FILE = 'firewall_log.csv'

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