# -*- coding: utf-8 -*-
"""
Centralized configuration for the LLM Prompt Firewall project.
"""

# --- General Settings ---
LOG_FILE = 'firewall_log.csv'

# --- Stage 2: ML Scorer Thresholds ---
# These values are based on the V2.0 specification.
STAGE2_ALLOW_THRESHOLD = 0.25  # Risk < 0.25 -> ALLOW
STAGE2_BLOCK_THRESHOLD = 0.60  # 0.25 <= Risk < 0.60 -> GRAY AREA, Risk >= 0.60 -> BLOCK

# --- Stage 3: Rewriter Settings ---
# Configuration for the Stage3Rewriter component.
REWRITER_CONFIG = {
    "similarity_model": "all-MiniLM-L6-v2",
    "risk_threshold": STAGE2_ALLOW_THRESHOLD, # The same threshold is used inside the rewriter's safety check.
    "similarity_threshold": 0.85,
}