# -*- coding: utf-8 -*-
import re

# This is a functional placeholder for Stage 1.
# A real implementation would load rules from stage1_rules.yaml.

def check(text: str) -> str:
    """
    Checks the input text against a predefined set of blacklist/whitelist rules.
    """
    # Example blacklist: block common command injection/SQLi attempts or harmful requests
    blacklist_patterns = [
        r'\b(rm -rf|execute|eval|os\.system|subprocess)\b',
        r'\b(SELECT\s.*FROM\s.*|INSERT\s.*INTO\s.*|DROP\s.*TABLE)\b',
        r'<script.*?>',
        r'password'
    ]

    for pattern in blacklist_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            print(f"[Debug Stage 1] Blacklist pattern found: '{pattern}'")
            return "BLOCK"

    # Example whitelist: simple, safe questions
    whitelist_patterns = [
        r'^\s*(what is|who is|tell me about|explain)\s'
    ]

    for pattern in whitelist_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            print(f"[Debug Stage 1] Whitelist pattern found: '{pattern}'")
            return "ALLOW"

    print("[Debug Stage 1] No specific pattern found. Escalating to Stage 2.")
    return "ESCALATE"
