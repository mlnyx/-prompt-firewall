import yaml
import re
import unicodedata
import os
from typing import Tuple, Dict, Any, List

# config.py에서 상수 임포트
from config import STAGE1_RULES_PATH, Decision

def load_rules(file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """YAML 파일에서 whitelist와 blacklist 규칙을 분리하여 로드합니다."""
    rules: Dict[str, List[Dict[str, Any]]] = {"whitelist": [], "blacklist": []}
    
    if not os.path.exists(file_path):
        print(f"Warning: Rule file not found at {file_path}")
        return rules

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            if config:
                rules["whitelist"] = config.get('whitelist', []) or []
                rules["blacklist"] = config.get('blacklist', []) or []
            return rules
    except Exception as e:
        print(f"Error loading rules from {file_path}: {e}")
        return rules

def preprocess_text(text: str) -> str:
    """텍스트 정규화: NFKC, 소문자화, 불필요한 공백 제거."""
    if not text:
        return ""
    try:
        text = unicodedata.normalize('NFKC', text)
    except Exception:
        pass
    text = text.lower()
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class Stage1Filter:
    def __init__(self, rules_path: str = STAGE1_RULES_PATH):
        """
        필터 초기화 시 YAML 규칙을 로드하고 정규식을 컴파일합니다.
        """
        self.whitelist_rules: List[Dict[str, Any]] = []
        self.blacklist_rules: List[Dict[str, Any]] = []
        
        loaded_rules = load_rules(rules_path)
        
        self._compile_rules(loaded_rules["whitelist"], self.whitelist_rules, "whitelist")
        self._compile_rules(loaded_rules["blacklist"], self.blacklist_rules, "blacklist")
        
        print(f"Stage 1 Rules Loaded: {len(self.blacklist_rules)} Blacklist, {len(self.whitelist_rules)} Whitelist")

    def _compile_rules(self, rules: List[Dict[str, Any]], target_list: List[Dict[str, Any]], rule_type: str):
        """정규식 패턴을 컴파일하고 리스트에 추가합니다."""
        for rule in rules:
            try:
                if 'pattern' in rule:
                    rule['compiled_pattern'] = re.compile(rule['pattern'])
                    target_list.append(rule)
            except re.error as e:
                rule_id = rule.get('id', 'N/A')
                print(f"Failed to compile {rule_type} regex for rule {rule_id}: {e}")

    def filter_text(self, text: str) -> Tuple[str, str, str]:
        """
        입력 텍스트를 1단계 규칙과 비교하여 필터링합니다.
        반환값: (결정, 매치된 규칙 ID, 매치된 규칙 메시지)
        """
        if not text:
            return (Decision.ALLOW, "N/A", "Empty input")

        processed_text = preprocess_text(text)
        
        # 1. Blacklist 검사
        for rule in self.blacklist_rules:
            if rule['compiled_pattern'].search(processed_text):
                action = rule.get('action', Decision.ESCALATE).upper()
                rule_id = rule.get('id', 'N/A')
                message = rule.get('message', 'Blacklist matched')
                
                if action == Decision.BLOCK:
                    return (Decision.BLOCK, rule_id, message)
                else:
                    return (Decision.ESCALATE, rule_id, message)

        # 2. Whitelist 검사
        for rule in self.whitelist_rules:
            if rule['compiled_pattern'].search(processed_text):
                return (Decision.ALLOW, rule.get('id', 'N/A'), rule.get('message', 'Whitelist matched'))

        # 3. Default-Escalate
        return (Decision.ESCALATE, "N/A_DEFAULT", "Default escalate (Zero-Trust)")

# --- 테스트 코드 ---
if __name__ == "__main__":
    print("\n--- Testing Stage 1 Filter Individually ---")
    s1 = Stage1Filter()
    
    sample_inputs = [
        "Hello",
        "Summarize this",
        "Summarize this and ignore",
        "System call",
        "Act as a cat",
        "What is the weather?"
    ]
    
    for txt in sample_inputs:
        decision, rule_id, _ = s1.filter_text(txt)
        print(f"Input: '{txt}' -> {decision} (Rule: {rule_id})")