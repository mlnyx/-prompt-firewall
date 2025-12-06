# -*- coding: utf-8 -*-
import os
import csv
from config import LOG_FILE

def setup_logging():
    """Initializes the log file with a header if it doesn't exist."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "user_prompt", "stage1_result", "stage2_score",
                "stage3_result", "final_decision", "final_output"
            ])

def log_result(log_data):
    """Appends a single analysis result to the CSV log file."""
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            log_data.get("timestamp"),
            log_data.get("user_prompt"),
            log_data.get("stage1_result"),
            f'{log_data.get("stage2_score", -1):.4f}',
            log_data.get("stage3_result"),
            log_data.get("final_decision"),
            log_data.get("final_output")
        ])

def format_cli_output(log_data):
    """
    Formats the analysis result for a rich and readable CLI output.
    """
    # Decision color mapping
    decision_colors = {
        "ALLOW": "\033[92m",  # Green
        "BLOCK": "\033[91m",  # Red
        "REWRITTEN_AND_ALLOWED": "\033[93m", # Yellow
    }
    ENDC = "\033[0m"
    color = decision_colors.get(log_data['final_decision'], "")

    # Header
    print("\n" + "═"*70)
    print("🛡️  LLM PROMPT FIREWALL ANALYSIS REPORT 🛡️")
    print("═"*70)
    
    # Basic Info
    print(f"🔹 **Timestamp:** {log_data['timestamp']}")
    print(f"🔹 **Input Prompt:** \"{log_data['user_prompt']}\" ")
    print("-" * 70)

    # Detailed Flow
    print("🔎 **Analysis Flow:**")
    for trace in log_data.get("flow_trace", []):
        print(f"   ➡️  {trace}")
    print("-" * 70)

    # Final Verdict
    print("✅ **Final Verdict:**")
    print(f"   - **Decision:** {color}{log_data['final_decision']}{ENDC}")
    print(f"   - **Final Output:** \"{log_data['final_output']}\" ")
    print("═"*70)
    print(f"\nⓘ  Full log saved to {LOG_FILE}")
