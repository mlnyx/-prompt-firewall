# -*- coding: utf-8 -*-
"""
Main entry point for the LLM Prompt Firewall CLI.
"""
import argparse

# Import shared components and utilities
from components import rewriter
from firewall import firewall_pipeline
from utils import setup_logging, log_result, format_cli_output

def main():
    """
    Parses command-line arguments and runs the firewall pipeline.
    """
    parser = argparse.ArgumentParser(
        description="🛡️ LLM Prompt Firewall CLI - Analyze a prompt for security risks."
    )
    parser.add_argument(
        "prompt", 
        type=str, 
        help="The user prompt to analyze."
    )
    args = parser.parse_args()

    try:
        # Initialize logging
        setup_logging()

        # Execute the pipeline using the shared rewriter instance
        print("Analyzing prompt...")
        result = firewall_pipeline(args.prompt, rewriter)

        # Log the result to a file and format the output for the console
        log_result(result)
        format_cli_output(result)

    except FileNotFoundError as e:
        print(f"❌ [Error] A required file was not found: {e}. Please ensure all dependencies are installed.")
    except Exception as e:
        print(f"💥 [Critical Error] An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()