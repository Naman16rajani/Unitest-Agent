#!/usr/bin/env python3
"""
CLI Runner for LLM-Optimized CrewAI Unit Test Generation System
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

from config import LLMProvider, get_default_config, create_llm_instance
from helpers.analysers.code_analyzer import CodeAnalyzer
from agents.unittest_agent.unittest_agent import UnitTestAgent

from workflow import workflow


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="LLM-Optimized CrewAI Unit Test Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_runner.py                              # Use default sample file
  python cli_runner.py sample_code/calculator.py   # Test specific file
  python cli_runner.py --llm openai sample.py      # Use OpenAI
  python cli_runner.py --help                      # Show this help

Supported LLM providers: gemini, openai, claude, ollama
        """,
    )

    parser.add_argument(
        "file",
        nargs="?",
        default="sample_code/calculator.py",
        help="Python file to generate tests for (default: sample_code/calculator.py)",
    )

    parser.add_argument(
        "--llm",
        choices=["gemini", "openai", "claude", "ollama"],
        default="gemini",
        help="LLM provider to use (default: gemini)",
    )

    parser.add_argument(
        "--output",
        default="./generated_tests",
        help="Output directory for generated tests (default: ./generated_tests)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Validate input file
    if not Path(args.file).exists():
        print(f"❌ Error: File '{args.file}' not found")
        print("\nTip: Make sure the file exists or try the default:")
        print("python cli_runner.py")
        sys.exit(1)

    # Header
    print("🚀 LLM-Optimized AI Unit Test Generation System")
    print("=" * 60)
    print(f"📄 Input file: {args.file}")
    print(f"🤖 LLM provider: {args.llm}")
    print(f"📁 Output directory: {args.output}")
    print()

    try:
        # Initialize the generator with the selected LLM provider
        llm_provider_enum = getattr(LLMProvider, args.llm.upper())
        config = get_default_config(llm_provider_enum)
        llm_instance = create_llm_instance(config.llm)
        file = args.file

        workflow(llm_instance,file)

    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("   • Check your API key is set correctly")
        print("   • Verify internet connection")
        print("   • Try with a different LLM provider")
        print("   • Check the file syntax is valid")
        sys.exit(1)


if __name__ == "__main__":
    main()
