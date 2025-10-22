#!/usr/bin/env python3
"""
CLI Runner for LLM-Optimized CrewAI Unit Test Generation System
"""

import logging
import sys
import argparse
from pathlib import Path
from datetime import datetime

from config import LLMProvider, get_default_config, create_llm_instance
from helpers.analysers.code_analyzer import CodeAnalyzer
from agents.unittest_agent.unittest_agent import UnitTestAgent

from workflow import workflow

from pathlib import Path


def get_tree(
    path: str,
    prefix: str = "",
    exclude: list[str] = [
        ".env",
        ".venv",
        "__pycache__",
        ".git",
        ".idea",
        ".pytest_cache",
        ".ipynb_checkpoints",
        "node_modules",
        "venv",
        "env",
        ".DS_Store",
        "logs"
    ],
) -> str:
    """
    Return a folder structure like the Linux 'tree' command as a string.

    Args:
        path (str): Directory path to visualize.
        prefix (str): Internal use for formatting recursion.
        exclude (list[str]): List of folder or file names to exclude (e.g., [".env", ".venv", "__pycache__"])
    """
    p = Path(path)
    if not p.exists():
        return f"❌ Path not found: {path}\n"

    tree_str = ""

    # For the root call, show only the folder name
    if prefix == "":
        tree_str += f"{p.name}/\n"

    # Filter out excluded items and non-.py files
    contents = sorted(
        [
            c
            for c in p.iterdir()
            if (c.is_dir() or c.suffix == ".py") and c.name not in exclude
        ],
        key=lambda x: (x.is_file(), x.name.lower()),
    )

    pointers = ["├── "] * (len(contents) - 1) + ["└── "] if contents else []

    for pointer, child in zip(pointers, contents):
        line = prefix + pointer + (child.name + "/" if child.is_dir() else child.name)
        tree_str += line + "\n"

        if child.is_dir():
            extension = "│   " if pointer == "├── " else "    "
            tree_str += get_tree(child, prefix + extension, exclude)

    return tree_str


# def main():
#     """Main CLI function"""
#     parser = argparse.ArgumentParser(
#         description="LLM-Optimized CrewAI Unit Test Generation System",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   python cli_runner.py                              # Use default sample file
#   python cli_runner.py sample_code/calculator.py   # Test specific file
#   python cli_runner.py --llm openai sample.py      # Use OpenAI
#   python cli_runner.py --help                      # Show this help

# Supported LLM providers: gemini, openai, claude, ollama
#         """,
#     )

#     parser.add_argument(
#         "file",
#         nargs="?",
#         default="sample_code/calculator.py",
#         help="Python file to generate tests for (default: sample_code/calculator.py)",
#     )

#     parser.add_argument(
#         "--llm",
#         choices=["gemini", "openai", "claude", "ollama"],
#         default="gemini",
#         help="LLM provider to use (default: gemini)",
#     )

#     parser.add_argument(
#         "--output",
#         default="sample_code/generated_tests",
#         help="Output directory for generated tests (default: sample_code/generated_tests)",
#     )

#     parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

#     args = parser.parse_args()

#     # Validate input file
#     if not Path(args.file).exists():
#         print(f"❌ Error: File '{args.file}' not found")
#         print("\nTip: Make sure the file exists or try the default:")
#         print("python cli_runner.py")
#         sys.exit(1)

#     # Header
#     print("🚀 LLM-Optimized AI Unit Test Generation System")
#     print("=" * 60)
#     print(f"📄 Input file: {args.file}")
#     print(f"🤖 LLM provider: {args.llm}")
#     print(f"📁 Output directory: {args.output}")
#     print()

#     try:
#         # Initialize the generator with the selected LLM provider
#         llm_provider_enum = getattr(LLMProvider, args.llm.upper())
#         config = get_default_config(llm_provider_enum)
#         llm_instance = create_llm_instance(config.llm)
#         file = args.file

#         workflow(llm_instance,file)

#     except Exception as e:
#         print(f"\n❌ Unexpected error: {str(e)}")
#         print("\n🔧 Troubleshooting tips:")
#         print("   • Check your API key is set correctly")
#         print("   • Verify internet connection")
#         print("   • Try with a different LLM provider")
#         print("   • Check the file syntax is valid")
#         sys.exit(1)


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="LLM-Optimized CrewAI Unit Test Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_runner.py                              # Use default sample file
  python cli_runner.py sample_code/calculator.py   # Test specific file
  python cli_runner.py sample_code/                # Process all files in folder
  python cli_runner.py --llm openai sample.py      # Use OpenAI
  python cli_runner.py --help                      # Show this help

Supported LLM providers: gemini, openai, claude, ollama
        """,
    )

    parser.add_argument(
        "path",
        nargs="?",
        default="sample_code/calculator.py",
        help="Python file or folder to generate tests for (default: sample_code/simple_cal)",
    )

    parser.add_argument(
        "--llm",
        choices=["gemini", "openai", "claude", "ollama"],
        default="gemini",
        help="LLM provider to use (default: gemini)",
    )

    parser.add_argument(
        "--output",
        default="sample_code/generated_tests",
        help="Output directory for generated tests (default: sample_code/simple_cal/generated_tests)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    input_path = Path(args.path)

    # Validate input path
    if not input_path.exists():
        print(f"❌ Error: Path '{args.path}' not found")
        sys.exit(1)

    # Header
    print("🚀 LLM-Optimized AI Unit Test Generation System")
    print("=" * 60)
    print(f"📄 Input path: {args.path}")
    print(f"🤖 LLM provider: {args.llm}")
    print(f"📁 Output directory: {args.output}")
    print()

    try:
        # Initialize LLM provider
        llm_provider_enum = getattr(LLMProvider, args.llm.upper())
        config = get_default_config(llm_provider_enum, output_directory=args.output)
        llm_instance = create_llm_instance(config.llm)

        # Collect files
        files_to_process = []
        if input_path.is_file():
            files_to_process = [input_path]
        elif input_path.is_dir():
            files_to_process = list(input_path.rglob("*.py"))

        if not files_to_process:
            print("⚠️ No Python files found in the specified folder.")
            sys.exit(0)

        # Determine base path for relative calculations
        base_path = input_path.parent if input_path.is_file() else input_path
        token = 0
        import os

        current_dir = os.getcwd()
        print(current_dir)
        folder_structure = get_tree(current_dir)
        for file in files_to_process:
            if file.name.startswith("test_"):
                continue  # Skip already generated test files
            print(f"🧩 Processing: {file}")
            relative_base_path = file.parent.relative_to(base_path)
            file_path= None
            if relative_base_path == Path("."):
                relative_base_path = Path("")
                file_path = Path(config.output_directory) 
            else:
                file_path = Path(config.output_directory) / f"test_{file.parent.relative_to(base_path)}"
            file_path.mkdir(parents=True, exist_ok=True)
            save_path = file_path / f"test_{file.name}"
            print("creating path:", save_path)

            token += workflow(llm_instance, str(file), save_path, folder_structure)
            print("✅ Done\n")
        print("🎉 All tasks completed!"
              f"\nTotal tokens used: {token}")

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
