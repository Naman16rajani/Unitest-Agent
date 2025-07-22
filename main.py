#!/usr/bin/env python3
"""
CLI Runner for LLM-Optimized CrewAI Unit Test Generation System
"""

from calendar import c
import code
from pprint import pprint
import sys
import argparse
from pathlib import Path
import os
from datetime import datetime

from config import LLMProvider, SystemConfig, get_default_config, create_llm_instance
from helpers.code_analyser import CodeAnalyzer
from helpers.prompt_writter import PromptWriter
from agents.unit_test_agent import UnitTestAgent


def save_single_test_to_file(
    test_code,
    item_name,
    item_type,
    source_file_path,
    output_directory,
    method_name=None,
):
    """Save a single test to file immediately after generation"""

    # Clean up the test code by removing markdown code block markers
    cleaned_test_code = test_code.strip()

    # Remove ```python from the beginning
    if cleaned_test_code.startswith("```python"):
        cleaned_test_code = cleaned_test_code[9:].strip()
    elif cleaned_test_code.startswith("```"):
        cleaned_test_code = cleaned_test_code[3:].strip()

    # Remove ``` from the end
    if cleaned_test_code.endswith("```"):
        cleaned_test_code = cleaned_test_code[:-3].strip()

    # Create output directory if it doesn't exist
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get the base name of the source file
    source_name = Path(source_file_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if item_type == "class_method":
        # For class methods, create or append to class file
        class_file_name = f"test_{source_name}_{item_name.lower()}.py"
        class_file_path = output_path / class_file_name

        # Check if file exists to determine if we need to add header
        file_exists = class_file_path.exists()

        with open(class_file_path, "a", encoding="utf-8") as f:
            if not file_exists:
                # Add header for new file
                f.write(f"# Generated tests for class: {item_name}\n")
                f.write(f"# Source file: {source_file_path}\n")
                f.write(
                    f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write("\n")

            # Add method test
            f.write(f"# Tests for method: {method_name}\n")
            f.write(cleaned_test_code)
            f.write("\n\n")

        return str(class_file_path)

    else:
        # For standalone functions
        function_file_name = f"test_{source_name}_{item_name.lower()}_{timestamp}.py"
        function_file_path = output_path / function_file_name

        with open(function_file_path, "w", encoding="utf-8") as f:
            f.write(f"# Generated tests for function: {item_name}\n")
            f.write(f"# Source file: {source_file_path}\n")
            f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")
            f.write(cleaned_test_code)

        return str(function_file_path)


def save_tests_to_files(all_generated_tests, source_file_path, output_directory):
    """Save generated tests to files organized by class/function"""

    def clean_test_code(test_code):
        """Clean up test code by removing markdown markers"""
        cleaned = test_code.strip()

        # Remove ```python from the beginning
        if cleaned.startswith("```python"):
            cleaned = cleaned[9:].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:].strip()

        # Remove ``` from the end
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        return cleaned

    # Create output directory if it doesn't exist
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get the base name of the source file
    source_name = Path(source_file_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    saved_files = []

    for item_name, tests in all_generated_tests.items():
        if isinstance(tests, dict):
            # Handle class tests (multiple methods)
            class_file_name = f"test_{source_name}_{item_name.lower()}_{timestamp}.py"
            class_file_path = output_path / class_file_name

            # Combine all method tests for this class
            combined_test_code = []
            combined_test_code.append(f"# Generated tests for class: {item_name}")
            combined_test_code.append(f"# Source file: {source_file_path}")
            combined_test_code.append(
                f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            combined_test_code.append("")

            for method_name, test_code in tests.items():
                combined_test_code.append(f"# Tests for method: {method_name}")
                combined_test_code.append(clean_test_code(test_code))
                combined_test_code.append("")

            # Write to file
            with open(class_file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(combined_test_code))

            saved_files.append(
                {
                    "type": "class",
                    "name": item_name,
                    "file": str(class_file_path),
                    "methods_count": len(tests),
                }
            )

        else:
            # Handle function tests (single function)
            function_file_name = (
                f"test_{source_name}_{item_name.lower()}_{timestamp}.py"
            )
            function_file_path = output_path / function_file_name

            # Create test file content
            test_content = []
            test_content.append(f"# Generated tests for function: {item_name}")
            test_content.append(f"# Source file: {source_file_path}")
            test_content.append(
                f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            test_content.append("")
            test_content.append(clean_test_code(tests))

            # Write to file
            with open(function_file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(test_content))

            saved_files.append(
                {
                    "type": "function",
                    "name": item_name,
                    "file": str(function_file_path),
                    "methods_count": 1,
                }
            )

    return saved_files


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

        code_analyzer = CodeAnalyzer()
        # pprint(code_analyzer.analyze_file(args.file))
        analysis_data = code_analyzer.analyze_file(args.file)

        prompt_writer = PromptWriter()

        universal_prompt = "write me pytest unitttest for the above information and use magicmock to mock the functions and classes. if i had given constructor mock the class using that constructor and do the pytest unittest. Import the real class just mock the constructor methods that can be mocked. Aim is to cover 100% code. Follow type safety if type is provided. Just give me unittest code no explanation"
        updated_analysis_universal = prompt_writer.convert_to_function_list_format(
            analysis_data, universal_prompt
        )

        # Initialize the UnitTestAgent
        unit_test_agent = UnitTestAgent(llm_instance)

        print("🧪 Generating unit tests for each function/method...")
        print()

        # Iterate through the analysis data and generate tests
        all_generated_tests = {}
        saved_files_tracking = {}  # Track saved files to avoid duplicates

        for item_name, item_data in updated_analysis_universal.items():
            print(f"📝 Processing: {item_name}")

            if isinstance(item_data, list):
                # Handle class methods (list of method dictionaries)
                print(f"   ├── Class with {len(item_data)} methods")
                class_tests = {}

                for method_dict in item_data:
                    # Each method_dict is like {'method_name': {method_data}}
                    for method_name, method_data in method_dict.items():
                        print(f"   │   ├── Generating test for method: {method_name}")

                        # Create the format expected by unit_test_agent
                        method_input = {method_name: method_data}

                        try:
                            # Generate the test code
                            test_code = unit_test_agent.generate_unit_test(method_input)
                            class_tests[method_name] = test_code

                            # Save immediately after generation
                            saved_file_path = save_single_test_to_file(
                                test_code,
                                item_name,
                                "class_method",
                                args.file,
                                args.output,
                                method_name,
                            )

                            # Track the saved file
                            if item_name not in saved_files_tracking:
                                saved_files_tracking[item_name] = {
                                    "type": "class",
                                    "file": saved_file_path,
                                    "methods": [],
                                }
                            saved_files_tracking[item_name]["methods"].append(
                                method_name
                            )

                            print(
                                f"   │   │   ✅ Generated and saved to: {saved_file_path}"
                            )
                        except Exception as e:
                            print(f"   │   │   ❌ Failed: {str(e)}")
                            error_comment = (
                                f"# Error generating test for {method_name}: {str(e)}"
                            )
                            class_tests[method_name] = error_comment

                            # Save error comment too
                            try:
                                save_single_test_to_file(
                                    error_comment,
                                    item_name,
                                    "class_method",
                                    args.file,
                                    args.output,
                                    method_name,
                                )
                            except Exception:
                                pass  # Don't fail if we can't save error

                all_generated_tests[item_name] = class_tests

            else:
                # Handle standalone functions (direct dictionary)
                print("   ├── Standalone function")

                # Create the format expected by unit_test_agent
                function_input = {item_name: item_data}

                try:
                    # Generate the test code
                    test_code = unit_test_agent.generate_unit_test(function_input)
                    all_generated_tests[item_name] = test_code

                    # Save immediately after generation
                    saved_file_path = save_single_test_to_file(
                        test_code, item_name, "function", args.file, args.output
                    )

                    # Track the saved file
                    saved_files_tracking[item_name] = {
                        "type": "function",
                        "file": saved_file_path,
                        "methods": 1,
                    }

                    print(f"   │   ✅ Generated and saved to: {saved_file_path}")
                except Exception as e:
                    print(f"   │   ❌ Failed: {str(e)}")
                    error_comment = f"# Error generating test for {item_name}: {str(e)}"
                    all_generated_tests[item_name] = error_comment

                    # Save error comment too
                    try:
                        saved_file_path = save_single_test_to_file(
                            error_comment, item_name, "function", args.file, args.output
                        )
                        saved_files_tracking[item_name] = {
                            "type": "function",
                            "file": saved_file_path,
                            "methods": 1,
                        }
                    except Exception:
                        pass  # Don't fail if we can't save error

        print()
        print("📊 Generation Summary:")
        for item_name, tests in all_generated_tests.items():
            if isinstance(tests, dict):
                print(f"   📁 {item_name}: {len(tests)} method tests")
                for method_name in tests.keys():
                    print(f"      └── {method_name}")
            else:
                print(f"   📄 {item_name}: function test")

        print()
        print("🎯 Sample generated test (first item):")
        if all_generated_tests:
            first_item = list(all_generated_tests.values())[0]
            if isinstance(first_item, dict):
                # Show first method test of first class
                first_method_test = list(first_item.values())[0]
                print("=" * 60)
                print(
                    first_method_test[:500] + "..."
                    if len(first_method_test) > 500
                    else first_method_test
                )
                print("=" * 60)
            else:
                # Show first function test
                print("=" * 60)
                print(first_item[:500] + "..." if len(first_item) > 500 else first_item)
                print("=" * 60)

        # Display saved files summary
        print()
        print("✅ Test files have been saved during generation!")
        print()
        print("📁 Generated test files:")
        for item_name, file_info in saved_files_tracking.items():
            if file_info["type"] == "class":
                print(f"   📋 {item_name} (class): {file_info['file']}")
                print(
                    f"      └── {len(file_info['methods'])} methods: {', '.join(file_info['methods'])}"
                )
            else:
                print(f"   📄 {item_name} (function): {file_info['file']}")

        print()
        print("🚀 Next steps:")
        print("   • Review the generated test files")
        print("   • Install required packages: pytest, unittest.mock")
        print("   • Run tests with: pytest " + args.output)
        print("   • Modify tests as needed for your specific requirements")

        # generator = UnitTestGenerator(llm_provider)

        # # Generate tests
        # result = generator.generate_tests(args.file)

        # if result.get("success"):
        #     print("\n🎉 Test generation completed successfully!")
        #     print("📁 Generated files:")
        #     print(f"   • Test file: {result['test_file']}")
        #     print(f"   • Documentation: {result['documentation_file']}")
        #     print()
        #     print("💡 Next steps:")
        #     print("   • Review the generated test file")
        #     print(f"   • Run the tests: pytest {result['test_file']}")
        #     print("   • Check the documentation for details")
        # else:
        #     print(
        #         f"\n❌ Test generation failed: {result.get('error', 'Unknown error')}"
        #     )
        #     sys.exit(1)

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
