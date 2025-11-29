from agents.mock_class_agent.mock_class_agent import MockClassAgent
from helpers.analysers.class_formatter_analyzer import ClassFormatterAnalyzer
from helpers.logging_config import logger, create_logger
import json
from agents.unittest_agent.unittest_agent import UnitTestAgent
from helpers.analysers.method_formatter_analyzer import MethodFormatterAnalyzer
from helpers.check_syntax import check_syntax
from helpers.clean_code import clean_code
from helpers.read_file import read_file
from helpers.save_code import save_code
from helpers.test_runner import (
    run_pytest_with_coverage,
    display_coverage_summary,
    display_test_results,
)
from pathlib import Path

logger = create_logger("Workflow")
MAX_AGENT_FLOW_COUNT = 3


from agents.existing_test_analyzer_agent.existing_test_analyzer_agent import (
    ExistingTestAnalyzerAgent,
)


def workflow(llm_instance, file, save_path, folder_structure) -> int:
    try:
        if check_syntax(read_file(file)):
            method_analyzer = MethodFormatterAnalyzer()

            analysis_data = method_analyzer.format_analysis(file)
            # logger.debug(
            #     "Method Analysis Data:\n" + json.dumps(analysis_data, indent=4)
            # )
            token = 0

            class_analyzer = ClassFormatterAnalyzer()
            class_analysis_data = class_analyzer.format_analysis(file)
            # logger.debug(
            #     "Class Analysis Data:\n" + json.dumps(class_analysis_data, indent=4)
            # )

            unit_test_agent = UnitTestAgent(llm_instance)
            mock_class_agent = MockClassAgent(llm_instance)
            existing_test_analyzer_agent = ExistingTestAnalyzerAgent(llm_instance)
            print("Generating Mock Classes...")
            for classes in class_analysis_data:
                print("Generating mock class for: " + classes["class_name"])
                mock_code, token_inc = mock_class_agent.invoke(
                    user_prompt=classes,
                    input_folder_path=file,
                    output_folder_path=save_path,
                    folder_structure=folder_structure,
                )
                token += token_inc
                logger.debug("Generated Mock Code for: " + classes["class_name"])
                mock_code = clean_code(mock_code)

                save_code(save_path, mock_code)

            print("Generating Unit Tests...")
            for methods in analysis_data:
                content = ""
                existing_test_analysis = None
                print("Generating unit tests for: " + methods["method_name"])
                if save_path.exists():
                    content = read_file(save_path)
                    logger.debug("Content Data:\n" + content)

                    # Analyze existing tests
                    print("Analyzing existing tests...")
                    existing_test_analysis, token_inc = (
                        existing_test_analyzer_agent.invoke(content)
                    )
                    token += token_inc
                    logger.debug(
                        "Existing Test Analysis:\n" + str(existing_test_analysis)
                    )

                else:
                    content = ""

                code, token_inc = unit_test_agent.invoke(
                    methods,
                    code=content,
                    folder_structure=folder_structure,
                    existing_test_analysis=existing_test_analysis,
                )
                code = clean_code(code)
                logger.debug("Generated Code:\n" + code)
                i = 0
                while not check_syntax(code) and i < MAX_AGENT_FLOW_COUNT:
                    print("Regenerating unittest, attempt " + str(i + 1))
                    code, token_inc = unit_test_agent.invoke(
                        methods,
                        code=read_file(save_path),
                        folder_structure=folder_structure,
                        existing_test_analysis=existing_test_analysis,
                    )
                    token += token_inc
                    i += 1

                if not check_syntax(code):
                    return "not able to generate unittest"
                token += token_inc

                save_code(save_path, code)

            print("✅ Unit tests generated successfully!")
            print(f"📊 Tokens used: {token}\n")

            # Run pytest with coverage
            print("=" * 60)
            print("🧪 Running Tests with Coverage...")
            print("=" * 60)

            source_file = Path(file)
            test_file = Path(save_path)

            success, output, coverage_pct = run_pytest_with_coverage(
                test_file_path=test_file,
                source_file_path=source_file,
                htmlcov_dir="htmlcov",
            )

            # Display test results
            display_test_results(output, success)

            # Display coverage summary
            display_coverage_summary(output)

            # Show path to HTML report
            htmlcov_path = Path("htmlcov") / "index.html"
            if htmlcov_path.exists():
                print(f"📄 HTML Coverage Report: {htmlcov_path.absolute()}")
                print(f"   Open in browser: file://{htmlcov_path.absolute()}\n")

            return token

        else:
            return 0
    except Exception as e:
        print(e)
        return 0
