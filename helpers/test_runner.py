"""
Helper module to run pytest with coverage reporting
"""

import subprocess
import sys
from pathlib import Path
from typing import Tuple, Optional
from helpers.logging_config import create_logger

logger = create_logger("TestRunner")


def run_pytest_with_coverage(
    test_file_path: Path, source_file_path: Path, htmlcov_dir: str = "htmlcov"
) -> Tuple[bool, str, Optional[str]]:
    """
    Run pytest with coverage on a specific test file.

    Args:
        test_file_path: Path to the test file to run
        source_file_path: Path to the source file to measure coverage for
        htmlcov_dir: Directory to store HTML coverage reports (default: "htmlcov")

    Returns:
        Tuple of (success, output, coverage_percentage)
        - success: True if tests passed, False otherwise
        - output: Terminal output from pytest
        - coverage_percentage: Coverage percentage as string (e.g., "95%") or None if failed
    """
    logger.info(f"Running pytest with coverage for {test_file_path}")

    # Ensure paths are absolute
    test_file_path = Path(test_file_path).absolute()
    source_file_path = Path(source_file_path).absolute()

    # Build pytest command with coverage
    # --cov: measure coverage for the source file
    # --cov-report=term: show coverage in terminal
    # --cov-report=html: generate HTML coverage report
    # -v: verbose output
    cmd = [
        sys.executable,  # Use the same Python interpreter
        "-m",
        "pytest",
        str(test_file_path),
        f"--cov={source_file_path.parent}",
        f"--cov-report=term",
        f"--cov-report=html:{htmlcov_dir}",
        "-v",
        "--tb=short",  # Short traceback format
    ]

    logger.debug(f"Running command: {' '.join(cmd)}")

    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

        output = result.stdout + result.stderr
        success = result.returncode == 0

        # Extract coverage percentage from output
        coverage_percentage = extract_coverage_percentage(output)

        logger.info(f"Pytest completed with return code: {result.returncode}")
        if coverage_percentage:
            logger.info(f"Coverage: {coverage_percentage}")

        return success, output, coverage_percentage

    except FileNotFoundError:
        error_msg = "❌ pytest not found. Please install pytest and pytest-cov:\n   pip install pytest pytest-cov"
        logger.error(error_msg)
        return False, error_msg, None
    except Exception as e:
        error_msg = f"❌ Error running pytest: {str(e)}"
        logger.error(error_msg)
        return False, error_msg, None


def extract_coverage_percentage(output: str) -> Optional[str]:
    """
    Extract coverage percentage from pytest-cov output.

    Args:
        output: pytest output string

    Returns:
        Coverage percentage as string (e.g., "95%") or None if not found
    """
    import re

    # Look for coverage percentage in output
    # Format: "TOTAL    123    45    63%"
    match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+%)", output)
    if match:
        return match.group(1)

    # Alternative format
    match = re.search(r"Coverage:\s*(\d+%)", output)
    if match:
        return match.group(1)

    return None


def display_coverage_summary(output: str):
    """
    Display a formatted coverage summary in the terminal.

    Args:
        output: pytest output containing coverage information
    """
    print("\n" + "=" * 60)
    print("📊 COVERAGE REPORT")
    print("=" * 60)

    # Extract and display the coverage table
    lines = output.split("\n")
    in_coverage_section = False

    for line in lines:
        # Start of coverage section
        if "Name" in line and "Stmts" in line and "Miss" in line:
            in_coverage_section = True
            print(line)
            continue

        # End of coverage section
        if in_coverage_section and line.strip().startswith("---"):
            print(line)
            continue

        if in_coverage_section:
            if line.strip() and not line.strip().startswith("="):
                print(line)
            elif line.strip().startswith("="):
                print(line)
                break

    print("=" * 60)

    # Extract coverage percentage
    coverage = extract_coverage_percentage(output)
    if coverage:
        coverage_num = int(coverage.rstrip("%"))

        # Color-code based on coverage
        if coverage_num >= 90:
            emoji = "🎉"
            status = "Excellent"
        elif coverage_num >= 75:
            emoji = "✅"
            status = "Good"
        elif coverage_num >= 50:
            emoji = "⚠️"
            status = "Fair"
        else:
            emoji = "❌"
            status = "Poor"

        print(f"\n{emoji} Overall Coverage: {coverage} - {status}")

    print("=" * 60 + "\n")


def display_test_results(output: str, success: bool):
    """
    Display formatted test results.

    Args:
        output: pytest output
        success: whether tests passed
    """
    print("\n" + "=" * 60)
    print("🧪 TEST RESULTS")
    print("=" * 60)

    # Extract test summary line
    lines = output.split("\n")
    for line in lines:
        if "passed" in line.lower() or "failed" in line.lower():
            if "==" in line:  # This is the summary line
                print(line)

    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        # Show failed test details
        in_failed_section = False
        for line in lines:
            if "FAILED" in line:
                in_failed_section = True
            if in_failed_section:
                print(line)
                if line.strip() == "":
                    in_failed_section = False

    print("=" * 60 + "\n")
