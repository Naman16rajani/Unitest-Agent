#!/usr/bin/env python3
"""
Script to clean up existing test files with duplicate imports and fixtures.
"""
import sys
from pathlib import Path
from helpers.merge_test_code import merge_test_code


def clean_test_file(file_path: Path):
    """Clean a single test file by merging it with itself to remove duplicates."""
    print(f"Cleaning {file_path}...")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Merge with empty string to deduplicate
    cleaned_content = merge_test_code("", content)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(cleaned_content)

    print(f"✅ Cleaned {file_path}")


def main():
    # Find all test files in generated_unittest directory
    test_dir = Path("generated_unittest")

    if not test_dir.exists():
        print(f"Directory {test_dir} not found!")
        return

    test_files = list(test_dir.rglob("test_*.py"))

    if not test_files:
        print("No test files found!")
        return

    print(f"Found {len(test_files)} test file(s) to clean:")
    for f in test_files:
        print(f"  - {f}")

    print()

    for test_file in test_files:
        clean_test_file(test_file)

    print(f"\n🎉 Successfully cleaned {len(test_files)} test file(s)!")


if __name__ == "__main__":
    main()
