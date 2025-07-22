#!/usr/bin/env python3
"""
Test script to verify the file saving functionality
"""

from pathlib import Path
import tempfile
import os
from main import save_single_test_to_file


def test_save_functionality():
    """Test the save functionality with sample data"""

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Sample test code with markdown markers (simulating LLM output)
        sample_test_code_with_markers = '''```python
import pytest
from unittest.mock import MagicMock, patch

def test_sample_function():
    """Test for sample function"""
    assert True
    
def test_sample_function_with_mock():
    """Test with mock"""
    with patch('module.function') as mock_func:
        mock_func.return_value = "mocked"
        result = mock_func()
        assert result == "mocked"
```'''

        # Sample test code without markers
        sample_test_code_clean = '''import pytest
from unittest.mock import MagicMock, patch

def test_another_function():
    """Test for another function"""
    assert True'''

        # Test saving a function test with markdown markers
        print("Testing function test save with markdown markers...")
        func_file = save_single_test_to_file(
            sample_test_code_with_markers,
            "sample_function",
            "function",
            "sample_code/test.py",
            temp_dir,
        )
        print(f"Function test saved to: {func_file}")

        # Test saving class method tests
        print("Testing class method test save...")
        class_file = save_single_test_to_file(
            sample_test_code_clean,
            "SampleClass",
            "class_method",
            "sample_code/test.py",
            temp_dir,
            "method1",
        )
        print(f"Class method test saved to: {class_file}")

        # Save another method to the same class file with markers
        class_file2 = save_single_test_to_file(
            sample_test_code_with_markers,
            "SampleClass",
            "class_method",
            "sample_code/test.py",
            temp_dir,
            "method2",
        )
        print(f"Second class method test appended to: {class_file2}")

        # Verify files exist and have content
        if os.path.exists(func_file):
            with open(func_file, "r") as f:
                content = f.read()
                print(f"Function file size: {len(content)} characters")
                # Check that markdown markers were removed
                has_markers = "```python" in content or "```" in content
                print(f"Contains markdown markers: {has_markers}")

        if os.path.exists(class_file):
            with open(class_file, "r") as f:
                content = f.read()
                print(f"Class file size: {len(content)} characters")
                print(
                    "File contains both methods:",
                    "method1" in content and "method2" in content,
                )
                # Check that markdown markers were removed
                has_markers = "```python" in content or "```" in content
                print(f"Contains markdown markers: {has_markers}")

        print("✅ Save functionality test completed!")


if __name__ == "__main__":
    test_save_functionality()
