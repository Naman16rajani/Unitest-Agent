import os
from datetime import datetime
from helpers.merge_test_code import merge_test_code


def save_code(file_path, code):
    """
    Intelligently merge and save code to a file.
    Creates the file and directories if they don't exist.
    Avoids duplicate imports and fixtures when appending to existing test files.

    Args:
        file_path (str): Path to the file where code should be saved
        code (str): Code content to merge with existing file
    """
    # Create directories if they don't exist
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Read existing content if file exists
    existing_code = ""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            existing_code = file.read()

    # Merge new code with existing code (deduplicates imports and fixtures)
    merged_code = merge_test_code(existing_code, code)

    # Write the merged code (overwrite mode)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(merged_code)
