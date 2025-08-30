import os
from datetime import datetime


def save_code(file_path, code):
    """
    Append code to a file. Creates the file and directories if they don't exist.

    Args:
        file_path (str): Path to the file where code should be saved
        code (str): Code content to append to the file
    """
    # Create directories if they don't exist
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Append code to file (creates file if it doesn't exist)
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(f"# Code added at {timestamp}\n")
        file.write(code)
        # Add newline if code doesn't end with one
        if not code.endswith('\n'):
            file.write('\n')