import re


def clean_code(text):
    """Clean up test code by removing markdown markers and extracting code blocks"""
    # Pattern to find code blocks: ```python ... ``` or ``` ... ```
    # We use non-greedy matching .*? to get the first block
    pattern = r"```(?:python)?\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback for existing logic if no matching pair found (e.g. only start or only end?)
    # Or if the code is raw without backticks
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove first line if it is ``` or ```python
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            cleaned = "\n".join(lines[1:])

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    return cleaned
