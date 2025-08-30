def clean_code(test_code):
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