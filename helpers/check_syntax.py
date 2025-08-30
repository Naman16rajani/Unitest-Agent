import ast


def check_syntax(code: str) -> bool:
    """
    Checks if the syntax of the provided Python code string is valid.

    Args:
        code (str): The Python code to check.

    Returns:
        bool: True if the syntax is valid, False otherwise.
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
