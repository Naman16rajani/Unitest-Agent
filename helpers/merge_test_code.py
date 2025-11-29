import re
import ast
from typing import Set, List


def extract_imports(code: str) -> tuple[Set[str], Set[str]]:
    """
    Extract import statements and from-import statements from code.

    Args:
        code: Python code as string

    Returns:
        Tuple of (regular_imports, from_imports) as sets of strings
    """
    regular_imports = set()
    from_imports = set()

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_line = f"import {alias.name}"
                    if alias.asname:
                        import_line += f" as {alias.asname}"
                    regular_imports.add(import_line)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    from_line = f"from {module} import {alias.name}"
                    if alias.asname:
                        from_line += f" as {alias.asname}"
                    from_imports.add(from_line)
    except SyntaxError:
        # Fallback to regex if AST parsing fails
        import_pattern = r"^import\s+[\w.,\s]+(?:\s+as\s+\w+)?$"
        from_pattern = r"^from\s+[\w.]+\s+import\s+[\w.,\s]+(?:\s+as\s+\w+)?$"

        for line in code.split("\n"):
            line = line.strip()
            if re.match(import_pattern, line):
                regular_imports.add(line)
            elif re.match(from_pattern, line):
                from_imports.add(line)

    return regular_imports, from_imports


def extract_fixtures(code: str) -> List[str]:
    """
    Extract pytest fixture function definitions from code, including decorators.

    Args:
        code: Python code as string

    Returns:
        List of fixture definitions as strings (with decorators)
    """
    fixtures = []

    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                # Check if function has @pytest.fixture decorator
                has_fixture_decorator = any(
                    (isinstance(dec, ast.Name) and dec.id == "fixture")
                    or (isinstance(dec, ast.Attribute) and dec.attr == "fixture")
                    for dec in node.decorator_list
                )

                if has_fixture_decorator:
                    # Find the starting line (including decorators)
                    start_line = node.lineno
                    if node.decorator_list:
                        start_line = node.decorator_list[0].lineno

                    # Find the ending line
                    end_line = node.end_lineno

                    # Extract the code including decorators
                    lines = code.split("\n")
                    # Adjust for 0-based indexing
                    fixture_code = "\n".join(lines[start_line - 1 : end_line])

                    if fixture_code:
                        fixtures.append(fixture_code)
    except SyntaxError:
        # Fallback to regex-based extraction
        fixture_pattern = r'@pytest\.fixture.*?\ndef\s+\w+.*?:\s*(?:""".*?"""|\'\'\'.*?\'\'\')?.*?(?=\n(?:@|def|class|\Z))'
        fixtures = re.findall(fixture_pattern, code, re.DOTALL | re.MULTILINE)

    return fixtures


def extract_test_functions(code: str) -> List[str]:
    """
    Extract test function definitions (excluding fixtures) from code,
    including their decorators.

    Args:
        code: Python code as string

    Returns:
        List of test function definitions as strings (with decorators)
    """
    test_functions = []

    try:
        tree = ast.parse(code)
        # Get all top-level nodes
        for i, node in enumerate(tree.body):
            if isinstance(node, ast.FunctionDef):
                # Check if it's a test function (not a fixture)
                is_fixture = any(
                    (isinstance(dec, ast.Name) and dec.id == "fixture")
                    or (isinstance(dec, ast.Attribute) and dec.attr == "fixture")
                    for dec in node.decorator_list
                )

                if not is_fixture and (
                    node.name.startswith("test_")
                    or any(
                        isinstance(dec, ast.Call)
                        and isinstance(dec.func, ast.Attribute)
                        and dec.func.attr == "parametrize"
                        for dec in node.decorator_list
                    )
                ):
                    # Find the starting line (including decorators)
                    start_line = node.lineno
                    if node.decorator_list:
                        start_line = node.decorator_list[0].lineno

                    # Find the ending line
                    end_line = node.end_lineno

                    # Extract the code including decorators
                    lines = code.split("\n")
                    # Adjust for 0-based indexing
                    test_code = "\n".join(lines[start_line - 1 : end_line])

                    if test_code:
                        test_functions.append(test_code)
    except SyntaxError:
        pass

    return test_functions


def extract_constants(code: str) -> List[str]:
    """
    Extract module-level constants (e.g., VALIDATE_NUMBERS_PATH).

    Args:
        code: Python code as string

    Returns:
        List of constant definitions as strings
    """
    constants = []

    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                # Check if it's a constant (uppercase variable name)
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constant_code = ast.get_source_segment(code, node)
                        if constant_code:
                            constants.append(constant_code)
    except SyntaxError:
        pass

    return constants


def remove_timestamp_comments(code: str) -> str:
    """Remove timestamp comments like '# Code added at 20251124-095718'"""
    return re.sub(r"^# Code added at \d{8}-\d{6}\s*\n", "", code, flags=re.MULTILINE)


def merge_test_code(existing_code: str, new_code: str) -> str:
    """
    Intelligently merge new test code with existing test code,
    avoiding duplicate imports and fixtures.

    Args:
        existing_code: Existing test file content
        new_code: New test code to add

    Returns:
        Merged code with deduplicated imports and fixtures
    """
    # Remove timestamp comments
    existing_code = remove_timestamp_comments(existing_code)
    new_code = remove_timestamp_comments(new_code)

    # Extract components from existing code
    existing_regular_imports, existing_from_imports = extract_imports(existing_code)
    existing_fixtures = set(extract_fixtures(existing_code))
    existing_constants = set(extract_constants(existing_code))

    # Extract components from new code
    new_regular_imports, new_from_imports = extract_imports(new_code)
    new_fixtures = extract_fixtures(new_code)
    new_test_functions = extract_test_functions(new_code)
    new_constants = extract_constants(new_code)

    # Merge imports (combine sets to avoid duplicates)
    all_regular_imports = sorted(existing_regular_imports | new_regular_imports)
    all_from_imports = sorted(existing_from_imports | new_from_imports)

    # Merge constants (avoid duplicates)
    all_constants = sorted(existing_constants | set(new_constants))

    # Merge fixtures (avoid duplicates based on normalized content)
    all_fixtures_dict = {}
    for fixture in existing_fixtures:
        # Use fixture function name as key
        match = re.search(r"def\s+(\w+)\s*\(", fixture)
        if match:
            all_fixtures_dict[match.group(1)] = fixture

    for fixture in new_fixtures:
        match = re.search(r"def\s+(\w+)\s*\(", fixture)
        if match:
            fixture_name = match.group(1)
            # Only add if we don't already have this fixture
            if fixture_name not in all_fixtures_dict:
                all_fixtures_dict[fixture_name] = fixture

    # Build the merged file
    sections = []

    # 1. Regular imports
    if all_regular_imports:
        sections.append("\n".join(all_regular_imports))

    # 2. From imports
    if all_from_imports:
        sections.append("\n".join(all_from_imports))

    # 3. Constants
    if all_constants:
        sections.append("\n".join(all_constants))

    # 4. Fixtures
    if all_fixtures_dict:
        sections.append("\n\n".join(all_fixtures_dict.values()))

    # 5. Extract existing test functions
    existing_test_functions = extract_test_functions(existing_code)

    # 6. Add all test functions (existing + new)
    all_test_functions = existing_test_functions + new_test_functions
    if all_test_functions:
        sections.append("\n\n".join(all_test_functions))

    # Join all sections with double newlines
    merged_code = "\n\n".join(sections)

    # Add final newline
    if not merged_code.endswith("\n"):
        merged_code += "\n"

    return merged_code
