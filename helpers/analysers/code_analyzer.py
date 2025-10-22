import ast
import os
from typing import Dict, List, Any, Optional, Union

from helpers.read_file import read_file


class CodeAnalyzer:
    """
    A comprehensive code analyzer that extracts detailed information about Python files,
    including classes, methods, functions, and their metadata.
    """

    def __init__(self):
        self.file_path = None
        self.source_code = None
        self.tree = None
        self.modules = []

    def remove_empty_values(self, obj):
        """Recursively remove keys with empty values from a dictionary or list."""
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                cleaned_value = self.remove_empty_values(v)
                if cleaned_value not in (None, "", [], {}):
                    cleaned[k] = cleaned_value
            return cleaned
        elif isinstance(obj, list):
            cleaned_list = []
            for item in obj:
                cleaned_item = self.remove_empty_values(item)
                if cleaned_item not in (None, "", [], {}, "false", False):
                    cleaned_list.append(cleaned_item)
            return cleaned_list
        else:
            return obj

    def _analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Main method to analyze a Python file and return comprehensive details.

        Args:
            file_path (str): Path to the Python file to analyze

        Returns:
            Dict[str, Any]: Detailed analysis of the file
        """
        self.file_path = file_path
        self._load_file()
        self._parse_ast()
        self._extract_modules()

        result = {
            "classes": self._analyze_classes(),
            "functions": self._analyze_functions(),
            "modules": self.modules
        }

        return result

    def _load_file(self) -> None:
        """Load the Python file content."""
        self.source_code = read_file(self.file_path)

    def _parse_ast(self) -> None:
        """Parse the source code into an AST."""
        try:
            self.tree = ast.parse(self.source_code)
        except SyntaxError as e:
            raise Exception(f"Syntax error in {self.file_path}: {str(e)}")

    def _extract_modules(self) -> None:
        """Extract all imported modules from the file."""
        self.modules = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.modules.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ""
                for alias in node.names:
                    if module:
                        self.modules.append(f"{module}.{alias.name}")
                    else:
                        self.modules.append(alias.name)

    def _analyze_classes(self) -> List[Dict[str, Any]]:
        """Analyze all classes in the file."""
        classes = []
        for node in self.tree.body:
            if isinstance(node, ast.ClassDef):
                class_info = self._analyze_single_class(node)
                classes.append(class_info)
        return classes

    def _analyze_single_class(self, class_node: ast.ClassDef) -> Dict[str, Any]:
        """Analyze a single class and extract its details."""
        class_name = class_node.name
        class_description = self._get_docstring(class_node)

        # Get source location
        source_location = {
            "file_path": self.file_path,
            "line_start": class_node.lineno,
            "line_end": self._get_end_line(class_node)
        }

        # Extract methods
        constructor = None
        class_methods = {}
        dunder_methods = []
        class_method_names = []

        for item in class_node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._analyze_method(item)

                if item.name == "__init__":
                    constructor = {item.name: method_info}
                elif item.name.startswith("__") and item.name.endswith("__"):
                    dunder_methods.append(item.name)
                else:
                    class_methods[item.name] = method_info
                    class_method_names.append(item.name)

        # Build qualified name
        module_name = os.path.splitext(os.path.basename(self.file_path))[0]
        qualified_name = f"{module_name}.{class_name}"

        return {
            "class_name": class_name,
            "class_description": class_description,
            "qualified_name": qualified_name,
            "package": self._get_package_name(),
            "source_location": source_location,
            "constructor": constructor or {},
            "dunder_methods": dunder_methods,
            "class_method_names": class_method_names,
            "class_methods": class_methods
        }

    def _analyze_functions(self) -> List[Dict[str, Any]]:
        """Analyze all standalone functions in the file."""
        functions = []
        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef):
                function_info = self._analyze_method(node)
                functions.append({node.name: function_info})
        return functions

    def _analyze_method(self, func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze a method or function and extract its details."""
        method_name = func_node.name
        method_code = self._get_method_code(func_node)
        method_description = self._get_docstring(func_node)

        # Extract type annotations
        return_type = self._get_return_type(func_node)
        parameter_types = self._get_parameter_types(func_node)

        # Extract decorators
        decorators = [self._get_decorator_name(dec) for dec in func_node.decorator_list]

        # Analyze method body
        raises = self._get_raised_exceptions(func_node)
        catches = self._get_caught_exceptions(func_node)
        is_async = isinstance(func_node, ast.AsyncFunctionDef)
        is_generator = self._is_generator(func_node)
        external_dependencies = self._get_external_dependencies(func_node)

        # Get parameters
        input_parameters = self._get_input_parameters(func_node)

        # Get module paths used in this method
        module_paths = self._get_method_module_paths(func_node)

        # Source location
        source_location = {
            "file_path": self.file_path,
            "line_start": func_node.lineno,
            "line_end": self._get_end_line(func_node)
        }

        return {
            "method_name": method_name,
            "method_code": method_code,
            "method_description": method_description,
            "return_type": return_type,
            "parameter_types": parameter_types,
            "decorators": decorators,
            "raises": raises,
            "catches": catches,
            "is_async": is_async,
            "is_generator": is_generator,
            "external_dependencies": external_dependencies,
            "input_parameters": input_parameters,
            "source_location": source_location,
            "module_path": module_paths
        }

    def _get_docstring(self, node: Union[ast.ClassDef, ast.FunctionDef]) -> str:
        """Extract docstring or top comment from a class or function."""
        if (node.body and isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            return node.body[0].value.value.strip()

        # Try to get comments from source
        lines = self.source_code.split('\n')
        start_line = node.lineno - 1

        # Look for comments right after the definition
        for i in range(start_line + 1, min(start_line + 5, len(lines))):
            line = lines[i].strip()
            if line.startswith('#'):
                return line[1:].strip()
            elif line and not line.startswith(' ') and not line.startswith('\t'):
                break

        return ""

    def _get_method_code(self, func_node: ast.FunctionDef) -> str:
        """Extract the complete code of a method."""
        lines = self.source_code.split('\n')
        start_line = func_node.lineno - 1
        end_line = self._get_end_line(func_node) - 1

        method_lines = lines[start_line:end_line + 1]
        return '\n'.join(method_lines)

    def _get_end_line(self, node: ast.AST) -> int:
        """Get the end line number of an AST node."""
        if hasattr(node, 'end_lineno') and node.end_lineno:
            return node.end_lineno

        # Fallback: find the maximum line number in the node
        max_line = node.lineno
        for child in ast.walk(node):
            if hasattr(child, 'lineno') and child.lineno:
                max_line = max(max_line, child.lineno)
        return max_line

    def _get_return_type(self, func_node: ast.FunctionDef) -> str:
        """Extract return type annotation."""
        if func_node.returns:
            return ast.unparse(func_node.returns)
        return "Any"

    def _get_parameter_types(self, func_node: ast.FunctionDef) -> Dict[str, str]:
        """Extract parameter type annotations."""
        param_types = {}
        for arg in func_node.args.args:
            if arg.annotation:
                param_types[arg.arg] = ast.unparse(arg.annotation)
            else:
                param_types[arg.arg] = "Any"
        return param_types

    def _get_decorator_name(self, decorator: ast.AST) -> str:
        """Get decorator name as string."""
        if isinstance(decorator, ast.Name):
            return f"@{decorator.id}"
        elif isinstance(decorator, ast.Attribute):
            return f"@{ast.unparse(decorator)}"
        elif isinstance(decorator, ast.Call):
            return f"@{ast.unparse(decorator.func)}"
        else:
            return f"@{ast.unparse(decorator)}"

    def _get_raised_exceptions(self, func_node: ast.FunctionDef) -> List[str]:
        """Extract exceptions that are explicitly raised."""
        raises = []
        for node in ast.walk(func_node):
            if isinstance(node, ast.Raise) and node.exc:
                if isinstance(node.exc, ast.Name):
                    raises.append(node.exc.id)
                elif isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
                    raises.append(node.exc.func.id)
        return list(set(raises))

    def _get_caught_exceptions(self, func_node: ast.FunctionDef) -> List[str]:
        """Extract exceptions that are caught in try/except blocks."""
        catches = []
        for node in ast.walk(func_node):
            if isinstance(node, ast.ExceptHandler):
                if node.type:
                    if isinstance(node.type, ast.Name):
                        catches.append(node.type.id)
                    elif isinstance(node.type, ast.Tuple):
                        for elt in node.type.elts:
                            if isinstance(elt, ast.Name):
                                catches.append(elt.id)
        return list(set(catches))

    def _is_generator(self, func_node: ast.FunctionDef) -> bool:
        """Check if function is a generator (uses yield)."""
        for node in ast.walk(func_node):
            if isinstance(node, (ast.Yield, ast.YieldFrom)):
                return True
        return False

    def _get_external_dependencies(self, func_node: ast.FunctionDef) -> List[str]:
        """Extract external dependencies used in the method."""
        dependencies = []

        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    # Method calls like math.sqrt
                    full_name = ast.unparse(node.func)
                    dependencies.append(full_name)
                elif isinstance(node.func, ast.Name):
                    # Function calls
                    dependencies.append(node.func.id)
            elif isinstance(node, ast.Name):
                # Variable references that might be external
                if node.id in [m.split('.')[-1] for m in self.modules]:
                    dependencies.append(node.id)

        return list(set(dependencies))

    def _get_input_parameters(self, func_node: ast.FunctionDef) -> List[str]:
        """Extract input parameter names."""
        params = []
        for arg in func_node.args.args:
            params.append(arg.arg)

        # Add *args and **kwargs if present
        if func_node.args.vararg:
            params.append(f"*{func_node.args.vararg.arg}")
        if func_node.args.kwarg:
            params.append(f"**{func_node.args.kwarg.arg}")

        return params

    def _get_method_module_paths(self, func_node: ast.FunctionDef) -> List[str]:
        """Get module paths used specifically in this method."""
        method_modules = []

        # This is a simplified approach - in practice, you'd need more sophisticated analysis
        # to determine which imports are actually used in each method
        for node in ast.walk(func_node):
            if isinstance(node, ast.Attribute):
                parts = ast.unparse(node).split('.')
                if len(parts) > 1:
                    potential_module = parts[0]
                    for module in self.modules:
                        if module.endswith(potential_module) or potential_module in module:
                            method_modules.append(module)

        return list(set(method_modules))

    def _get_package_name(self) -> Optional[str]:
        """Extract package name from file path."""
        path_parts = self.file_path.split(os.sep)
        # Find if there's a parent directory that could be a package
        for i, part in enumerate(path_parts[:-1]):
            if part and not part.startswith('.'):
                # Check if there's an __init__.py in this directory
                potential_package_dir = os.sep.join(path_parts[:i+1])
                if os.path.exists(os.path.join(potential_package_dir, '__init__.py')):
                    return part
        return None
