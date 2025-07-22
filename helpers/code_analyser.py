import ast
from typing import Dict, Any


class CodeAnalyzer:
    """Analyzes Python code to extract class and function information."""

    def __init__(self):
        self.classes = []
        self.functions = []
        self.imports = []
        self.code_lines = []  # Store original code lines
        self.module_name = ""  # Store module name

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a Python file and extract its structure."""
        with open(file_path, "r", encoding="utf-8") as file:
            code_content = file.read()

        try:
            tree = ast.parse(code_content)
            self.code_lines = (
                code_content.splitlines()
            )  # Store code lines for extraction

            # Extract module name from file path
            import os

            self.module_name = os.path.splitext(os.path.basename(file_path))[0]

            self._analyze_node(tree)
            return self._generate_analysis()
        except SyntaxError as e:
            return {"error": f"Syntax error in file: {e}"}

    def _analyze_node(self, node):
        """Recursively analyze AST nodes."""
        for child in ast.walk(node):
            if isinstance(child, ast.ClassDef):
                self._analyze_class(child)
            elif isinstance(child, ast.FunctionDef):
                # Only include top-level functions (not methods)
                if not any(
                    isinstance(parent, ast.ClassDef)
                    for parent in ast.walk(node)
                    if any(c == child for c in ast.walk(parent))
                ):
                    self._analyze_function(child)
            elif isinstance(child, ast.Import) or isinstance(child, ast.ImportFrom):
                self._analyze_import(child)

    def _analyze_class(self, class_node: ast.ClassDef):
        """Analyze a class definition."""
        class_info = {
            "name": class_node.name,
            "methods": [],
            "used_classes": set(),
            "used_functions": set(),
        }

        # Analyze class methods
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                method_info = self._analyze_function(node, is_method=True)
                class_info["methods"].append(method_info)
                class_info["used_classes"].update(
                    method_info.get("used_classes", set())
                )
                class_info["used_functions"].update(
                    method_info.get("used_functions", set())
                )

        # Analyze class body for other references
        for node in ast.walk(class_node):
            if isinstance(node, ast.Name):
                # Check if it's a known class (starts with uppercase or is a known exception)
                if node.id[0].isupper() or node.id in [
                    "ValueError",
                    "TypeError",
                    "Exception",
                    "RuntimeError",
                    "KeyError",
                    "IndexError",
                ]:
                    class_info["used_classes"].add(node.id)
            elif isinstance(node, ast.Call):
                # Handle function calls
                if isinstance(node.func, ast.Name):
                    class_info["used_functions"].add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    # Handle method calls like self.add()
                    if hasattr(node.func, "attr"):
                        class_info["used_functions"].add(node.func.attr)

        # Convert sets to lists for JSON serialization and filter out built-ins
        builtin_functions = {
            "len",
            "sum",
            "max",
            "min",
            "range",
            "isinstance",
            "str",
            "int",
            "float",
            "bool",
        }
        actual_functions = class_info["used_functions"] - builtin_functions

        class_info["used_classes"] = list(class_info["used_classes"])
        class_info["used_functions"] = list(actual_functions)

        self.classes.append(class_info)

    def _analyze_function(self, func_node: ast.FunctionDef, is_method: bool = False):
        """Analyze a function definition."""
        func_info = {
            "name": func_node.name,
            "inputs": [],
            "outputs": [],
            "used_classes": set(),
            "used_functions": set(),
            "is_method": is_method,
            "code": self._extract_function_code(func_node),
        }

        # Analyze function arguments
        for arg in func_node.args.args:
            arg_info = {"name": arg.arg, "type": "Any"}
            if arg.annotation:
                arg_info["type"] = self._get_annotation_string(arg.annotation)
            func_info["inputs"].append(arg_info)

        # Analyze return type
        if func_node.returns:
            func_info["outputs"] = [
                {"type": self._get_annotation_string(func_node.returns)}
            ]
        else:
            func_info["outputs"] = [{"type": "Any"}]

        # Analyze function body for references
        for node in ast.walk(func_node):
            if isinstance(node, ast.Name):
                # Check if it's a known class (starts with uppercase or is a known exception)
                if node.id[0].isupper() or node.id in [
                    "ValueError",
                    "TypeError",
                    "Exception",
                    "RuntimeError",
                    "KeyError",
                    "IndexError",
                ]:
                    func_info["used_classes"].add(node.id)
            elif isinstance(node, ast.Call):
                # Handle function calls
                if isinstance(node.func, ast.Name):
                    func_info["used_functions"].add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    # Handle method calls like self.add()
                    if hasattr(node.func, "attr"):
                        if isinstance(node.func.value, ast.Name):
                            # For calls like logging.info() or self.add()
                            func_info["used_functions"].add(
                                f"{node.func.value.id}.{node.func.attr}"
                            )
                        else:
                            # For more complex attribute access, just use the method name
                            func_info["used_functions"].add(node.func.attr)

        # Convert sets to lists and filter out built-ins and variables
        # Filter out built-in functions and variables that aren't actual function calls
        builtin_functions = {
            "len",
            "sum",
            "max",
            "min",
            "range",
            "isinstance",
            "str",
            "int",
            "float",
            "bool",
        }
        actual_functions = func_info["used_functions"] - builtin_functions

        func_info["used_classes"] = list(func_info["used_classes"])
        func_info["used_functions"] = list(actual_functions)

        if not is_method:
            self.functions.append(func_info)

        return func_info

    def _analyze_import(self, import_node):
        """Analyze import statements."""
        if isinstance(import_node, ast.Import):
            for alias in import_node.names:
                self.imports.append(alias.name)
        elif isinstance(import_node, ast.ImportFrom):
            module = import_node.module or ""
            for alias in import_node.names:
                self.imports.append(f"{module}.{alias.name}")

    def _get_annotation_string(self, annotation):
        """Convert AST annotation to string."""
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Constant):
            return str(annotation.value)
        elif isinstance(annotation, ast.Attribute):
            return f"{annotation.value.id}.{annotation.attr}"
        else:
            return "Any"

    def _extract_function_code(self, func_node: ast.FunctionDef) -> str:
        """Extract the source code of a function."""
        if not self.code_lines:
            return ""

        start_line = func_node.lineno - 1  # Convert to 0-based indexing
        end_line = func_node.end_lineno if func_node.end_lineno else start_line + 1

        # Extract the function code
        function_lines = self.code_lines[start_line:end_line]
        return "\n".join(function_lines)

    def _generate_analysis(self) -> Dict[str, Any]:
        """Generate the final analysis structure."""
        analysis = {}

        # Add classes
        for class_info in self.classes:
            class_name = class_info["name"]
            analysis[class_name] = {
                "overview": f"Class {class_name}",
                "other_classes_used": [
                    cls for cls in class_info["used_classes"] if cls != class_name
                ],
                "other_functions_used": class_info["used_functions"],
                "class_functions": {},
                "module_name": self.module_name,  # Classes are not modules
            }

            # Add methods
            for method in class_info["methods"]:
                method_name = method["name"]
                analysis[class_name]["class_functions"][method_name] = {
                    "inputs": method["inputs"],
                    "outputs": method["outputs"],
                    "overview": f"Method {method_name}",
                    "functions_used": method["used_functions"],
                    "classes_used": method["used_classes"],
                    "code": method["code"],
                    "module_name": self.module_name,  # Methods are not modules
                }

        # Add standalone functions
        for func_info in self.functions:
            func_name = func_info["name"]
            analysis[func_name] = {
                "inputs": func_info["inputs"],
                "outputs": func_info["outputs"],
                "overview": f"Function {func_name}",
                "functions_used": func_info["used_functions"],
                "classes_used": func_info["used_classes"],
                "code": func_info["code"],
                "module_name": self.module_name,  # Functions are not modules
            }

        # Add module information
        # if self.module_name:
        #     analysis[f"module_{self.module_name}"] = {
        #         "module_name": self.module_name,
        #         "overview": f"Module {self.module_name}",
        #         "imports": self.imports,
        #         "classes": [cls["name"] for cls in self.classes],
        #         "functions": [func["name"] for func in self.functions],
        #         "is_module": True,  # This is a module
        #     }

        return analysis
