from helpers.analysers.code_analyzer import CodeAnalyzer
from typing import Dict, List, Any, Optional


class ClassFormatterAnalyzer(CodeAnalyzer):
    def __init__(self):
        super().__init__()

    def format_analysis(self, file_path: str):
        analyze = self._analyze_file(file_path) or {}
        simplified = []

        for cls in analyze.get("classes", []):
            new_cls = {
                "class_name": cls.get("class_name"),
                "class_description": cls.get("class_description"),
                "qualified_name": cls.get("qualified_name"),
                "package": cls.get("package"),
                "file_path": cls.get("source_location").get("file_path"),
                "dunder_methods": cls.get("dunder_methods", []),
            }

            constructor = cls.get("constructor", {})
            if "__init__" in constructor:
                init_data = constructor["__init__"]
                new_cls["constructor"] = {
                    "code": init_data.get("method_code"),
                    "description": init_data.get("method_description"),
                    "return_type": init_data.get("return_type"),
                    "parameter_types": init_data.get("parameter_types"),
                    "decorators": init_data.get("decorators", []),
                    "raises": init_data.get("raises", []),
                    "catches": init_data.get("catches", []),
                    "is_async": init_data.get("is_async", False),
                    "is_generator": init_data.get("is_generator", False),
                    "external_dependencies": init_data.get("external_dependencies", []),
                    "input_parameters": init_data.get("input_parameters", []),
                    "file_path": init_data["source_location"].get("file_path"),
                    "module_path": init_data.get("module_path", [])
                }
            else:
                new_cls["constructor"] = {}

            simplified.append(new_cls)

        return self.remove_empty_values(simplified)
