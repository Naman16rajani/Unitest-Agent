from helpers.analysers.code_analyzer import CodeAnalyzer
from typing import Dict, List, Any, Optional


class MethodFormatterAnalyzer(CodeAnalyzer):
    def __init__(self):
        super().__init__()

   

    def format_analysis(self, file_path: str):
        analyze = self._analyze_file(file_path) or {}
        classes = analyze.get("classes", []) or []
        functions = analyze.get("functions", []) or []

        result: List[Dict[str, Any]] = []

        for class_dict in classes:
            if not isinstance(class_dict, dict):
                continue

            class_info: Dict[str, Any] = {}
            # Copy in only non-empty class-level fields
            for field in (
                "class_name",
                "class_description",
                "qualified_name",
                "package",
                "source_location",
                "class_method_names",
                "module_path",
            ):
                v = class_dict.get(field)
                if v not in (None, "", [], {}):
                    class_info[field] = v

            # Constructor (deep-cleaned)
            if class_dict.get("constructor"):
                class_info["constructor"] = self.remove_empty_values(
                    class_dict["constructor"]
                )

            # Dunder methods (proper emptiness check)
            if class_dict.get("dunder_methods"):
                class_info["dunder_methods"] = class_dict["dunder_methods"]

            # Methods
            class_methods = class_dict.get("class_methods") or {}
            if isinstance(class_methods, dict):
                for method_name, method in class_methods.items():
                    if not isinstance(method, dict):
                        continue
                    method_info = dict(class_info)  # start from class metadata
                    method_data: Dict[str, Any] = {"method_name": method_name}

                    # Only add non-empty, plus preserve explicit booleans
                    if method.get("method_code"):
                        method_data["method_code"] = method["method_code"]
                    if method.get("method_description"):
                        method_data["method_description"] = method["method_description"]
                    if method.get("return_type"):
                        method_data["return_type"] = method["return_type"]
                    if method.get("parameter_types"):
                        method_data["parameter_types"] = method["parameter_types"]
                    if method.get("decorators"):
                        method_data["decorators"] = method["decorators"]
                    if method.get("raises"):
                        method_data["raises"] = method["raises"]
                    if method.get("catches"):
                        method_data["catches"] = method["catches"]
                    if method.get("is_async") is not None:
                        method_data["is_async"] = method["is_async"]
                    if method.get("is_generator") is not None:
                        method_data["is_generator"] = method["is_generator"]
                    if method.get("external_dependencies"):
                        method_data["external_dependencies"] = method[
                            "external_dependencies"
                        ]
                    if method.get("input_parameters"):
                        method_data["input_parameters"] = method["input_parameters"]
                    if method.get("source_location"):
                        method_data["source_location"] = method["source_location"]
                    if method.get("module_path"):
                        method_data["module_path"] = method["module_path"]

                    method_info.update(method_data)
                    # Only keep if there's meaningful data after cleaning
                    # cleaned = self.remove_empty_values(method_info)
                    if method_info:
                        result.append(method_info)

        # Top-level functions
        for func_map in functions:
            if not isinstance(func_map, dict):
                continue
            for func_name, func in func_map.items():
                if not isinstance(func, dict):
                    continue
                # keep function name and non-empty fields
                filtered = {"function_name": func_name}
                for k, v in func.items():
                    if v not in (None, "", [], {}):
                        filtered[k] = v
                # cleaned = self.remove_empty_values(filtered)
                if filtered:
                    result.append(filtered)

        cleaned_result = self.remove_empty_values(result)
        
        return cleaned_result
