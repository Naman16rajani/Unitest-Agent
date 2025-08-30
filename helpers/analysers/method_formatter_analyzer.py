from helpers.analysers.code_analyzer import CodeAnalyzer
from typing import Dict, List, Any, Optional, Union


class MethodFormatterAnalyzer(CodeAnalyzer):
    def __init__(self):
        super().__init__()

    def format_analysis(self,file_path: str)-> List[Any]:
        analyze = self.analyze_file(file_path)
        result = []
        for class_dict in analyze["classes"]:

            class_info = {
                "class_name": class_dict["class_name"],  # Fixed: added quotes
                "class_description": class_dict["class_description"],
                "qualified_name": class_dict["qualified_name"],
                "package": class_dict["package"],
                "source_location": class_dict["source_location"],
                "constructor": class_dict["constructor"],
                "dunder_methods": class_dict["dunder_methods"],
                "class_method_names": class_dict["class_method_names"],
                # "module_path": class_dict["module_path"]
            }

            for key,value in class_dict["class_methods"].items():
                k = key
                method_info = class_info.copy()
                method_info.update({
                    "method_name": value["method_name"],
                    "method_code": value["method_code"],
                    "method_description": value["method_description"],
                    "return_type": value["return_type"],
                    "parameter_types": value["parameter_types"],
                    "decorators": value["decorators"],
                    "raises": value["raises"],
                    "catches": value["catches"],
                    "is_async": value["is_async"],
                    "is_generator": value["is_generator"],
                    "external_dependencies": value["external_dependencies"],
                    "input_parameters": value["input_parameters"],
                    "source_location": value["source_location"],
                    "module_path": value["module_path"]
                })
                result.append(method_info)

        for method_dict in analyze["functions"]:
            for key, value in method_dict.items():
                result.append(value)

        return result