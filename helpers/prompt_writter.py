from typing import Dict, Any


class PromptWriter:
    """A class to add prompt instructions to function analysis data."""

    def __init__(self):
        self.default_prompt = "Follow best practices and ensure code quality."

    def add_prompt_to_function(
        self, function_data: Dict[str, Any], prompt: str
    ) -> Dict[str, Any]:
        """
        Add a prompt instruction to a single function's analysis data.

        Args:
            function_data: The function analysis dictionary
            prompt: The prompt instruction to add

        Returns:
            Updated function data with prompt instruction
        """
        if not isinstance(function_data, dict):
            raise ValueError("function_data must be a dictionary")

        # Create a copy to avoid modifying the original
        updated_data = function_data.copy()
        updated_data["instructions_to_follow"] = prompt

        return updated_data

    def add_prompt_to_class_methods(
        self, class_data: Dict[str, Any], prompt: str
    ) -> Dict[str, Any]:
        """
        Add a prompt instruction to all methods in a class's analysis data.

        Args:
            class_data: The class analysis dictionary
            prompt: The prompt instruction to add

        Returns:
            Updated class data with prompt instructions added to all methods
        """
        if not isinstance(class_data, dict):
            raise ValueError("class_data must be a dictionary")

        # Create a copy to avoid modifying the original
        updated_data = class_data.copy()

        if "class_functions" in updated_data:
            updated_functions = {}
            for method_name, method_data in updated_data["class_functions"].items():
                updated_functions[method_name] = self.add_prompt_to_function(
                    method_data, prompt
                )
            updated_data["class_functions"] = updated_functions

        return updated_data

    def add_prompt_to_analysis(
        self, analysis_data: Dict[str, Any], prompts: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Add prompt instructions to the entire analysis data structure.

        Args:
            analysis_data: The complete analysis dictionary from CodeAnalyzer
            prompts: Dictionary mapping function/class names to their prompts
                    Format: {'function_name': 'prompt', 'ClassName.method_name': 'prompt'}

        Returns:
            Updated analysis data with prompt instructions
        """
        if not isinstance(analysis_data, dict):
            raise ValueError("analysis_data must be a dictionary")

        updated_analysis = {}

        for item_name, item_data in analysis_data.items():
            # Check if this is a class (has 'class_functions' key)
            if isinstance(item_data, dict) and "class_functions" in item_data:
                # Handle class
                updated_class = item_data.copy()
                updated_functions = {}

                for method_name, method_data in item_data["class_functions"].items():
                    # Look for specific method prompt or class-wide prompt
                    method_key = f"{item_name}.{method_name}"
                    prompt = prompts.get(
                        method_key, prompts.get(item_name, self.default_prompt)
                    )
                    updated_functions[method_name] = self.add_prompt_to_function(
                        method_data, prompt
                    )

                updated_class["class_functions"] = updated_functions
                updated_analysis[item_name] = updated_class

            else:
                # Handle standalone function
                prompt = prompts.get(item_name, self.default_prompt)
                updated_analysis[item_name] = self.add_prompt_to_function(
                    item_data, prompt
                )

        return updated_analysis

    def add_single_prompt_to_all(
        self, analysis_data: Dict[str, Any], prompt: str
    ) -> Dict[str, Any]:
        """
        Add the same prompt instruction to all functions and methods in the analysis.

        Args:
            analysis_data: The complete analysis dictionary from CodeAnalyzer
            prompt: The prompt instruction to add to all functions/methods

        Returns:
            Updated analysis data with the same prompt added everywhere
        """
        prompts = {}

        # Create a prompts dictionary with the same prompt for all items
        for item_name, item_data in analysis_data.items():
            if isinstance(item_data, dict) and "class_functions" in item_data:
                # Add prompt for the class
                prompts[item_name] = prompt
                # Add prompt for each method
                for method_name in item_data["class_functions"].keys():
                    prompts[f"{item_name}.{method_name}"] = prompt
            else:
                # Standalone function
                prompts[item_name] = prompt

        return self.add_prompt_to_analysis(analysis_data, prompts)

    def set_default_prompt(self, prompt: str):
        """Set the default prompt for functions that don't have specific prompts."""
        self.default_prompt = prompt

    def create_custom_prompts(self, base_prompt: str, **kwargs) -> str:
        """
        Create a custom prompt by formatting a base prompt with provided arguments.

        Args:
            base_prompt: Base prompt string with placeholders like {function_name}
            **kwargs: Keyword arguments to format the prompt

        Returns:
            Formatted prompt string
        """
        return base_prompt.format(**kwargs)

    def convert_to_function_list_format(
        self, analysis_data: Dict[str, Any], prompt: str
    ) -> Dict[str, Any]:
        """
        Convert analysis data to a flattened format with prompt + function_name instructions.

        Args:
            analysis_data: The complete analysis dictionary from CodeAnalyzer
            prompt: The base prompt to combine with function names

        Returns:
            Dictionary with format: {'class_name': [list_of_methods], 'function_name': function_details}
        """
        result = {}

        for item_name, item_data in analysis_data.items():
            # Check if this is a class (has 'class_functions' key)
            if isinstance(item_data, dict) and "class_functions" in item_data:
                # Handle class - convert to list of methods
                methods_list = []

                # Get the __init__ method to use as class_constructor
                init_method = item_data["class_functions"].get("__init__", None)

                for method_name, method_data in item_data["class_functions"].items():
                    # Create method entry with instructions_to_follow
                    method_entry = method_data.copy()

                    # Add instructions_to_follow to all methods
                    method_entry["instructions_to_follow"] = f"{prompt} {method_name}"

                    # Add class_constructor for all methods (including __init__ itself)
                    if init_method:
                        # Create a copy of the __init__ method with its own instructions
                        class_constructor = init_method.copy()
                        class_constructor["class_name"] = item_name  # Add class name
                        # Don't add instructions_to_follow to the constructor itself
                        method_entry["class_constructor"] = class_constructor

                    methods_list.append({method_name: method_entry})

                result[item_name] = methods_list

            else:
                # Handle standalone function
                function_entry = item_data.copy()
                function_entry["instructions_to_follow"] = f"{prompt} {item_name}"
                result[item_name] = function_entry

        return result


# Example usage functions
def example_usage():
    """Example of how to use the PromptWriter class."""

    # Sample analysis data (like what you'd get from CodeAnalyzer)
    sample_analysis = {
        "Calculator": {
            "overview": "Class Calculator",
            "other_classes_used": ["ValueError"],
            "other_functions_used": ["append"],
            "class_functions": {
                "divide": {
                    "classes_used": ["ValueError"],
                    "code": '    def divide(self, a: float, b: float) -> float:\n        """Divide a by b"""\n        if b == 0:\n            raise ValueError("Cannot divide by zero")\n        result = a / b\n        self.history.append(f"{a} / {b} = {result}")\n        return result',
                    "functions_used": ["ValueError", "append"],
                    "inputs": [
                        {"name": "self", "type": "Any"},
                        {"name": "a", "type": "float"},
                        {"name": "b", "type": "float"},
                    ],
                    "outputs": [{"type": "float"}],
                    "overview": "Method divide",
                }
            },
        }
    }

    # Initialize PromptWriter
    prompt_writer = PromptWriter()

    # Method 1: Add specific prompts to specific functions
    prompts = {
        "Calculator.divide": "Ensure proper error handling for division by zero and validate input types."
    }

    updated_analysis = prompt_writer.add_prompt_to_analysis(sample_analysis, prompts)

    # Method 2: Add the same prompt to all functions
    universal_prompt = "Follow best coding practices and ensure robust error handling."
    updated_analysis_universal = prompt_writer.add_single_prompt_to_all(
        sample_analysis, universal_prompt
    )

    # Method 3: Convert to function list format (NEW METHOD)
    base_prompt = "Implement the following function with best practices:"
    function_list_format = prompt_writer.convert_to_function_list_format(
        sample_analysis, base_prompt
    )

    return updated_analysis, updated_analysis_universal, function_list_format


# if __name__ == "__main__":
#     # Run example
#     result1, result2, result3 = example_usage()
#     print("Updated analysis with specific prompts:")
#     print(result1["Calculator"]["class_functions"]["divide"]["instructions_to_follow"])

#     print("\nFunction list format:")
#     print("Calculator methods:", len(result3["Calculator"]))
#     if result3["Calculator"]:
#         first_method = list(result3["Calculator"][0].keys())[0]
#         print(
#             f"First method '{first_method}' instructions:",
#             result3["Calculator"][0][first_method]["instructions_to_follow"],
#         )
