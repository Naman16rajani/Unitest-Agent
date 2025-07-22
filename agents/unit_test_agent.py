from typing import Dict, Any
from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.language_model import BaseLanguageModel


class UnitTestAgent:
    """Agent for generating pytest unit tests using LangChain and LLM."""

    def __init__(self, llm: BaseLanguageModel):
        """
        Initialize the UnitTestAgent with a language model.

        Args:
            llm: LangChain language model instance
        """
        self.llm = llm
        self.system_prompt = """You are an expert Python test engineer specializing in writing comprehensive pytest unit tests.

Your task is to generate pytest unit tests based on the provided function/method analysis data. Follow these guidelines:

1. Always import pytest and unittest.mock (MagicMock, patch)
2. Import the actual class/module being tested
3. Mock constructor methods and external dependencies when needed
4. Write comprehensive test cases covering:
   - Normal/happy path scenarios
   - Edge cases
   - Error conditions and exception handling
   - Boundary values
5. Use descriptive test method names following the pattern: test_<method_name>_<scenario>
6. Include proper assertions for return values, side effects, and mock calls
7. Mock external functions and dependencies appropriately
8. If a class constructor is provided, use it to understand class initialization
9. Generate only the test code without explanations or comments
10. Ensure tests are executable and follow pytest conventions

Return only the Python test code, nothing else."""

    def generate_unit_test(
        self, function_data: Dict[str, Any], custom_prompt: str = None
    ) -> str:
        """
        Generate unit test code for a given function/method.

        Args:
            function_data: Dictionary containing function analysis data
            custom_prompt: Optional custom prompt to override the default system prompt

        Returns:
            Generated pytest unit test code as string
        """
        # Extract method/function name from the data
        method_name = self._extract_method_name(function_data)

        # Build the user prompt with function details
        user_prompt = self._build_user_prompt(function_data, method_name)

        # Use custom prompt if provided, otherwise use default system prompt
        system_prompt = (
            self.system_prompt + custom_prompt if custom_prompt else self.system_prompt
        )

        # Create messages for the LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # Generate the test code using LLM
        response = self.llm.invoke(messages)

        # Extract content from response
        if hasattr(response, "content"):
            return response.content
        else:
            return str(response)

    def _extract_method_name(self, function_data: Dict[str, Any]) -> str:
        """Extract the method/function name from the data structure."""
        # The function_data is structured as {'method_name': {...}}
        if isinstance(function_data, dict):
            return list(function_data.keys())[0]
        return "unknown_function"

    def _build_user_prompt(
        self, function_data: Dict[str, Any], method_name: str
    ) -> str:
        """Build the detailed user prompt for test generation."""
        method_info = function_data[method_name]

        prompt_parts = []

        # Add method information
        prompt_parts.append(
            f"Generate pytest unit tests for the following method: {method_name}"
        )
        prompt_parts.append(f"\nMethod Overview: {method_info.get('overview', 'N/A')}")

        # Add module information
        if "module_name" in method_info:
            prompt_parts.append(f"Module Name: {method_info['module_name']}")

        # Add method code
        if "code" in method_info:
            prompt_parts.append(f"\nMethod Code:\n{method_info['code']}")

        # Add inputs information
        if "inputs" in method_info:
            inputs_str = ", ".join(
                [f"{inp['name']}: {inp['type']}" for inp in method_info["inputs"]]
            )
            prompt_parts.append(f"\nInputs: {inputs_str}")

        # Add outputs information
        if "outputs" in method_info:
            outputs_str = ", ".join(
                [f"{out['type']}" for out in method_info["outputs"]]
            )
            prompt_parts.append(f"Outputs: {outputs_str}")

        # Add classes used
        if "classes_used" in method_info and method_info["classes_used"]:
            prompt_parts.append(
                f"Classes Used: {', '.join(method_info['classes_used'])}"
            )

        # Add functions used
        if "functions_used" in method_info and method_info["functions_used"]:
            prompt_parts.append(
                f"Functions Used: {', '.join(method_info['functions_used'])}"
            )

        # Add class constructor information if available
        if "class_constructor" in method_info:
            constructor = method_info["class_constructor"]
            prompt_parts.append("\nClass Constructor Information:")
            prompt_parts.append(f"Class Name: {constructor.get('class_name', 'N/A')}")
            if "code" in constructor:
                prompt_parts.append(f"Constructor Code:\n{constructor['code']}")

        # Add specific instructions
        if "instructions_to_follow" in method_info:
            prompt_parts.append(
                f"\nSpecific Instructions: {method_info['instructions_to_follow']}"
            )

        return "\n".join(prompt_parts)

    def generate_multiple_tests(
        self, functions_data: Dict[str, Dict[str, Any]], custom_prompt: str = None
    ) -> Dict[str, str]:
        """
        Generate unit tests for multiple functions/methods.

        Args:
            functions_data: Dictionary with multiple function analysis data
            custom_prompt: Optional custom prompt to override the default system prompt

        Returns:
            Dictionary mapping function names to their generated test codes
        """
        test_results = {}

        for func_name, func_data in functions_data.items():
            # Wrap single function data in expected format
            wrapped_data = {func_name: func_data}
            test_code = self.generate_unit_test(wrapped_data, custom_prompt)
            test_results[func_name] = test_code

        return test_results

    def generate_test_suite(
        self, class_methods_data: Dict[str, Any], custom_prompt: str = None
    ) -> str:
        """
        Generate a complete test suite for a class with multiple methods.

        Args:
            class_methods_data: Dictionary containing class methods analysis data
            custom_prompt: Optional custom prompt to override the default system prompt

        Returns:
            Complete test suite as string
        """
        test_suite_parts = []

        # Generate tests for each method
        for method_name, method_data in class_methods_data.items():
            wrapped_data = {method_name: method_data}
            test_code = self.generate_unit_test(wrapped_data, custom_prompt)
            test_suite_parts.append(f"# Tests for {method_name}")
            test_suite_parts.append(test_code)
            test_suite_parts.append("")  # Add spacing between test methods

        return "\n".join(test_suite_parts)


# Example usage
def example_usage():
    """Example of how to use the UnitTestAgent."""
    # Note: You would need to provide an actual LLM instance
    # from langchain.llms import OpenAI  # or any other LLM
    # llm = OpenAI(temperature=0)

    # Initialize agent (you need to provide actual LLM)
    # agent = UnitTestAgent(llm)

    # Generate test for single method
    # test_code = agent.generate_unit_test(sample_data)
    # print(test_code)

    return "Example setup complete. Provide LLM instance to run."


if __name__ == "__main__":
    print(example_usage())
