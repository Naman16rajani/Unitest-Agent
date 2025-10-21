from agents.prompt import Prompt


class UnittestAgentPrompt(Prompt):
    """
    This class contains the prompt for Agent One.
    """
    def __init__(self,):
        prompt = (
            "# Purpose",
            "Generate a complete Pytest unit test suite for a specific method in a specified class, using a provided object schema as input.",
            "Begin with a concise checklist (3–7 bullets) outlining core sub-tasks to ensure methodical test generation.",
            "# Instructions",
            "- The method to be tested is 'methodName', located in the class 'className'.",
            "- Instantiate the class using the 'constructor' details from the input schema.",
            "- Invoke the method within the test suite to achieve thorough validation.",
            "## Requirements",
            "- Write method-focused unit tests contained within a test class named `Test{methodName}` (e.g., if `methodName` is `addOne`, use `TestAddOne`).",
            "- Achieve 100% code coverage for the method, including all execution branches, exceptions, and edge cases (including simulated interrupts/errors as appropriate).",
            "- Mock all dependencies specified in `used_functions` and `used_classes` using `unittest.mock.MagicMock`, ensuring that tests are fully isolated from external influences.",
            "- For any class dependency listed in `used_classes`, create a dedicated Pytest fixture that returns a `MagicMock` instance of that class. The fixture should:",
            "- Be named using the format `mock_{dependency_class_name}_instance` (e.g., `mock_calculator_instance`).",
            "- Initialize the mock using `MagicMock(spec=<DependencyClass>)`.",
            "- Set up key attributes or methods as mockable (e.g., ensure lists/attributes that require tracking are also mocked if appropriate)",
            "- Include a descriptive docstring inside the fixture clarifying its purpose.",
            "- Structure tests using fixtures for set-up and tear-down (where necessary), parameterize tests using `pytest.mark.parametrize` for varied inputs, and include comprehensive assertions.",
            "- Adhere to Pytest best practices:",
            "- Use `pytest.mark.parametrize` for multiple input sets.",
            "- Use mock patching for methods from classes or functions in dependencies.",
            "- Ensure each test is isolated, repeatable, and independent of the environment.",
            "- For each aspect of the method, include example tests for normal operation, error cases, and boundary conditions to ensure complete coverage.",
            "- Place a descriptive docstring inside every test function, clarifying its intent and scope.",
            "- Add in-line comments throughout the code for clarity.",
            "- Output only the Python test code; do not include any additional text or explanations.",
            "- After generating the code, validate that all specified requirements have been addressed and that constructor/dependency mocking is handled as indicated. If validation fails, self-correct before completion."
        )
        super().__init__(prompt.__str__())
    # @staticmethod
    # def get_prompt() -> str:
    #     return (
    #         "Using the above object schema as input, generate a complete Pytest unit test suite for the specified 'methodName'. This method is inside the class 'className'. To test it, create an instance of the class using the constructor details and then invoke the method.\n"
    #         "Requirements:\n"
    #         "- Create method-based unit tests focused solely on 'methodName' with Test{methodname}. for example: methodname is addOne then test will TestAddOne. \n"
    #         "- Aim for 100% code coverage of this method, including all branches, exceptions, and edge cases (e.g., simulate interrupts or errors if present).\n"
    #         "- Use 'unittest.mock.MagicMock' to mock all dependencies listed in 'used_functions' and 'used_classes' intelligently, ensuring tests isolate the method and run without external errors or dependencies.\n"
    #         "- Structure the tests with fixtures for setup/teardown if needed, parameterized tests for multiple inputs, and assertions for outputs.\n"
    #         "- Follow Pytest best practices: use 'pytest.mark.parametrize' for variations, mock patches for class methods, and ensure tests are isolated and repeatable.\n"
    #         "- Include examples for happy paths, error handling, and boundary conditions to achieve full coverage.\n"
    #         "- create a doc string inside function explaining the overall purpose of the tests, what they cover\n"
    #         "- also write comments inside the code\n"
    #         "- Output the tests as python code only do not add anything else\n"
    #     )