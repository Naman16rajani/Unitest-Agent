from agents.prompt import Prompt


class UnittestAgentPrompt(Prompt):
    """
    This class contains the prompt for Agent One.
    """
    def __init__(self,):
        prompt = (
            """# Purpose
Generate concise Pytest unit tests for a method in a specified module, using the provided object schema and fixture.

Begin with a concise checklist (3–7 bullets) outlining core sub-tasks to ensure methodical test generation.

# Instructions
- The method to be tested is 'methodName' in class 'className'.
- You will receive `file_path` and `unittest_path` for import resolution. Use these paths to resolve and write import statements correctly.
- Add correct imports for every module and class involved in constructing the MagicMock, using the correct relative import path based on `file_path` and `unittest_path`; do not use the literal file path as a module name.
- Do not use try-catch blocks when importing; simply use a plain import statement.
- Do not re-import classes, fixtures, or dependencies if they are already present in the provided code. Only add missing imports.
- Import the specified class from the correct module derived from the given `file_path`.
- If a fixture for the class is provided, always use the provided fixture for constructing instances in the tests. Do not create new fixtures for the same class.
- Do not create class instances in the tests if not required by the provided fixture.
- Do not assume or create additional context beyond what is provided in the provided files, imports, and fixtures.
- Generate only the minimal number of unittests required to achieve 100% code coverage for the method, covering all branches, exceptions, and edge cases.
- Output only the Python test code; do not include explanations or additional text.
- Invoke the method within the test suite to achieve thorough validation.
- Validate that the tests achieve 100% code coverage and correct use of the provided fixture.
- Write method-based unit tests (not class-based), naming each test as `test_{methodName}_{case}`.
- Adhere to Pytest best practices:
  - Use `pytest.mark.parametrize` for multiple input sets.
  - Use mock patching for methods from classes or functions in dependencies.
  - Ensure each test is isolated, repeatable, and independent of the environment.
- For each aspect of the method, include example tests for normal operation, error cases, and boundary conditions to ensure complete coverage.
- Place a descriptive docstring inside every test function, clarifying its intent and scope.
- Add in-line comments throughout the code for clarity.
- After generating the code, validate that all specified instructions have been addressed and that constructor/dependency mocking is handled as indicated. If validation fails, self-correct before completion.
"""
            # "# Purpose\n"
            # "Generate concise Pytest unit tests for a method in a specified method, using the provided object schema and fixture if available.\n\n"
            # "Begin with a concise checklist (3–7 bullets) outlining core sub-tasks to ensure methodical test generation.\n"
            # "# Instructions\n"
            # "- The method to be tested is 'methodName', located in class 'className'."
            # "- You will receive `file_path` and `unittest_path` for import resolution."
            # "- All necessary imports will be provided; only add imports not present in the provided code."
            # "- If the class fixture code is provided, use the fixture for constructing instances of the class in the tests."
            # "- Generate only the minimal number of unittests required to achieve 100% code coverage for the method, covering all branches, exceptions, and edge cases."
            # "- Output only the Python test code; do not include explanations or additional text."
            # "- Invoke the method within the test suite to achieve thorough validation.",
            # "- Validate that tests achieve 100% code coverage and correct use of the provided fixture.",
            # "- Write method-focused unit tests named `Test{methodName}` (e.g., if `methodName` is `addOne`, use `TestAddOne`).",
            # "- Be named using the format `mock_{dependency_class_name}_instance` (e.g., `mock_calculator_instance`).",
            # "- Adhere to Pytest best practices:",
            # "- Use `pytest.mark.parametrize` for multiple input sets.",
            # "- Use mock patching for methods from classes or functions in dependencies.",
            # "- Ensure each test is isolated, repeatable, and independent of the environment.",
            # "- For each aspect of the method, include example tests for normal operation, error cases, and boundary conditions to ensure complete coverage.",
            # "- Place a descriptive docstring inside every test function, clarifying its intent and scope.",
            # "- Add in-line comments throughout the code for clarity.",
            # "- Add only missing imports; do not include redundant ones.",
            # "- After generating the code, validate that all specified instructions have been addressed and that constructor/dependency mocking is handled as indicated. If validation fails, self-correct before completion.\n",
            # "# Purpose",
            # "Generate a complete Pytest unit test suite for a specific method in a specified class, using a provided object schema as input.",
            # "Begin with a concise checklist (3–7 bullets) outlining core sub-tasks to ensure methodical test generation.",
            # "# Instructions",
            # "- The method to be tested is 'methodName', located in the class 'className'.",
            # "- Instantiate the class using the 'constructor' details from the input schema.",
            # "- Invoke the method within the test suite to achieve thorough validation.",
            # "- Use provided code and generate tests focused solely on 'methodName'.",
            # "- Add correct imports for every module and class involved in constructing the MagicMock.",
            # "- Receive `file_path` and `unittest_path` as string input parameters for import resolution.",
            # "- Import the specified class from the given `file_path`.",
            # "## Requirements",
            # "- Write method-focused unit tests contained within a test class named `Test{methodName}` (e.g., if `methodName` is `addOne`, use `TestAddOne`).",
            # "- Achieve 100% code coverage for the method, including all execution branches, exceptions, and edge cases (including simulated interrupts/errors as appropriate).",
            # "- Mock all dependencies specified in `used_functions` and `used_classes` using `unittest.mock.MagicMock`, ensuring that tests are fully isolated from external influences.",
            # "- For any class dependency listed in `used_classes`, create a dedicated Pytest fixture that returns a `MagicMock` instance of that class. The fixture should:",
            # "- Be named using the format `mock_{dependency_class_name}_instance` (e.g., `mock_calculator_instance`).",
            # "- Initialize the mock using `MagicMock(spec=<DependencyClass>)`.",
            # "- Set up key attributes or methods as mockable (e.g., ensure lists/attributes that require tracking are also mocked if appropriate)",
            # "- Include a descriptive docstring inside the fixture clarifying its purpose.",
            # "- Structure tests using fixtures for set-up and tear-down (where necessary), parameterize tests using `pytest.mark.parametrize` for varied inputs, and include comprehensive assertions.",
            # "- Adhere to Pytest best practices:",
            # "- Use `pytest.mark.parametrize` for multiple input sets.",
            # "- Use mock patching for methods from classes or functions in dependencies.",
            # "- Ensure each test is isolated, repeatable, and independent of the environment.",
            # "- For each aspect of the method, include example tests for normal operation, error cases, and boundary conditions to ensure complete coverage.",
            # "- Place a descriptive docstring inside every test function, clarifying its intent and scope.",
            # "- Add in-line comments throughout the code for clarity.",
            # "- Output only the Python test code; do not include any additional text or explanations.",
            # "- After generating the code, validate that all specified requirements have been addressed and that constructor/dependency mocking is handled as indicated. If validation fails, self-correct before completion.",
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


# above prompt is for creating unittests in pytest framework. I am providing `file_path` and `unittest_path` for import resolution but still it is importing wrongly i want you to fix this and also it is using try catch while importing i dont want it and.  also i  am providing existing code  and it has some import and fixtures but it is still creating same fixture and imports, i dont want it .  and it not using provided fixtures for the classes. and i want method based uniitest not class based unittest. also mention do not create class instance and do not assume anyhting
