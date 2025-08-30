from agents.prompt import Prompt


class UnittestAgentPrompt(Prompt):
    """
    This class contains the prompt for Agent One.
    """
    def __init__(self,):
        prompt = (
            "Using the above object schema as input, generate a complete Pytest unit test suite for the specified 'methodName'. This method is inside the class 'className'. To test it, create an instance of the class using the constructor details and then invoke the method.\n"
            "Requirements:\n"
            "- Create method-based unit tests focused solely on 'methodName' with Test{methodname}. for example: methodname is addOne then test will TestAddOne. \n"
            "- Aim for 100% code coverage of this method, including all branches, exceptions, and edge cases (e.g., simulate interrupts or errors if present).\n"
            "- Use 'unittest.mock.MagicMock' to mock all dependencies listed in 'used_functions' and 'used_classes' intelligently, ensuring tests isolate the method and run without external errors or dependencies.\n"
            "- Structure the tests with fixtures for setup/teardown if needed, parameterized tests for multiple inputs, and assertions for outputs.\n"
            "- Follow Pytest best practices: use 'pytest.mark.parametrize' for variations, mock patches for class methods, and ensure tests are isolated and repeatable.\n"
            "- Include examples for happy paths, error handling, and boundary conditions to achieve full coverage.\n"
            "- create a doc string inside function explaining the overall purpose of the tests, what they cover\n"
            "- also write comments inside the code\n"
            "- Output the tests as python code only do not add anything else\n"
        )
        super().__init__(prompt)
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