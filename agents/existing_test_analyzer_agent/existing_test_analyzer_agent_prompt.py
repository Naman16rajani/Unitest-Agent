from agents.prompt import Prompt


class ExistingTestAnalyzerAgentPrompt(Prompt):
    """
    This class contains the prompt for the Existing Test Analyzer Agent.
    """

    def __init__(self):
        prompt = """# Purpose
Analyze the provided existing unit test code and extract key information to guide the generation of new tests.

# Instructions
- You will receive the content of an existing unit test file.
- Analyze the code to understand:
    1. The testing style (e.g., pytest fixtures, parametrization, naming conventions).
    2. Which methods or scenarios are already tested.
    3. Any existing fixtures or helper functions that should be reused.
    4. Any specific patterns or constraints observed in the existing tests.
- Provide a concise summary that can be passed to a unit test generation agent.
- The summary should explicitly list:
    - "Existing Tests": A list of test function names or descriptions of what is covered.
    - "Reusable Components": Names of fixtures or helper functions available in the file.
    - "Style Guidelines": Observations on coding style to maintain consistency.
- If the provided code is empty or contains no tests, state that no existing tests were found.
- Output the analysis in a structured format (e.g., bullet points or a JSON-like structure) that is easy for another LLM to parse.
"""
        super().__init__(prompt.__str__())
