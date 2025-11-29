from langchain_core.language_models import BaseLanguageModel
from helpers.logging_config import create_logger
from agents.agent import Agent
from agents.existing_test_analyzer_agent.existing_test_analyzer_agent_prompt import (
    ExistingTestAnalyzerAgentPrompt,
)

logger = create_logger("ExistingTestAnalyzerAgent")


class ExistingTestAnalyzerAgent(Agent):
    def __init__(self, llm: BaseLanguageModel):
        prompt = ExistingTestAnalyzerAgentPrompt()
        system_prompt = prompt.get_prompt()
        super().__init__(llm, system_prompt)

    def invoke(self, code: str) -> tuple[str, int]:
        logger.debug("Invoking ExistingTestAnalyzerAgent...")
        if not code or not code.strip():
            return "No existing tests found.", 0

        user_prompt = f"Existing Test Code:\n{code}\n\nPlease analyze the above code."
        return self._invoke(user_prompt=user_prompt)
