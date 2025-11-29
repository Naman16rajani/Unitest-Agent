from langchain_core.language_models import BaseLanguageModel
from helpers.logging_config import create_logger
from agents.agent import Agent
from agents.unittest_agent.unittest_agent_prompt import UnittestAgentPrompt
from langchain.schema import HumanMessage, SystemMessage, messages

from helpers.clean_code import clean_code

logger = create_logger("UnitTestAgent")


class UnitTestAgent(Agent):
    def __init__(self, llm: BaseLanguageModel):
        unittest_prompt = UnittestAgentPrompt()
        system_prompt = unittest_prompt.get_prompt()

        super().__init__(llm, system_prompt)

    def invoke(
        self,
        user_prompt,
        code,
        folder_structure,
        existing_test_analysis: str = None,
    ) -> tuple[str, int]:
        logger.debug("Invoking UnitTestAgent...")
        logger.debug("Existing Code:\n" + str(code))
        logger.debug("User Prompt:\n" + str(user_prompt))

        prompt_parts = ["folder_structure: \n" + str(folder_structure)]

        if code:
            prompt_parts.append("Existing code:\n" + str(code))

        if existing_test_analysis:
            logger.debug("Existing Test Analysis:\n" + str(existing_test_analysis))
            prompt_parts.append(
                "Analysis of Existing Tests:\n" + str(existing_test_analysis)
            )

        prompt_parts.append(str(user_prompt))

        full_prompt = "\n".join(prompt_parts)
        return self._invoke(user_prompt=full_prompt)
