from langchain_core.language_models import BaseLanguageModel
from helpers.logging_config import logger
from agents.agent import Agent
from agents.unittest_agent.unittest_agent_prompt import UnittestAgentPrompt
from langchain.schema import HumanMessage, SystemMessage, messages

from helpers.clean_code import clean_code


class UnitTestAgent(Agent):
    def __init__(self,llm: BaseLanguageModel):
        unittest_prompt = UnittestAgentPrompt()
        system_prompt = unittest_prompt.get_prompt()

        super().__init__(llm,system_prompt)

    def invoke(self,user_prompt):
        logger.debug("User prompt:\n" + str(user_prompt))

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=str(user_prompt)),
        ]
        logger.debug("Messages sent to LLM:\n" + str(messages))
        response = self.llm.invoke(messages)

        # Extract content from response
        if hasattr(response, "content"):
            response = response.content
        else:
            response =  str(response)

        response = clean_code(response)

        return response
