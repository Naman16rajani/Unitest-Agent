from typing import Dict, Any
from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.language_model import BaseLanguageModel
from abc import ABC, abstractmethod
from helpers.logging_config import logger
from helpers.clean_code import clean_code


class Agent(ABC):
    """Agent for generating pytest unit tests using LangChain and LLM."""

    def __init__(self, llm: BaseLanguageModel, system_prompt: str):
        """
        Initialize the UnitTestAgent with a language model.

        Args:
            llm: LangChain language model instance
        """
        self.llm = llm
        self.system_prompt = system_prompt

    
    def _invoke(self, user_prompt: str):
        logger.debug("User prompt:\n" + str(user_prompt))

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=str(user_prompt)),
        ]
        logger.debug("Messages sent to LLM:\n" + str(messages))
        response = self.llm.invoke(messages)

        # Extract content from response
        if hasattr(response, "content"):
            response_content = response.content
        else:
            response_content = str(response)

        output_tokens = (
            self.llm.get_num_tokens(response_content)
            if hasattr(self.llm, "get_num_tokens")
            else 0
        )

        response_content = clean_code(response_content)

        return response_content, output_tokens
