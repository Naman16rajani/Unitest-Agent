from typing import Dict, Any
from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.language_model import BaseLanguageModel
from abc import ABC, abstractmethod


class Agent(ABC):
    """Agent for generating pytest unit tests using LangChain and LLM."""

    def __init__(self, llm: BaseLanguageModel,system_prompt:str):
        """
        Initialize the UnitTestAgent with a language model.

        Args:
            llm: LangChain language model instance
        """
        self.llm = llm
        self.system_prompt = system_prompt

    @abstractmethod
    def invoke(self,user_prompt:str):
        pass


