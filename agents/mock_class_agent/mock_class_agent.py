from langchain_core.language_models import BaseLanguageModel
from agents.mock_class_agent.mock_class_agent_prompt import MockAgentPrompt
from agents.agent import Agent


class MockClassAgent(Agent):
    def __init__(self, llm: BaseLanguageModel):
        mock_prompt = MockAgentPrompt()
        system_prompt = mock_prompt.get_prompt()
        super().__init__(llm, system_prompt)

    def invoke(
        self,
        user_prompt,
        folder_structure,
        input_folder_path=None,
        output_folder_path=None,
    ):
        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path
        prompt = "file_path: " + str(self.input_folder_path) + "\n" + "folder_structure: " + str(folder_structure) + "\n" + "unittest_path: " + str(self.output_folder_path) + "\n" + str(user_prompt)
        return self._invoke(user_prompt=prompt)
