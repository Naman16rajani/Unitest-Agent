class Prompt():
    """
    This class contains the prompt for Agent One.
    """
    def __init__(self,prompt:str):
        self.prompt = prompt


    def get_prompt(self) -> str:
        return self.prompt