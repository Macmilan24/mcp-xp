from typing import List, Dict
from abc import ABC, abstractmethod
from app.AI.llm_config._base_config import LLMModelConfig

# Abstract Provider Base Class
class LLMProvider(ABC):
    def __init__(self, model_config: LLMModelConfig) -> None:
        self.config = model_config

    @abstractmethod
    async def get_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Retrieve a response given a list of messages.
        """
        pass
