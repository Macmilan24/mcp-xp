from app.config import AZURE_API_KEY
from app.AI.llm_config._base_config import LLMModelConfig


class AZUREConfig(LLMModelConfig):
    @property
    def api_key(self) -> str:
        return AZURE_API_KEY
