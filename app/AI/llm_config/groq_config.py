from app.config import GROQ_API_KEY
from app.AI.llm_config._base_config import LLMModelConfig


class GROQConfig(LLMModelConfig):
    @property
    def api_key(self) -> str:
        return GROQ_API_KEY
