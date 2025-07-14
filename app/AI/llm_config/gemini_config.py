from app.config import GEMINI_API_KEY
from app.AI.llm_config._base_config import LLMModelConfig


class GEMINIConfig(LLMModelConfig):
    @property
    def api_key(self) -> str:
        return GEMINI_API_KEY
