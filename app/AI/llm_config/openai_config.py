from app.config import OPENAI_API_KEY
from app.AI.llm_config._base_config import LLMModelConfig


class OPENAIConfig(LLMModelConfig):
    @property
    def api_key(self) -> str:
        return OPENAI_API_KEY