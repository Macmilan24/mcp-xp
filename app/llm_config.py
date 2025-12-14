from typing import Any, Dict
from app.config import GEMINI_API_KEY, OPENAI_API_KEY

# Configuration Classes
class LLMModelConfig:
    def __init__(self, config_data: Dict[str, Any]) -> None:
        self.config_data = config_data
        self.model_name: str = config_data["model"]  # e.g., "llama-3.2-90b-vision-preview"
        self.provider: str = config_data["provider"]  # e.g., "groq"
        self.base_url: str = config_data["base_url"]
        self.embedding_model: str | None = config_data.get("embedding_model")
        

class GEMINIConfig(LLMModelConfig):
    @property
    def api_key(self) -> str:
        return GEMINI_API_KEY

class OPENAIConfig(LLMModelConfig):
    @property
    def api_key(self) -> str:
        return OPENAI_API_KEY