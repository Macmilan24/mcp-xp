from typing import Any, Dict

# Configuration Classes
class LLMModelConfig:
    def __init__(self, config_data: Dict[str, Any]) -> None:
        self.config_data = config_data
        self.model_name: str = config_data["model"]  # e.g., "llama-3.2-90b-vision-preview"
        self.provider: str = config_data["provider"]  # e.g., "groq"
        self.base_url: str = config_data["base_url"]