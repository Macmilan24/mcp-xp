from abc import ABC, abstractmethod
from typing import Any, Dict, List
import httpx
from app.config import GROQ_API_KEY, AZURE_API_KEY
from openai import OpenAI, AsyncOpenAI  # Use AsyncOpenAI for async support
import json

# Configuration Classes
class LLMModelConfig:
    def __init__(self, config_data: Dict[str, Any]) -> None:
        self.config_data = config_data
        self.model_name: str = config_data["model"]  # e.g., "llama-3.2-90b-vision-preview"
        self.provider: str = config_data["provider"]  # e.g., "groq"
        self.base_url: str = config_data["base_url"]

class GROQConfig(LLMModelConfig):
    @property
    def api_key(self) -> str:
        print(GROQ_API_KEY)
        return GROQ_API_KEY

class AZUREConfig(LLMModelConfig):
    @property
    def api_key(self) -> str:
        return AZURE_API_KEY

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

# Groq Provider
class GroqProvider(LLMProvider):
    async def get_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Sends a request to the Groq API and returns the generated response.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }
        payload = {
            "messages": messages,
            "model": self.config.model_name,
            "temperature": self.config.config_data.get("temperature", 0.7),
            "max_tokens": self.config.config_data.get("max_tokens", 4096),
            "top_p": self.config.config_data.get("top_p", 1),
            "stream": self.config.config_data.get("stream", False),
            "stop": self.config.config_data.get("stop", None),
        }
        try:
            with httpx.Client() as client:
                response = client.post(self.config.base_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                # print(data["choices"][0]["message"]["content"])
                return data["choices"][0]["message"]["content"]


        except httpx.RequestError as e:
            error_message = f"Error getting LLM response: {str(e)}"
            # logging.error(error_message)

            if isinstance(e, httpx.HTTPStatusError):
                status_code = e.response.status_code
                # logging.error(f"Status code: {status_code}")
                # logging.error(f"Response details: {e.response.text}")

            return (
                f"I encountered an error: {error_message}. "
                "Please try again or rephrase your request."
            )

# Azure Provider
class AzureProvider(LLMProvider):
    async def get_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Sends a request to the Azure API using streaming and returns the complete response text.
        """
        client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )

        try:
            response = await client.chat.completions.create(
                messages=messages,
                model=self.config.model_name,
                stream=True,
                stream_options={'include_usage': True}
            )
            content_chunks = []
            async for update in response:
                if update.choices and update.choices[0].delta:
                    chunk = update.choices[0].delta.content or ""
                    content_chunks.append(chunk)
            return "".join(content_chunks)
        except Exception as exc:
            raise RuntimeError(f"Azure API error: {exc}") from exc
        finally:
            await client.close()

