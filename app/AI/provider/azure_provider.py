from typing import Dict, List
from openai import AsyncOpenAI  # Use AsyncOpenAI for async support
from app.config import AZURE_API_KEY
from app.AI.provider._base_provider import LLMProvider 

# Azure Provider
class AzureProvider(LLMProvider):
    async def get_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Sends a request to the Azure API using streaming and returns the complete response text.
        """
        client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=AZURE_API_KEY,
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

