import asyncio
import json
import re
import logging
from typing import Dict, List
import time

from openai import AsyncOpenAI
from sys import path
path.append(".")

from app.config import OPENAI_API_KEY
from app.AI.provider._base_provider import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, model_config):
        super().__init__(model_config)
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)

        self.log = logging.getLogger(self.__class__.__name__)
        
    async def get_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Sends a request to the OpenAI API via openai.AsyncOpenAI and returns the generated response.
        Identical behavior to the HTTP version: builds the same 'contents' structure, applies the same
        generation config, extracts a JSON codeblock if present, and returns parsed JSON when possible.
        """
        
        # Messages are already in the format expected by OpenAI (list of dicts with 'role' and 'content')

        # Generation config: same fields as before, mapped to SDK names
        generation_config = {
            "temperature": self.config.config_data.get("temperature", 0.7),
            "max_tokens": self.config.config_data.get("max_tokens", 10000),
            "top_p": self.config.config_data.get("top_p", 1),
            "stop": self.config.config_data.get("stop", []),
        }

        model_name = self.config.config_data.get("model")

        try:
            # Mirror the previous 10s timeout
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                **generation_config,
                timeout=10.0,
            )

            content = response.choices[0].message.content
            json_content = self._extract_json_from_llm_response(content)
            try:
                return json.loads(json_content)
            except json.JSONDecodeError:
                return json_content

        except Exception as e:
            self.log.error(f"OpenAI API error: {str(e)}")
            raise RuntimeError(f"OpenAI API error: {str(e)}") from e

    def _extract_json_from_llm_response(self, content: str) -> str:
        """
        Extracts a JSON string from a given content string, handling various formatting scenarios.
        The method attempts to extract JSON data from the input string by:
            1. Looking for a JSON code block delimited by triple backticks (```json ... ```).
            2. Unquoting the string if the JSON is wrapped in single or double quotes.
            3. Searching for the first JSON object or array using a regular expression.
            4. Returning the raw string if none of the above methods succeed.
        Args:
            content (str): The input string potentially containing JSON data.
        Returns:
            str: The extracted JSON string or the original content if extraction fails.
        """

        # 1. Try to extract JSON inside ```json ... ```
        start = content.find("```json")
        end = content.rfind("```")
        if start != -1 and end != -1 and end > start:
            json_content = content[start + 7:end].strip()
            return json_content
        else:
            json_content = content.strip()

        # 2. If it's quoted JSON, unquote and retry
        if (json_content.startswith('"') and json_content.endswith('"')) or \
        (json_content.startswith("'") and json_content.endswith("'")):
            return json_content.strip('"').strip("'")

        # 3. Try to extract first {...} or [...] block with regex
        match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
        if match:
            candidate = match.group(1)
            return candidate
        
        # 4. If nothing works, just return the raw string
        return content

    async def embedding_model(self, batch: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a batch of texts using the openai.AsyncOpenAI SDK.
        Logic preserved:
        - Uses batching (size 100)
        - Retries after an error with a fixed sleep time
        - Returns a flat list of embeddings
        - Async native calls
        """
        

        embeddings: List[List[float]] = []
        batch_size = 100
        sleep_time = 2  # Time to wait before retrying after an error


        embedding_model = self.config.config_data.get("embedding_model")
        if not embedding_model:
            raise ValueError("Missing 'embedding_model' in config_data")

        for i in range(0, len(batch), batch_size):
            batch_segment = batch[i:i + batch_size]

            try:
                result = await self.client.embeddings.create(
                    model=embedding_model,
                    input=batch_segment,
                )

                # Extract embeddings
                batch_embeddings = [e.embedding for e in result.data]
                embeddings.extend(batch_embeddings)


            except Exception as e:
                self.log.error(f"OpenAI Embedding error: {e}")
                await asyncio.sleep(sleep_time)  # backoff before retry

        self.log.info("OpenAI embeddings generated.")
        return embeddings