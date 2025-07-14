import asyncio
import httpx
from typing import Dict,List 
import json
from app.config import GEMINI_API_KEY
from app.AI.provider._base_provider import LLMProvider 

# Gemini Provider
class GeminiProvider(LLMProvider):
    async def get_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Sends a request to the Gemini REST API and returns the generated response.
        """
        # print("gemini model ",self.config.config_data.get("model"))
        headers = {
            "Content-Type": "application/json",
        }

        # Convert messages to the format expected by Gemini
        content_parts = []
        for message in messages:
            # print("message " ,message)
            role = message.get("role", "user")
            text = message.get("content", "")
            content_parts.append({
                "role": role,
                "parts": [{"text": text}]
            })

        payload = {
            "contents": content_parts,
            "generationConfig": {
                "temperature": self.config.config_data.get("temperature", 0.7),
                "maxOutputTokens": self.config.config_data.get("max_tokens", 1024),
                "topP": self.config.config_data.get("top_p", 1),
                "stopSequences": self.config.config_data.get("stop", [])
            }
        }

        endpoint = f"{self.config.base_url}/{self.config.config_data.get('model')}?key={GEMINI_API_KEY}"
        # print("endpoint ", endpoint)
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(endpoint, headers=headers, json=payload)
                # print("response ",response)
                # print("Status:", response.status_code)
                # print("Response text:", response.text)
                response.raise_for_status()
                data = response.json()
                content= data["candidates"][0]["content"]["parts"][0]["text"]
                json_content = self._extract_json_from_codeblock(content)
                try:
                    return json.loads(json_content)
                except json.JSONDecodeError:
                    return json_content
        except Exception as e:
            return f"Gemini API error: {str(e)}"
        
    def _extract_json_from_codeblock(self, content: str) -> str:
        start = content.find("```json")
        end = content.rfind("```")
        if start != -1 and end != -1:
            json_content = content[start + 7:end].strip()
            return json_content
        else:
            return content
        
    async def gemini_embedding_model(self, batch: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a batch of texts using the Gemini REST API.
        """
        embeddings = []
        batch_size = 100
        sleep_time = 5  # Time to wait before retrying after an error

        headers = {
            "Content-Type": "application/json",
        }

        embedding_model = self.config.config_data.get('embedding_model')
        if embedding_model.startswith('models/'):
            embedding_model = embedding_model[len('models/'):]  # Strip prefix

        endpoint = f"{self.config.base_url}/{embedding_model}:batchEmbedContents?key={GEMINI_API_KEY}"

        for i in range(0, len(batch), batch_size):
            batch_segment = batch[i:i + batch_size]
            # print(f"Embedding batch {i // batch_size + 1} of {(len(batch) - 1) // batch_size + 1}")

            # Fix: Include model in each request
            requests = [
                {
                    "model": f"models/{embedding_model}",
                    "content": {"parts": [{"text": text}]}
                }
                for text in batch_segment
            ]
            payload = {"requests": requests}

            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(endpoint, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    batch_embeddings = [item['values'] for item in data['embeddings']]
                    embeddings.extend(batch_embeddings)
            except httpx.HTTPStatusError as e:
                # print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
                # print(f"Retrying in {sleep_time} seconds...")
                await asyncio.sleep(sleep_time)
            except Exception as e:
                # print(f"An unexpected error occurred: {e}")
                # print(f"Retrying in {sleep_time} seconds...")
                await asyncio.sleep(sleep_time)

        return embeddings
