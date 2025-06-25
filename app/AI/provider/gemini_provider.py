import httpx
from typing import Dict,List 
from app.config import GEMINI_API_KEY
from app.AI.provider._base_provider import LLMProvider 

# Gemini Provider
class GeminiProvider(LLMProvider):
    async def get_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Sends a request to the Gemini REST API and returns the generated response.
        """
        print("gemini model ",self.config.config_data.get("model"))
        headers = {
            "Content-Type": "application/json",
        }

        # Convert messages to the format expected by Gemini
        content_parts = []
        for message in messages:
            print("message " ,message)
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
        print("endpoint ", endpoint)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(endpoint, headers=headers, json=payload)
                print("response ",response)
                print("Status:", response.status_code)
                print("Response text:", response.text)
                response.raise_for_status()
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return f"Gemini API error: {str(e)}"
