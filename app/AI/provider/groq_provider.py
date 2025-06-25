import httpx
from typing import Dict,List 
from app.config import GROQ_API_KEY
from app.AI.provider._base_provider import LLMProvider 
 
# Groq Provider
class GroqProvider(LLMProvider):
    async def get_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Sends a request to the Groq API and returns the generated response.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
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
