from typing import List
from app.AI.provider._base_provider import LLMProvider

class E5_Model(LLMProvider):

    def __init__(self, model_config):
        super().__init__(model_config)
    
    async def get_response(self, messages):
        return await super().get_response(messages)
    
    async def embedding_model(self, batch: List[str])-> List[List[float]]:
        #TODO: Implement embedding logic with the finetuned/ basemodel here.
        pass