import os
import logging
from typing import List
import asyncio
import yaml
import torch
from sentence_transformers import SentenceTransformer

from sys import path
path.append(".")

from app.AI.provider._base_provider import LLMProvider

class HuggingFaceModel(LLMProvider):

    def __init__(self, model_config):
        super().__init__(model_config)
        self.log = logging.getLogger(__class__.__name__)
        
        config_file = "config.yml"
        if not os.path.exists(config_file):
            self.log.error(f"Configuration file {config_file} not found.")
            raise FileNotFoundError(f"Configuration file {config_file} not found.")
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)
        model_name = self.config["agent"]["finetuned_model"]
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.log.info(f"Loading SentenceTransformer on device: {device}")

        self.model = SentenceTransformer(model_name, device=device)
        # Load the SentenceTransformer model
    
    def get_response(self, messages):
        return super().get_response(messages)
    

    async def embedding_model(self, batch: List[str]) -> List[List[float]]:
        """ Generates embeddings for a batch of texts using a local SentenceTransformer model. Returns a flat list of embeddings """

        embeddings: List[List[float]] = []
        batch_size = 100
        sleep_time = 2  # seconds to wait before retry

        for i in range(0, len(batch), batch_size):
            batch_segment = batch[i:i + batch_size]

            for attempt in range(3):  # up to 3 retries
                try:
                    loop = asyncio.get_event_loop()
                    # Run encode in a background thread so we don't block the event loop
                    result = await loop.run_in_executor(
                        None,
                        lambda: self.model.encode(
                            batch_segment,
                            convert_to_numpy=True,
                            show_progress_bar=False
                        ).tolist()
                    )
                    embeddings.extend(result)
                    break  # success, move to next batch

                except Exception as e:
                    self.log.error(
                        f"SentenceTransformer encode error (attempt {attempt + 1}/3): {e}"
                    )
                    await asyncio.sleep(sleep_time)

            else:
                # If all retries fail, log and continue
                self.log.warning(f"Failed to encode batch segment starting at index {i} after 3 attempts.")

        self.log.info("SentenceTransformer embeddings generated.")
        return embeddings