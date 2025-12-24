import os
import asyncio
import traceback
import pandas as pd
import numpy as np
import logging
import random
import json
import time
import tqdm

from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import httpx

from app.AI.provider.gemini_provider import GeminiProvider
from app.AI.llm_config._base_config import LLMModelConfig
from app.log_setup import configure_logging




load_dotenv()

class InformerManager:
    """
    Manages all vector database operations for the GalaxyInformer, including
    data preparation, embedding generation, storage, and retrieval from Qdrant.
    """
    _logging_configured = False

    def __init__(self):
        # This init should be lightweight and non-blocking.
        self.client = None
        self.llm = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.embedding_size = 384  # sentence-transformers/all-MiniLM-L6-v2 uses 384 dimensions
        self.hf_token = os.getenv("HUGGING_FACE_TOKEN")
        self.http_client = None

    @classmethod
    async def create(cls):
        """Asynchronous factory to create and initialize an InformerManager instance."""
        self = cls()
        load_dotenv()
        configure_logging()
        try:
            self.logger.info("Initializing Qdrant client...")
            self.client = QdrantClient(os.environ.get('QDRANT_CLIENT', 'http://localhost:6333'))
            
            # Use a persistent httpx client for connection pooling
            headers = {}
            if self.hf_token:
                headers["Authorization"] = f"Bearer {self.hf_token}"
            http_timeout = float(os.getenv("HF_HTTP_TIMEOUT", "120"))
            self.http_client = httpx.AsyncClient(
                headers=headers,
                timeout=http_timeout
            )
            self.logger.info("Hugging Face Inference API client configured.")
            self.logger.info("InformerManager connected to Qdrant successfully.")
        except Exception as e:
            self.logger.exception(f"InformerManager initialization failed: {e}")
            if self.http_client:
                await self.http_client.aclose()
            raise
        
        return self
   
    async def get_embedding_model(self, input):
        return await self.llm.gemini_embedding_model(input)
    
    def _ensure_collection_exists(self, collection_name: str):
        """
        A private helper to ensure a collection exists in Qdrant, creating it if necessary.
        """
        try:
            self.client.get_collection(collection_name)
        except Exception:
            self.logger.warning(f"Collection '{collection_name}' not found. Creating it now.")
            try:
                self.client.create_collection(
                    collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_size,
                        distance=models.Distance.COSINE
                    )
                )
                self.logger.info(f"Collection '{collection_name}' created successfully.")
            except Exception as e:
                self.logger.error(f"Failed to create collection '{collection_name}': {e}")
                traceback.print_exc()

    def _prepare_dataframe(self, entities: list[dict]) -> pd.DataFrame:
        """
        A private helper to convert the list of entity dictionaries into a pandas DataFrame.
        """
        if isinstance(entities, list) and all(isinstance(d, dict) for d in entities):
            return pd.DataFrame(entities)
        else:
            self.logger.error("Invalid data format. Expected a list of dictionaries.")
            raise ValueError("Input data must be a list of dictionaries.")

    async def _post_with_retries(self, url: str, json_payload: dict, max_retries: int = 5) -> httpx.Response:
        """
        Helper to POST with retries on 429/5xx using exponential backoff.
        """
        backoff = 1.0
        for attempt in range(max_retries):
            try:
                resp = await self.http_client.post(url, json=json_payload)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError("retryable", request=resp.request, response=resp)
                resp.raise_for_status()
                return resp
            except httpx.HTTPStatusError as e:
                code = e.response.status_code if e.response else None
                if code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    self.logger.warning(f"HF API {code}. Retrying in {backoff:.1f}s (attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                    continue
                self.logger.error(f"HF API request failed with status {code}: {e.response.text if e.response else e}")
                raise
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"HF API error: {e}. Retrying in {backoff:.1f}s (attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                    continue
                self.logger.error(f"HF API request failed after retries: {e}")
                raise

    async def _generate_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        A private helper to generate vector embeddings for the 'content' column of a DataFrame.
        """
        self.logger.info("Generating vector embeddings using Hugging Face Inference API (BAAI/bge-small-en-v1.5).")
        api_url = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5"

        texts_to_embed = df['content'].astype(str).tolist()
        all_embeddings = []
        batch_size = int(os.getenv("HF_EMBED_BATCH", "64"))

        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:i + batch_size]
            payload = {
                "inputs": batch,
                "options": {"wait_for_model": True}
            }
            
            try:
                response = await self._post_with_retries(api_url, payload)
                batch_embeddings = response.json()
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                self.logger.error(f"Embedding batch {i//batch_size + 1} failed: {e}")
                raise

        df['dense'] = all_embeddings
        self.logger.info("Embeddings generated successfully.")
        return df
    
    async def embed_query(self, query: str) -> list[float]:
        """
        Generates an embedding for a single query string using the HF Inference API.
        """
        self.logger.info(f"Embedding single query: '{query}'")
        api_url = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5"

        payload = {
            "inputs": [str(query)],
            "options": {"wait_for_model": True}
        }
        
        try:
            response = await self._post_with_retries(api_url, payload)
            embedding = response.json()[0]
            return embedding
        except Exception as e:
            self.logger.error(f"Query embedding failed: {e}")
            raise
        
    def _upsert_points(self, collection_name: str, df: pd.DataFrame):
        """
        A private helper to perform the final upsert operation into the Qdrant collection.
        """
        try:
            # Define which columns from the DataFrame become the payload
            excluded_columns = {"dense"}
            payload_columns = [col for col in df.columns if col not in excluded_columns]
            payloads_list = [
                {col: getattr(item, col) for col in payload_columns}
                for item in df.itertuples(index=False)
            ]

            if 'id' not in df.columns:
                df['id'] = [random.randint(100000, 999999) for _ in range(len(df))]

            self._ensure_collection_exists(collection_name)
            
            batch_size = int(os.getenv("QDRANT_UPSERT_BATCH", "200"))
            total = len(df)
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                ids_batch = df['id'].tolist()[start:end]
                vectors_batch = df['dense'].tolist()[start:end]
                payloads_batch = payloads_list[start:end]

                self.client.upsert(
                    collection_name=collection_name,
                    points=models.Batch(
                        ids=ids_batch,
                        vectors=vectors_batch,
                        payloads=payloads_batch,
                    ),
                )

            self.logger.info(f"Successfully upserted {len(df)} points to '{collection_name}' in batches of {batch_size}.")
            return "Data Successfully Uploaded"
        except Exception as e:
            self.logger.error(f"Error upserting data to Qdrant: {e}")
            traceback.print_exc()

    async def embed_and_store_entities(self, entities: list[dict], collection_name: str):
        """
        Processes and saves Galaxy entities to a specified Qdrant collection.
        This is the main public method for saving data.

        Called in: GalaxyInformer.retrive_informer_data
        """
        try:
            df = self._prepare_dataframe(entities)
            df_embedded = await self._generate_embeddings(df)
            
            if df_embedded is None or 'dense' not in df_embedded.columns or df_embedded.empty:
                self.logger.error("Embedding failed, aborting Qdrant upsert.")
                return None 

            return self._upsert_points(collection_name, df_embedded)
        except Exception as e:
            self.logger.error(f"Failed to embed and store entities in '{collection_name}': {e}")
            
            return None

    def search_by_vector(self, collection: str, query_vector: list, entity_type: str) -> dict:
        """
        Performs a semantic search in a Qdrant collection based on a query vector.

        Called in: GalaxyInformer.semantic_search
        """

        try:
            result = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                with_payload=True,
                score_threshold=0.3,
                limit=10
            )
            
            response = {}
            if entity_type == "tool":

                for i,point in enumerate(result):
                    response[i]={
                        "id": point.id,
                        "score": point.score,
                        "description": point.payload.get('description', 'description not available'),
                        "tool_id": point.payload.get('tool_id'),
                        "name": point.payload.get('name'),
                        "available_in_instance": point.payload.get('available_in_instance', False)

                    }
            elif entity_type == "workflow":
                for i,point in enumerate(result):
                    response[i]={
                        "id": point.id,
                        "score": point.score,
                        "model_class": point.payload.get('model_class', 'unknown'),
                        'description': point.payload.get('description', 'unkown'),
                        "owner": point.payload.get('owner', 'unknown'),
                        "workflow_id": point.payload.get('workflow_id'),
                        "name": point.payload.get('name'),
                        "available_in_instance": point.payload.get('available_in_instance', False)
                    }
            elif entity_type == 'dataset':
                for i, point in enumerate(result):
                    response[i]={
                        'id': point.id,
                        'score': point.score,
                        "dataset_id": point.payload.get('dataset_id'),
                        "name": point.payload.get('name'),
                        "full_path": point.payload.get('full_path', 'unknown'),
                        "type": point.payload.get('type', 'unknown'),
                        "source": point.payload.get('source')
                    }
                
            return response
        except Exception as e:
            self.logger.error(f"Error searching collection '{collection}': {e}")
            traceback.print_exc()
            return {"error": "Search failed"}

    def delete_collection(self, collection_name: str):
        """
        Deletes a Qdrant collection if it exists.

        Called in: GalaxyInformer.retrive_informer_data
        """
        try:
            self.client.delete_collection(collection_name)
            self.logger.info(f"Successfully deleted collection: {collection_name}")
        except Exception as e:
            # Qdrant client might raise an exception if the collection doesn't exist.
            # We can log this as info or a warning instead of an error.
            self.logger.warning(f"Could not delete collection '{collection_name}'. It might not exist. Details: {e}")