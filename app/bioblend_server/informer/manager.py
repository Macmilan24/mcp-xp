import os
import traceback
import pandas as pd
import numpy as np
import logging
import random
import json

from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

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
        self.embedding_size = 768  # Gemini embedding vector size

    @classmethod
    async def create(cls):
        """Asynchronous factory to create and initialize an InformerManager instance."""
        self = cls()
        load_dotenv()
        configure_logging()
        try:
            # Initialize Qdrant client asynchronously if possible, or run in executor
            self.client = QdrantClient(os.getenv('QDRANT_CLIENT'))
            self.logger.info("Qdrant Client initialized")

            with open('app/AI/llm_config/llm_config.json', 'r') as f:
                model = json.load(f)
            gemini_cfg = LLMModelConfig(model['providers']['gemini'])
            self.llm = GeminiProvider(model_config=gemini_cfg)
            
            self.logger.info("InformerManager connected to Qdrant successfully.")
        except Exception as e:
            self.logger.exception(f"Qdrant connection failed: {e}")
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

    async def _generate_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        A private helper to generate vector embeddings for the 'content' column of a DataFrame.
        """
        try:
            self.logger.info("Generating vector embeddings for entity content.")
            embeddings = await self.get_embedding_model(df['content'].tolist())
            embed_array = np.array(embeddings).reshape(-1, self.embedding_size)
            df['dense'] = embed_array.tolist()
            self.logger.info("Embeddings generated successfully.")
            return df
        except Exception as e:
            self.logger.error(f"Error generating dense embeddings: {e}")
            traceback.print_exc()
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
            
            self.client.upsert(
                collection_name=collection_name,
                points=models.Batch(
                    ids=df['id'].tolist(),
                    vectors=df["dense"].tolist(),
                    payloads=payloads_list,
                ),
            )
            self.logger.info(f"Successfully upserted {len(df)} points to '{collection_name}'.")
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
            if df_embedded is not None:
                return self._upsert_points(collection_name, df_embedded)
        except Exception as e:
            self.logger.error(f"Failed to embed and store entities in '{collection_name}': {e}")
            traceback.print_exc()

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
                        "name": point.payload.get('name')

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
                        "name": point.payload.get('name')
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