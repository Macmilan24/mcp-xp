import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
from sys import path
path.append(".")

from app.bioblend_server.informer.manager import InformerManager
from app.bioblend_server.informer.indexer import QdrantIndexer, RedisIndexer

# --- Manager Tests ---
class TestInformerManager:
    @pytest.fixture
    def manager(self):
        with patch("app.bioblend_server.informer.manager.QdrantClient"), \
             patch("app.bioblend_server.informer.manager.LLMResponse") as mock_llm:
            
            man = InformerManager()
            man.client = MagicMock()
            man.embedder = mock_llm()
            # Mock embeddings return
            man.embedder.get_embeddings = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
            man.embedder.embedding_size = 2
            return man

    @pytest.mark.asyncio
    async def test_embed_and_store_entities(self, manager):
        """Test the full pipeline: Data -> DataFrame -> Embed -> Qdrant Upsert."""
        entities = [
            {"name": "Tool A", "content": "desc A", "id": 1},
            {"name": "Tool B", "content": "desc B", "id": 2}
        ]
        
        await manager.embed_and_store_entities(entities, "test_collection")
        
        # Verify Embeddings were generated for content
        manager.embedder.get_embeddings.assert_called_once()
        
        # Verify Upsert was called
        manager.client.upsert.assert_called_once()
        call_args = manager.client.upsert.call_args
        
        # Verify payload structure
        assert call_args.kwargs['collection_name'] == "test_collection"
        batch = call_args.kwargs['points']
        assert len(batch.ids) == 2
        assert len(batch.vectors) == 2

    def test_search_by_vector_parsing(self, manager):
        """
        Test that Qdrant points are correctly converted to dicts.
        This validates the 'Retrieval' aspect of Qdrant functionalities.
        """
        # Mock Qdrant point result object
        mock_point = MagicMock()
        mock_point.score = 0.95
        mock_point.payload = {
            "tool_id": "t1", 
            "name": "Blast", 
            "description": "Aligns stuff",
            "content": "raw content"
        }
        
        # Setup client mock return
        manager.client.query_points.return_value.points = [mock_point]
        
        # Run search
        results = manager.search_by_vector(
            collection="test_coll",
            query_vector=[0.1, 0.2],
            entity_type="tool"
        )
        
        # Verify parsing logic
        assert len(results) == 1
        assert results[0]['tool_id'] == "t1"
        assert results[0]['score'] == 0.95
        assert results[0]['description'] == "Aligns stuff"
        assert results[0]['name'] == "Blast"

# --- Indexer Tests ---
class TestIndexers:
    @pytest.mark.asyncio
    async def test_redis_indexer(self):
        """Test Redis Indexer wrapper."""
        mock_cache = MagicMock()
        indexer = RedisIndexer(mock_cache)
        
        entities = [{"id": 1}]
        await indexer.index_entities(entities, "coll", 300)
        
        mock_cache.set_entities.assert_called_once_with("coll", entities, 300)

    @pytest.mark.asyncio
    async def test_qdrant_indexer(self):
        """Test Qdrant Indexer wrapper."""
        mock_manager = AsyncMock()
        indexer = QdrantIndexer(mock_manager)
        
        entities = [{"id": 1}]
        # Mock that collection exists so it deletes first
        mock_manager.client.collection_exists.return_value = True
        
        await indexer.index_entities(entities, "coll")
        
        mock_manager.delete_collection.assert_called_with("coll")
        mock_manager.embed_and_store_entities.assert_called_with(
            entities=entities, collection_name="coll"
        )