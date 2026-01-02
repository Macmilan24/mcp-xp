import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sys import path
path.append(".")

from app.galaxy import GalaxyClient
from app.bioblend_server.informer.informer import GalaxyInformer

# --- Fixtures ---

@pytest.fixture
def mock_galaxy_client():
    """Fixture for a mocked GalaxyClient."""
    mock = MagicMock(spec=GalaxyClient)
    mock.whoami = "test_user"
    mock.gi_client = MagicMock()
    mock.gi_admin = MagicMock()
    return mock

@pytest.fixture
def mock_informer_manager():
    """Fixture for a mocked InformerManager."""
    mock = AsyncMock()
    mock.create = AsyncMock(return_value=mock)
    return mock

@pytest.fixture
def mock_redis_cache():
    """Fixture for a mocked RedisCache."""
    return MagicMock()

@pytest.fixture
def mock_redis_indexer():
    """Fixture for a mocked RedisIndexer."""
    return MagicMock()

@pytest.fixture
def mock_qdrant_indexer():
    """Fixture for a mocked QdrantIndexer."""
    return MagicMock()

@pytest.fixture
def mock_search_engine():
    """Fixture for a mocked SearchEngine."""
    return MagicMock()

@pytest.fixture
def mock_reranker():
    """Fixture for a mocked InformerReranker."""
    return MagicMock()

@pytest.fixture
def mock_llm_response():
    """Fixture for a mocked LLMResponse."""
    mock = MagicMock()
    mock.get_response = AsyncMock(return_value="Mocked LLM Response")
    mock.get_embeddings = AsyncMock(return_value=[0.1, 0.2])
    return mock


@pytest.fixture
async def galaxy_informer(
    mock_galaxy_client,
    mock_informer_manager,
    mock_redis_cache,
    mock_redis_indexer,
    mock_qdrant_indexer,
    mock_search_engine,
    mock_reranker,
    mock_llm_response,
):
    """Fixture to create a GalaxyInformer instance with mocked dependencies."""
    with patch("app.bioblend_server.informer.manager.InformerManager", new=mock_informer_manager), \
         patch("app.bioblend_server.informer.informer.RedisCache", return_value=mock_redis_cache), \
         patch("app.bioblend_server.informer.informer.RedisIndexer", return_value=mock_redis_indexer), \
         patch("app.bioblend_server.informer.informer.QdrantIndexer", return_value=mock_qdrant_indexer), \
         patch("app.bioblend_server.informer.informer.SearchEngine", return_value=mock_search_engine), \
         patch("app.bioblend_server.informer.informer.InformerReranker", return_value=mock_reranker), \
         patch("app.bioblend_server.informer.informer.LLMResponse", return_value=mock_llm_response):
        
        # Initialize as 'tool' by default
        informer = await GalaxyInformer.create(galaxy_client=mock_galaxy_client, entity_type="tool")
        return informer


# --- Test Classes ---

class TestGalaxyInformerCreation:
    """Tests for the creation of a GalaxyInformer instance."""

    @pytest.mark.asyncio
    async def test_galaxy_informer_creation(self, galaxy_informer):
        """Test that GalaxyInformer can be created successfully."""
        assert galaxy_informer is not None
        assert galaxy_informer.entity_type == "tool"
        assert galaxy_informer.username == "test_user"
        assert galaxy_informer.manager is not None
        assert galaxy_informer.cache is not None
        assert galaxy_informer.redis_indexer is not None
        assert galaxy_informer.qdrant_indexer is not None
        assert galaxy_informer.search_engine is not None
        assert galaxy_informer.reranker is not None


class TestGetAllEntities:
    """Tests for the get_all_entities method."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("entity_type, mock_method_name", [
        ("tool", "get_tools"),
        ("workflow", "get_workflows"),
        ("dataset", "get_datasets"),
    ])
    async def test_get_all_entities(self, galaxy_informer, entity_type, mock_method_name):
        """Test that get_all_entities calls the correct data_provider method."""
        # Arrange
        galaxy_informer.entity_type = entity_type
        mock_entity_data = ([{"id": "1", "name": f"test_{entity_type}"}], ["test_name"])
        
        # Mock the specific method on the data_provider
        mock_method = MagicMock(return_value=mock_entity_data)
        setattr(galaxy_informer.data_provider, mock_method_name, mock_method)

        # Update the config to point to the new mock method
        galaxy_informer._entity_config[entity_type]['get_method'] = mock_method

        # Act
        result = galaxy_informer.get_all_entities()

        # Assert
        mock_method.assert_called_once()
        assert result == mock_entity_data


class TestGetCachedOrFreshEntities:
    """Tests for the get_cached_or_fresh_entities method."""

    @pytest.mark.asyncio
    async def test_get_cached_entities_hit(self, galaxy_informer):
        """Test cache hit scenario."""
        # Arrange
        cached_entities = [{"id": "1", "name": "cached_tool"}]
        galaxy_informer.cache.get_entities.return_value = cached_entities
        galaxy_informer.refresh_and_cache_entities = AsyncMock()
        galaxy_informer.search_engine.get_collection_name.return_value = ("collection", "corpus")

        # Act
        result = await galaxy_informer.get_cached_or_fresh_entities()

        # Assert
        galaxy_informer.cache.get_entities.assert_called_once()
        galaxy_informer.refresh_and_cache_entities.assert_not_called()
        assert result == cached_entities

    @pytest.mark.asyncio
    async def test_get_fresh_entities_miss(self, galaxy_informer):
        """Test cache miss scenario."""
        # Arrange
        fresh_entities = [{"id": "2", "name": "fresh_tool"}]
        galaxy_informer.cache.get_entities.return_value = None
        galaxy_informer.refresh_and_cache_entities = AsyncMock(return_value=fresh_entities)
        galaxy_informer.search_engine.get_collection_name.return_value = ("collection", "corpus")

        # Act
        result = await galaxy_informer.get_cached_or_fresh_entities()

        # Assert
        galaxy_informer.cache.get_entities.assert_called_once()
        galaxy_informer.refresh_and_cache_entities.assert_called_once()
        assert result == fresh_entities


class TestRefreshAndCacheEntities:
    """Tests for the refresh_and_cache_entities method."""

    @pytest.mark.asyncio
    async def test_refresh_and_cache_entities_calls_indexers(self, galaxy_informer):
        """Test that refresh_and_cache_entities calls the correct indexer methods."""
        # Arrange
        entities = [{"id": "1", "name": "test_entity"}]
        name_corpus = ["test_entity"]
        galaxy_informer.get_all_entities = MagicMock(return_value=(entities, name_corpus))
        
        galaxy_informer.redis_indexer.index_entities = AsyncMock()
        galaxy_informer.qdrant_indexer.index_entities = AsyncMock()
        galaxy_informer.search_engine.get_collection_name.return_value = ("collection", "corpus")

        # Act
        await galaxy_informer.refresh_and_cache_entities()

        # Assert
        galaxy_informer.redis_indexer.index_entities.assert_any_call(
            entities=name_corpus,
            collection_name="corpus",
            ttl=galaxy_informer._entity_config[galaxy_informer.entity_type]['ttl']
        )
    
        galaxy_informer.redis_indexer.index_entities.assert_any_call(
            entities=entities,
            collection_name="collection",
            ttl=galaxy_informer._entity_config[galaxy_informer.entity_type]['ttl']
        )
        galaxy_informer.qdrant_indexer.index_entities.assert_called_once_with(
            entities=entities,
            collection_name="collection"
        )


class TestSearchEntities:
    """Tests for the search_entities method."""

    @pytest.mark.asyncio
    async def test_search_entities_flow(self, galaxy_informer):
        """Test the full flow of the search_entities method."""
        # Arrange
        query = "test query"
        entities = [{"tool_id": "1", "name": "test_entity"}]
        keywords = ["test", "query"]
        fuzzy_results = [{"tool_id": "1", "name": "test_entity"}]
        semantic_results = [{"tool_id": "1", "name": "test_entity"}]
        reranked_results = [{"tool_id": "1", "name": "test_entity"}]

        galaxy_informer.get_cached_or_fresh_entities = AsyncMock(return_value=entities)
        galaxy_informer.search_engine.extract_keywords.return_value = keywords
        galaxy_informer.search_engine.fuzzy_search = AsyncMock(return_value=fuzzy_results)
        galaxy_informer.search_engine.semantic_search = AsyncMock(return_value=semantic_results)
        galaxy_informer.reranker.rerank_results = AsyncMock(return_value=reranked_results)
        
        # Act
        result = await galaxy_informer.search_entities(query)

        # Assert
        galaxy_informer.get_cached_or_fresh_entities.assert_called_once()
        galaxy_informer.search_engine.extract_keywords.assert_called_once_with(query)
        assert galaxy_informer.search_engine.fuzzy_search.call_count == len(keywords)
        galaxy_informer.search_engine.semantic_search.assert_called_once_with(query, entities)
        galaxy_informer.reranker.rerank_results.assert_called_once_with(
            query=query,
            fuzzy_results=fuzzy_results,
            semantic_results=semantic_results,
            entity_type=galaxy_informer.entity_type
        )
        assert result == reranked_results


class TestGetEntityDetails:
    """Tests for the get_entity_details method."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("entity_type, method_name, is_async", [
        ("dataset", "show_dataset", False),
        ("tool", "show_tool", True),
        ("workflow", "show_workflow", False),
    ])
    async def test_get_entity_details_calls_correct_method(self, galaxy_informer, entity_type, method_name, is_async):
        """Test that get_entity_details calls the correct method on the data_provider."""
        # Arrange
        entity_id = "test_id"
        expected_details = {"id": entity_id, "name": "details"}
        galaxy_informer.entity_type = entity_type

        if is_async:
            mock_method = AsyncMock(return_value=expected_details)
        else:
            mock_method = MagicMock(return_value=expected_details)
        
        setattr(galaxy_informer.data_provider, method_name, mock_method)

        # Act
        result = await galaxy_informer.get_entity_details(entity_id)

        # Assert
        mock_method.assert_called_once_with(entity_id)
        assert result == expected_details


class TestGenerateFinalResponse:
    """Tests for the generate_final_response method."""

    @pytest.mark.asyncio
    async def test_summarizes_and_caches_content(self, galaxy_informer):
        """Test that content is summarized and the summary is cached."""
        # Arrange
        galaxy_informer.entity_type = "tool"
        galaxy_informer.username = "test_user"
        query = "test query"
        retrieved_content = [{"tool_id": "tool1", "content": "some tool content"}]
        summary = "This is a summary."

        galaxy_informer.llm_response.get_response = AsyncMock(return_value=summary)

        # Act
        await galaxy_informer.generate_final_response(query, retrieved_contents=retrieved_content)

        # Assert
        assert galaxy_informer.llm_response.get_response.call_count == 2 # Once for summary, once for final response
        
        # Check cache call
        galaxy_informer.cache.set_string.assert_called_once()
        cache_args, cache_kwargs = galaxy_informer.cache.set_string.call_args
        assert cache_kwargs['key'] == "test_user_tool_tool1"
        assert cache_kwargs['value'] == summary
        assert 'ttl' in cache_kwargs


class TestGetEntityInfo:
    """
    Tests for the main orchestration method: get_entity_info.
    This tests the integration logic between search, cache, details, and response generation.
    """

    @pytest.mark.asyncio
    async def test_direct_id_lookup_success(self, galaxy_informer):
        """Test that providing a valid entity_id bypasses search and uses that entity directly."""
        # Arrange
        entity_id = "tool_123"
        query = "irrelevant query"
        galaxy_informer.entity_type = "tool"
        
        # Mock entities existing in the system
        galaxy_informer.get_cached_or_fresh_entities = AsyncMock(return_value=[
            {"tool_id": "tool_123", "name": "Target Tool", "content": "raw content"}
        ])
        
        # Mocks
        galaxy_informer.search_entities = AsyncMock()

        galaxy_informer.cache.get_string.return_value = None 
        galaxy_informer.get_entity_details = AsyncMock(return_value={"detailed": "info"})
        galaxy_informer.generate_final_response = AsyncMock(return_value="Final Answer")

        # Act
        result = await galaxy_informer.get_entity_info(query, entity_id=entity_id)

        # Assert
        galaxy_informer.get_cached_or_fresh_entities.assert_called_once()
        galaxy_informer.search_entities.assert_not_called()
        galaxy_informer.get_entity_details.assert_called_once_with(entity_id) 
        assert result == "Final Answer"

    @pytest.mark.asyncio
    async def test_direct_id_lookup_invalid_falls_back_to_search(self, galaxy_informer):
        """Test that an invalid ID (not in entities list) triggers a fallback to search."""
        # Arrange
        entity_id = "fake_id"
        query = "actual query"
        galaxy_informer.entity_type = "tool"
        
        # Mock entities list that DOES NOT contain fake_id
        galaxy_informer.get_cached_or_fresh_entities = AsyncMock(return_value=[
            {"tool_id": "other_id", "name": "Other"}
        ])
        
        # Mocks
        galaxy_informer.search_entities = AsyncMock(return_value=[]) # Search is triggered
        galaxy_informer.generate_final_response = AsyncMock(return_value="No results")

        # Act
        await galaxy_informer.get_entity_info(query, entity_id=entity_id)

        # Assert
        galaxy_informer.search_entities.assert_called_once_with(query=query)

    @pytest.mark.asyncio
    async def test_processing_separates_user_and_global_results(self, galaxy_informer):
        """
        Test that:
        1. 'user_instance' items trigger get_entity_details.
        2. 'global' items are collected separately and do NOT trigger get_entity_details.
        """
        # Arrange
        query = "search"
        galaxy_informer.entity_type = "tool"
        galaxy_informer.username = "test_user"
        
        # Mock Search Results
        search_results = [
            {"tool_id": "local_1", "source": "user_instance", "content": "local content"},
            {"tool_id": "global_1", "source": "global", "content": "global content"}
        ]
        
        galaxy_informer.search_entities = AsyncMock(return_value=search_results)
        galaxy_informer.cache.get_string.return_value = None # No cached summary
        galaxy_informer.get_entity_details = AsyncMock(return_value={"details": "fetched details"})
        galaxy_informer.generate_final_response = AsyncMock(return_value="Done")

        # Act
        await galaxy_informer.get_entity_info(query)

        # Assert
        galaxy_informer.get_entity_details.assert_called_once_with("local_1")
        
        # Verify generate_final_response received the correct mix
        galaxy_informer.generate_final_response.assert_called_once()
        call_kwargs = galaxy_informer.generate_final_response.call_args.kwargs

        assert "global content" in call_kwargs['global_content']
        # Retrieved contents 
        assert call_kwargs['retrieved_contents'][0] == {"details": "fetched details"}

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_uses_cached_summary_if_available(self, galaxy_informer):
        """
        Test that if a summary is in Redis.

        """
        # Arrange
        galaxy_informer.entity_type = "tool"
        galaxy_informer.username = "u"
        
        # Search returns ONE item
        search_results = [{"tool_id": "t1", "source": "user_instance", "content": "..."}]
        galaxy_informer.search_entities = AsyncMock(return_value=search_results)
        
        # Cache Hit
        galaxy_informer.cache.get_string.return_value = "Cached Summary Text"
        galaxy_informer.get_entity_details = AsyncMock() 

        # Act
        await galaxy_informer.get_entity_info("query")

        # Assert
        galaxy_informer.get_entity_details.assert_not_called()
        
        # Verify the LLM received the cached summary
        final_call_args = galaxy_informer.llm_response.get_response.call_args_list[-1][0][0]
        assert "Cached Summary Text" in str(final_call_args)

    @pytest.mark.asyncio
    async def test_no_results_found(self, galaxy_informer):
        """Test behavior when search returns None or empty list."""
        # Arrange
        galaxy_informer.search_entities = AsyncMock(return_value=None)
        galaxy_informer.generate_final_response = AsyncMock(return_value="Not Found")

        # Act
        await galaxy_informer.get_entity_info("query")

        # Assert
        galaxy_informer.generate_final_response.assert_called_once()
        call_args = galaxy_informer.generate_final_response.call_args
        
        # retrieved_contents is the 2nd argument
        arg_2 = call_args[0][1] 
        assert arg_2 == [{"message": "No relevant items found."}]