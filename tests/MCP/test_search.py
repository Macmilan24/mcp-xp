import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sys import path
path.append(".")

from app.bioblend_server.informer.search.fuzzy_searcher import FuzzySearcher
from app.bioblend_server.informer.search.semantic_searcher import SemanticSearcher
from app.bioblend_server.informer.search import SearchEngine

## Global mock for LLMResponse
@pytest.fixture(autouse=True)
def mock_llm_response():
    """
    Automatically mock LLMResponse for ALL tests in this file.
    """
    with patch("app.bioblend_server.informer.search.semantic_searcher.LLMResponse") as mock:
        yield mock

## Fuzzy Search tests
class TestFuzzySearcher:
    def test_priority_field_match(self):
        """Test that matches in 'name' (priority) beat matches in 'description'."""
        searcher = FuzzySearcher(threshold=10)
        entities = [
            {"name": "Exact Target", "description": "irrelevant", "id": "1"},
            {"name": "Other", "description": "Exact Target in description", "id": "2"}
        ]
        
        # Only search 'name' field
        results = searcher.search("Exact Target", entities, search_fields=["name"])
        
        # ID 1 should be first
        assert results[0][0]['id'] == "1"
        
    def test_fuzzy_search_empty_input(self):
        """Test behavior with empty entities list (Robustness check)."""
        searcher = FuzzySearcher()
        results = searcher.search("query", [], ["name"])
        assert results == []

## Semantic Search tests
class TestSemanticSearcher:
    @pytest.fixture
    def semantic_searcher(self):
        """Initialize SemanticSearcher with mocked manager and LLM."""
        manager = MagicMock()
        return SemanticSearcher(manager, "tool", "user", 0.3, 10)

    def test_merge_and_deduplicate_prioritizes_user(self, semantic_searcher):
        """Test that if a tool is in both User and Global, User entry wins but gets enriched."""

        user_id = "toolshed/repo/toolname/v1.0"
        global_id = "toolshed/repo/toolname/v2.0"

        # User's installed tool
        user_results = [{"name": "My Tool", "tool_id": user_id, "score": 0.9}]
        # Global version of the tool
        global_results = [{"name": "My Tool", "tool_id": global_id, "content": "Better content", "score": 0.8}]
 
        all_entities = [{"name": "My Tool", "tool_id": user_id}]
        
        merged = semantic_searcher._merge_and_deduplicate(
            global_results, user_results, "tool", all_entities
        )
        
        assert len(merged) == 1
        item = merged[0]
        assert item['tool_id'] == user_id  # Kept local ID
        assert item['content'] == "Better content" # Enriched content
        
        # Use approx for float comparison
        assert item['score'] == pytest.approx(0.85)

    def test_merge_promotes_global_if_owned_by_user(self, semantic_searcher):
        """Test that if vector search missed local, but found global, and user owns it, we promote it."""
        user_id = "toolshed/repo/toolname/v1.0"
        global_id = "toolshed/repo/toolname/v2.0"
        
        user_results = [] # Local missed
        global_results = [{"name": "My Tool", "tool_id": global_id, "content": "Global Content"}]
        
        # User owns local version
        all_entities = [{"name": "My Tool", "tool_id": user_id}]
        
        merged = semantic_searcher._merge_and_deduplicate(
            global_results, user_results, "tool", all_entities
        )
        
        assert len(merged) == 1
        item = merged[0]
        assert item['source'] == "user_instance" 
        assert item['tool_id'] == user_id      
        assert item['content'] == "Global Content"

## Search Engine tests
class TestSearchEngine:
    def test_extract_keywords(self):
        """Test n-gram extraction against a known corpus."""
        mock_cache = MagicMock()
        # Only "blast" and "alignment" are valid
        mock_cache.get_entities.return_value = "|blast|alignment|"
        
        # Mock manager
        mock_manager = MagicMock()
        
        # Create SearchEngine
        engine = SearchEngine(mock_manager, mock_cache, "tool", "user")
        
        # Query has valid keywords and noise
        keywords = engine.extract_keywords("run blast alignment please")
        
        assert "blast" in keywords
        assert "alignment" in keywords
        assert "please" not in keywords # Noise filtered