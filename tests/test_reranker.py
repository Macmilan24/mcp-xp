import pytest
from unittest.mock import MagicMock, patch
from sys import path
path.append(".")

from app.bioblend_server.informer.reranker import InformerReranker

@pytest.fixture
def mock_cross_encoder():
    with patch("app.bioblend_server.informer.reranker.CrossEncoder") as mock:
        yield mock

@pytest.fixture
def reranker(mock_cross_encoder):
    # Initialize reranker with mocked model
    return InformerReranker()

class TestWeightedRRF:
    """Tests for the Reciprocal Rank Fusion algorithm."""
    
    def test_rrf_prioritizes_semantic(self, reranker):
        """Test that semantic results (higher weight) push items higher."""
        # Item A is #1 in Fuzzy, Item B is #1 in Semantic
        fuzzy_results = [{"tool_id": "A", "score": 100}, {"tool_id": "B", "score": 50}]
        semantic_results = [{"tool_id": "B", "score": 0.9}, {"tool_id": "A", "score": 0.5}]
        
        results = reranker._weighted_rrf_fusion(
            fuzzy_results=fuzzy_results,
            semantic_results=semantic_results,
            entity_type="tool",
            fuzzy_weight=1.0,
            semantic_weight=3.0, # Semantic is weighted 3x
            k=60
        )
        
        
        assert results[0]['tool_id'] == "B"
        assert results[1]['tool_id'] == "A"
        assert 'rrf_score' in results[0]

    def test_rrf_deduplication(self, reranker):
        """Test that items appearing in both lists are merged."""
        
        fuzzy = [{"name": "Tool A", "tool_id": "A", "score": 10}]
        semantic = [{"name": "Tool A", "tool_id": "A", "score": 0.5}]
        
        results = reranker._weighted_rrf_fusion(fuzzy, semantic, "tool")
        
        assert len(results) == 1
        assert results[0]['tool_id'] == "A"

class TestCrossEncoderRerank:
    """Tests for the AI-based second stage reranking."""

    def test_rerank_sorts_by_ce_score(self, reranker):
        """Test that the cross encoder score dictates the final order."""
        query = "dna analysis"
        candidates = [
            {"tool_id": "bad_match", "content": "irrelevant"},
            {"tool_id": "good_match", "content": "highly relevant"}
        ]
        
        # Mock predict to return scores: [Low, High]
        reranker.cross_encoder.predict.return_value = [-5.0, 5.0]
        
        results = reranker._cross_encoder_rerank(query, candidates, top_k=2)
        
        assert results[0]['tool_id'] == "good_match"
        assert results[1]['tool_id'] == "bad_match"
        assert results[0]['ce_score'] > results[1]['ce_score']

    def test_fallback_on_model_failure(self, reranker):
        """Test that it returns original candidates if model prediction fails."""
        reranker.cross_encoder.predict.side_effect = Exception("Model exploded")
        candidates = [{"tool_id": "1"}]
        
        results = reranker._cross_encoder_rerank("query", candidates)
        
        assert results == candidates