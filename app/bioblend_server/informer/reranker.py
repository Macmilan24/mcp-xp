
import json
import logging
from typing import List, Dict
from sentence_transformers import CrossEncoder

from app.bioblend_server.informer.utils import InformerHandler


class InformerReranker:
    """Result reranking with hybrid weighted RRF + cross-encoder."""
    
    def __init__(self):
        """ Initialize reranker with cross-encoder model.  """
        self.logger = logging.getLogger(__class__.__name__)
        
        # Initialize cross-encoder model
        try:
            self.logger.debug(f"Loading cross-encoder model: {InformerHandler.CROSS_ENCODER.value}")
            self.cross_encoder = CrossEncoder(InformerHandler.CROSS_ENCODER.value)
            self.logger.info("Cross-encoder model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load cross-encoder model: {e}")
            self.cross_encoder = None
            
            
    def _weighted_rrf_fusion(
        self,
        fuzzy_results: List[Dict],
        semantic_results: List[Dict],
        entity_type: str,
        fuzzy_weight: float = 1.0,
        semantic_weight: float = 3.0,
        top_n: int = 50,
        k: int = 60
    ) -> List[Dict]:
        """
        Stage 1: Weighted Reciprocal Rank Fusion.
        
        Args:
            fuzzy_results: Fuzzy search results with scores
            semantic_results: Semantic search results with scores
            id_field: 
            fuzzy_weight: Weight for fuzzy results (default: 1)
            semantic_weight: Weight for semantic results (default: 3)
            top_n: Number of top candidates to return
            k: RRF constant (default: 60)
            
        Returns:
            Top N candidates with RRF scores
        """
        id_field = None   
        # Normalize scores
        fuzzy_normalized = self._normalize_scores(fuzzy_results, 'score')
        semantic_normalized = self._normalize_scores(semantic_results, 'score')
        
        if entity_type == "tool":
            id_field = "tool_id"
        elif entity_type == "dataset":
            id_field ="dataset_id"
        else:
            id_field = "name"
        
        # Build rank maps
        fuzzy_rank_map = {
            item.get(id_field): (rank + 1, item) 
            for rank, item in enumerate(fuzzy_normalized)
        }
        semantic_rank_map = {
            item.get(id_field): (rank + 1, item) 
            for rank, item in enumerate(semantic_normalized)
        }
        
        # Collect all unique IDs
        all_ids = set(fuzzy_rank_map.keys()) | set(semantic_rank_map.keys())
        
        # Calculate RRF scores
        rrf_scores = []
        for item_id in all_ids:
            if item_id is None:
                continue
                
            rrf_score = 0.0
            item = None
            
            # Fuzzy contribution
            if item_id in fuzzy_rank_map:
                fuzzy_rank, fuzzy_item = fuzzy_rank_map[item_id]
                rrf_score += fuzzy_weight / (k + fuzzy_rank)
                item = fuzzy_item
            
            # Semantic contribution
            if item_id in semantic_rank_map:
                semantic_rank, semantic_item = semantic_rank_map[item_id]
                rrf_score += semantic_weight / (k + semantic_rank)
                # Prefer semantic item if available
                item = semantic_item
            
            if item:
                # Add RRF score to item
                item_copy = item.copy()
                item_copy['rrf_score'] = rrf_score
                rrf_scores.append(item_copy)
        
        # Sort by RRF score (descending) and return top N
        rrf_scores.sort(key=lambda x: x['rrf_score'], reverse=True)
        return rrf_scores[:top_n]
    
    def _cross_encoder_rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Stage 2: Cross-encoder reranking.
        
        Args:
            query: User query
            candidates: Candidate results from Stage 1
            top_k: Number of top results to return
            
        Returns:
            Top K reranked results
        """
        if not self.cross_encoder:
            self.logger.warning("Cross-encoder not available")
            return candidates[:top_k]
        
        if len(candidates) == 0:
            return []
        
        # Prepare query-candidate pairs
        pairs = []
        for candidate in candidates:
            # Build candidate text from available fields
            candidate_text = self._build_candidate_text(candidate)
            pairs.append([query, candidate_text])
        
        # Get cross-encoder scores
        try:
            ce_scores = self.cross_encoder.predict(pairs)
            
            # Add cross-encoder scores to candidates
            for idx, candidate in enumerate(candidates):
                candidate['ce_score'] = float(ce_scores[idx])
            
            # Sort by cross-encoder score (descending)
            candidates.sort(key=lambda x: x['ce_score'], reverse=True)
            
            return candidates[:top_k]
            
        except Exception as e:
            self.logger.error(f"Cross-encoder prediction failed: {e}")
            # Fallback to RRF ranking
            return candidates[:top_k]
    
    def _build_candidate_text(self, candidate: Dict) -> str:
        """
        Build text representation of candidate for cross-encoder.
        
        Args:
            candidate: Candidate result dictionary
            
        Returns:
            Text representation
        """
        # Prioritize certain fields for text representation
        text_parts = []
        
        # Add name
        if 'name' in candidate:
            text_parts.append(f"Name: {candidate['name']}")
        
        # Add description
        if 'description' in candidate and candidate['description']:
            text_parts.append(f"Description: {candidate['description']}")
        
        # Add content (if available and not too long)
        if 'content' in candidate and candidate['content']:
            content = str(candidate['content'])
            # Truncate if too long
            if len(content) > 500:
                content = content[:500] + "..."
            text_parts.append(f"Content: {content}")
        
        # Add tool_id
        if 'tool_id' in candidate:
            text_parts.append(f"ID: {candidate['tool_id']}")
        
        return " | ".join(text_parts) if text_parts else str(candidate)
    
    def _normalize_scores(self, results: List[Dict], score_field: str = 'score') -> List[Dict]:
        """
        Normalize scores using min-max normalization.
        
        Args:
            results: List of result dictionaries
            score_field: Field name containing the score
            
        Returns:
            Results with normalized scores
        """
        if not results:
            return []
        
        # Extract scores
        scores = [item.get(score_field, 0) for item in results]
        
        if not scores:
            return results
        
        # Min-max normalization
        min_score = min(scores)
        max_score = max(scores)
        
        # Handle case where all scores are the same
        if max_score == min_score:
            normalized_results = []
            for item in results:
                item_copy = item.copy()
                item_copy['normalized_score'] = 1.0
                normalized_results.append(item_copy)
            return normalized_results
        
        # Normalize
        normalized_results = []
        for item in results:
            item_copy = item.copy()
            score = item.get(score_field, 0)
            normalized_score = (score - min_score) / (max_score - min_score)
            item_copy['normalized_score'] = normalized_score
            normalized_results.append(item_copy)
        
        return normalized_results
    
    def _fallback_combine(self, fuzzy_results: list, semantic_results: list, id_field: str) -> dict:
        """
        Fallback: combine fuzzy and semantic results by ID.
        
        Args:
            fuzzy_results: Fuzzy search results
            semantic_results: Semantic search results
            id_field:
            
        Returns:
            Combined results dictionary
        """
        
        combined = {}

        # 1. Insert semantic results.
        for item in semantic_results:
            item_id = item.get(id_field)
            if item_id is not None:
                combined[item_id] = item

        # 2. Insert fuzzy results (only if ID exists, overwrite allowed) ----
        for item in fuzzy_results:
            item_id = item.get(id_field)
            if item_id is not None:
                # override or add
                combined[item_id] = item

        # 3. Return top 3 (or less)
        # Combined is a dict keyed by ID; keep original insertion order
        return dict(list(combined.items())[:3])
    
    
    async def rerank_results(
        self, 
        query: str, 
        fuzzy_results: list, 
        semantic_results: list, 
        entity_type: str,
        top_n: int = 50,
        final_k: int = 5
    ) -> dict:
        """
        Rerank and select top results using two-stage pipeline.
        
        Stage 1: Weighted RRF (fuzzy:1, semantic:3) -> top 50 candidates
        Stage 2: Cross-encoder reranking -> top K results
        
        Args:
            query: Original user query
            fuzzy_results: Results from fuzzy search (list of dicts with 'score' key)
            semantic_results: Results from semantic search (list of dicts with 'score' key)
            id_field:
            top_n: Number of candidates to pass to cross-encoder (default: 50)
            final_k: Number of final results to return (default: 10)
            
        Returns:
            List of top K reranked results
        """
        self.logger.info("Starting two-stage reranking pipeline")
        
        try:
            # Stage 1: Weighted RRF
            stage1_candidates = self._weighted_rrf_fusion(
                fuzzy_results=fuzzy_results,
                semantic_results=semantic_results,
                entity_type = entity_type,
                fuzzy_weight=1,
                semantic_weight=3,
                top_n=top_n
            )
            
            self.logger.info(f"Stage 1 complete: {len(stage1_candidates)} candidates selected")
            
            # Stage 2: Cross-encoder reranking
            final_results = []
            if self.cross_encoder and len(stage1_candidates) > 0:
                final_results = self._cross_encoder_rerank(
                    query=query,
                    candidates=stage1_candidates,
                    top_k=final_k
                )
                self.logger.info(f"Stage 2 complete: {len(final_results)} final results")
            else:
                # Fallback: return top K from Stage 1
                self.logger.warning("Cross-encoder unavailable, returning Stage 1 results")
                final_results = stage1_candidates[:final_k]
                
            for item in final_results:
                item.pop("score", None)
                item.pop("normalized_score", None)
        
            
            self.logger.debug(f"final reranked results{json.dumps(final_results, indent=2)}")
            
            return final_results

        except Exception as e:
            self.logger.error(f"Reranking failed: {e}", exc_info=True)
            # Ultimate fallback
            return self._fallback_combine(fuzzy_results, semantic_results, f"{entity_type}_id")