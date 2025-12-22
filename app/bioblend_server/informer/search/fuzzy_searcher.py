"""
Fuzzy search functionality for Galaxy Informer.

Implements fuzzy string matching with priority field search.
"""

import logging
from rapidfuzz import process, fuzz


class FuzzySearcher:
    """Executes fuzzy string matching searches."""
    
    def __init__(self, threshold: int = 30, limit: int = 10):
        """ Initialize fuzzy searcher. """
        
        self.threshold = threshold
        self.limit = limit
        self.logger = logging.getLogger(self.__class__.__name__)
           
        
    def search(self, query: str, entities: list[dict], search_fields: list) -> list[tuple]:
        """
        Execute fuzzy search with priority fields.
        
        Implements a two-tier search strategy:
        1. Search priority fields (name, ID fields)
        2. Fallback to all other fields if no priority matches
        
        Args:
            query: Search query string
            entities: List of entity dictionaries to search
            search_fields: search fields to prioritize in fuzzy search.
            
        Returns:
            List of tuples (entity, score) sorted by score descending, max 5 results
        """
        self.logger.info('Fuzzy search for the query by priority fields')
        
        # Prepare priority candidates
        priority_candidates = []
        entity_map = {}  # Map: field string -> entity
        
        for entity in entities:
            for field in search_fields:
                if field in entity and isinstance(entity[field], str):
                    field_value = entity[field]
                    priority_candidates.append(field_value)
                    entity_map[field_value] = entity
        
        # Run fuzzy match across all priority fields at once
        results = process.extract(query, priority_candidates, scorer=fuzz.WRatio, limit=10)
        
        # Filter by threshold and collect matches
        priority_matches = []
        for match_str, score, _ in results:
            if score >= self.threshold:
                priority_matches.append((entity_map[match_str], score))
        
        if priority_matches:
            self.logger.info(f'Found {len(priority_matches)} matches in priority fields for search query: {query}')
            return sorted(priority_matches, key=lambda x: x[1], reverse=True)[:5]
        
        # No good priority matches found → fallback to all other fields
        self.logger.info(f'No matches found in the priority fields for search query: {query}, searching in all fields as a fallback')
        
        fallback_candidates = []
        fallback_map = {}
        
        for entity in entities:
            for key, value in entity.items():
                if key not in search_fields and isinstance(value, str):
                    fallback_candidates.append(value)
                    fallback_map[value] = entity
        
        results = process.extract(query, fallback_candidates, scorer=fuzz.WRatio, limit=10)
        
        fallback_matches = []
        for match_str, score, _ in results:
            if score >= self.threshold:
                fallback_matches.append((fallback_map[match_str], score))
        
        if fallback_matches:
            self.logger.info(f'Found {len(fallback_matches)} matches in fallback fields for search query: {query}')
            return sorted(fallback_matches, key=lambda x: x[1], reverse=True)[:5]
        
        self.logger.info(f'No matches found in either priority or fallback fields for search query: {query}')
        return []  # Always return a list, even if empty
