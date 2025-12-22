import logging
import asyncio
import re

from app.bioblend_server.informer.cache import RedisCache
from app.bioblend_server.informer.manager import InformerManager
from app.bioblend_server.informer.search.semantic_searcher import SemanticSearcher
from app.bioblend_server.informer.search.fuzzy_searcher import FuzzySearcher
from app.bioblend_server.informer.utils import SearchThresholds


class SearchEngine:
    
    def __init__(self, 
                 manager: InformerManager,
                 cache: RedisCache,
                 entity_type: str, 
                 username: str, 
                 ):
        
        self.log = logging.getLogger(__class__.__name__)
        self.manager = manager
        self.cache = cache
        self.entity_type = entity_type
        self.username = username

        # Instantiate the semantic searcher.
        self.semantic_searcher=  SemanticSearcher(
            vector_manager= manager,
            entity_type = entity_type,
            username = self.username,
            score_threshold= SearchThresholds.SEMANTIC_THRESHOLD.value,
            limit = SearchThresholds.SEARCH_LIMIT.value
        )
        
        self.fuzzy_searcher = FuzzySearcher(
            threshold = SearchThresholds.FUZZY_THRESHOLD.value,
            limit = SearchThresholds.SEARCH_LIMIT.value
        )
    
    def get_collection_name(self) -> str:
        """Get the collection and corpus name for the current entity type and user."""
        if self.entity_type == "tool":
            return f'Galaxy_{self.entity_type}', f'Galaxy_{self.entity_type}_corpus'
        else:
            return f'Galaxy_{self.entity_type}_{self.username}', f'Galaxy_{self.entity_type}_{self.username}_corpus'
              
    async def semantic_search(self, query:str, entities: list[dict]):
        """ Semantic search for the user query. """
        
        results = await self.semantic_searcher.search(
            query = query,
            entity_type = self.entity_type,
            entities = entities
        )
        
        return results
    
    def _squash(self, text: str) -> str:
        """Helper to remove non-alphanumeric chars (except dash/space) for matching."""
        return re.sub(r"[^\w\s\-]", "", text.lower())
    
    def _get_or_build_index(self, corpus_name: str) -> str:
        """ Builds and caches a single normalized string containing all tool names. """
        
        indexed_corpus = "indexed_corpus"
        index = self.cache.get_entities(indexed_corpus)
        # Check Cache
        if index:
            return index

        # Fetch Raw entity names
        raw_corpus_names = self.cache.get_entities(collection_name=corpus_name)
        
        # Normalize and Join
        valid_terms = {self._squash(name) for name in raw_corpus_names}
        
        # Join with a unique separator
        index = "|" + "|".join(valid_terms) + "|"
        self.cache.set_entities(collection_name=indexed_corpus, entities=index, ttl = 30)        
        self.log.info(f"Built keyword index for {corpus_name} with {len(valid_terms)} unique names.")
        return index

    def extract_keywords(self, query: str, max_ngram: int = 3) -> list[str]:
        """ Extracts validated n-grams by noise filtering Corpus Index validation. """
        
        _, corpus_name = self.get_collection_name()
        # Get Index
        search_index = self._get_or_build_index(corpus_name)
        # preprocess query
        query_clean = self._squash(query)
        tokens = [t for t in query_clean.split()]
        
        if not tokens:
            return []

        # generate and validate candidates
        validated_keywords = set()
        
        for n in range(1, max_ngram + 1):
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i:i + n])
                
                if ngram in search_index:
                    validated_keywords.add(ngram)
                    
        result_list = list(validated_keywords)
        self.log.info(f"Query: '{query}' -> Validated Keywords: {result_list}")
        
        return result_list
    
    async def fuzzy_search(self, keyword: str, entities: list[dict], search_fields: list):
        
        results = await asyncio.to_thread(
            self.fuzzy_searcher.search,
            query=keyword,
            entities=entities,
            search_fields = search_fields
        )
        
        # structure results by converting them to dict.
        structured_results = [{**entity, "score": score, "source" : "user_instance"} for entity, score in results]
        return structured_results