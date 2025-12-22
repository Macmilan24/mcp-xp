"""
Galaxy Informer - Intelligent Information Retrieval for Galaxy Bioinformatics.

This package provides a modular RAG (Retrieval-Augmented Generation) system
for searching and retrieving information about Galaxy entities.

Modular Architecture:
- cache: Redis-based caching
- indexing: Redis and Qdrant indexing operations  
- search: Fuzzy and semantic search functionality
- reranking: LLM-based result reranking
- orchestrator: Main coordination layer

Usage:
    from app.bioblend_server.informer import GalaxyInformer
    
    informer = await GalaxyInformer.create(
        galaxy_client=client,
        entity_type="tool",
        llm_provider="gemini"
    )
    
    result = await informer.get_entity_info("RNA-seq quality control")
"""

# New modular architecture
from .informer import GalaxyInformer

# Modular components
from .cache import RedisCache
from .indexer import BaseIndexer, RedisIndexer, QdrantIndexer
from .search import FuzzySearcher, SemanticSearcher, SearchEngine
from .reranker import InformerReranker

# Existing components (backward compatibility)
from .data_provider import GalaxyDataProvider
from .manager import InformerManager
from .global_rec import GlobalRecommender

__all__ = [
    # Main interface
    'GalaxyInformer',
    
    # Modular components
    'RedisCache',
    'BaseIndexer',
    'RedisIndexer',
    'QdrantIndexer',
    'FuzzySearcher',
    'SemanticSearcher',
    'SearchEngine',
    'InformerReranker',
    
    # Supporting components
    'GalaxyDataProvider',
    'InformerManager',
    'GlobalRecommender',
]

__version__ = '2.0.0'
