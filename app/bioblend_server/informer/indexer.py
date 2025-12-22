import logging
from abc import ABC, abstractmethod

from app.bioblend_server.informer.manager import InformerManager
from app.bioblend_server.informer.cache import RedisCache

class BaseIndexer(ABC):
    """Abstract base class for entity indexers."""
    
    @abstractmethod
    async def index_entities(self, entities: list[dict], collection_name: str):
        """ Index entities into storage. """
        pass
        
    @abstractmethod
    def delete_index(self, collection_name: str):
        """ Delete an index/collection. """
        pass


class QdrantIndexer(BaseIndexer):
    """Indexes entities into Qdrant vector database."""
    
    def __init__(self, manager: InformerManager):
        """ Initialize Qdrant indexer. """
        
        self.manager = manager
        self.logger = logging.getLogger(self.__class__.__name__)
        
    async def index_entities(self, entities: list[dict], collection_name: str):
        """
        Generate embeddings and index into Qdrant.
        
        Args:
            entities: List of entity dictionaries to index
            collection_name: Name of the collection
        """
        if not entities:
            self.logger.warning(f'No entities to index for {collection_name}')
            return
            
        try:
            # Delete old collection if they exist and add to new collection
            if self.manager.client.collection_exists(collection_name=collection_name):
                self.delete_index(collection_name=collection_name)
                
            await self.manager.embed_and_store_entities(
                entities=entities,
                collection_name=collection_name
            )
            self.logger.info(f'Successfully indexed {len(entities)} entities to Qdrant')
        except Exception as e:
            self.logger.error(f'Failed to index entities to Qdrant: {e}')
    
    def delete_index(self, collection_name: str):
        """ Delete Qdrant collection. """
        
        self.manager.delete_collection(collection_name)


class RedisIndexer(BaseIndexer):
    """ Indexes entities into Redis cache. """
    
    def __init__(self, cache: RedisCache):
        """ Initialize Redis indexer. """
        self.cache = cache
        self.logger = logging.getLogger(self.__class__.__name__)
        
    async def index_entities(self, entities: list[dict], collection_name: str, ttl: int):
        """
        Index entities into Redis with TTL.
        
        Args:
            entities: List of entity dictionaries to index
            collection_name: Name of the collection
            ttl: Time-to-live in seconds
        """
        
        if not entities:
            self.logger.warning(f'No entities to index for {collection_name}')
            return
            
        success = self.cache.set_entities(collection_name, entities, ttl)
        if success:
            self.logger.info(f'Successfully indexed {len(entities)} entities to Redis')
        else:
            self.logger.error(f'Failed to index entities to Redis')
    
    def delete_index(self, collection_name: str):
        """ Delete Redis collection. """
        
        self.cache.delete_entities(collection_name)
