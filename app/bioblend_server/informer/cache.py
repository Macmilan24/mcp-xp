"""
Redis cache management for Galaxy Informer.

This module handles all Redis caching operations for entity data.
"""

import redis
import json
import logging


class RedisCache:
    """Manages Redis caching for Galaxy entities."""
    
    def __init__(self, host: str, port: int, db: int = 0):
        """
        Initialize Redis cache client.
        
        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        
    def get_entities(self, collection_name: str) -> list[dict] | None:
        """ Retrieve cached entities from Redis. """
        
        try:
            entities_str = self.client.get(collection_name)
            if entities_str:
                self.logger.info(f'Retrieved cached entities from Redis: {collection_name}')
                return json.loads(entities_str)
            return None
        except (redis.RedisError, json.JSONDecodeError) as e:
            self.logger.error(f'Redis cache retrieval failed: {e}')
            return None
    
    def set_entities(self, collection_name: str, entities: list[dict] | str, ttl: int) -> bool:
        """ Cache entities in Redis with TTL. """
        
        try:
            if isinstance(entities, str):
                self.client.setex(collection_name, ttl, entities)
            else:
                self.client.setex(collection_name, ttl, json.dumps(entities))
                self.logger.info(f'Saved entities to Redis: {collection_name} (TTL: {ttl}s)')
            return True
        except Exception as e:
            self.logger.error(f'Failed to save entities to Redis: {e}')
            return False
    
    def get_string(self, key: str) -> str | None:
        """ Retrieve a raw string value from Redis. """
        
        try:
            value = self.client.get(key)
            if value:
                self.logger.info(f'Retrieved string from Redis: {key}')
                return value
            return None
        except redis.RedisError as e:
            self.logger.error(f'Redis string retrieval failed: {e}')
            return None

    def set_string(self, key: str, value: str, ttl: int) -> bool:
        """ Cache a raw string in Redis with TTL. """
        
        try:
            self.client.setex(key, ttl, value)
            self.logger.info(f'Saved string to Redis: {key} (TTL: {ttl}s)')
            return True
        except redis.RedisError as e:
            self.logger.error(f'Failed to save string to Redis: {e}')
            return False
        
    def delete_entities(self, collection_name: str) -> bool:
        """ Delete cached entities from Redis. """
        
        try:
            self.client.delete(collection_name)
            self.logger.info(f'Deleted cached entities: {collection_name}')
            return True
        except redis.RedisError as e:
            self.logger.error(f'Failed to delete cached entities: {e}')
            return False
