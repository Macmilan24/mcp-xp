import json
from typing import Dict, List, Optional
from datetime import datetime
import redis
import logging
import asyncio

from app.bioblend_server.executor.workflow_manager import WorkflowManager

log = logging.getLogger(__name__)

class InvocationCache:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.local_cache = {}
        self.last_deleted_refresh = {}
        self.log = logging.getLogger(__class__.__name__)
        
    # Workflow metadata caching
    async def get_workflows_cache(self, api_key: str) -> Optional[List[Dict]]:
        """Get cached processed workflows list"""
        try:
            cache_key = f"workflows:{api_key}"
            cached_data = self.redis.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception as e:
            self.log.error(f"Error getting workflows cache: {e}")
            return None
    
    async def set_workflows_cache(self, api_key: str, workflows: List[Dict], ttl: int = 3600):
        """Cache processed workflows list with 1-hour TTL"""
        try:
            cache_key = f"workflows:{api_key}"
            self.redis.setex(cache_key, ttl, json.dumps(workflows))
        except Exception as e:
            self.log.error(f"Error setting workflows cache: {e}")
    
    # Invocation-Workflow mapping cache
    async def get_invocation_workflow_mapping(self, api_key: str) -> Dict[str, Dict]:
        """Get cached mapping of invocation_id -> {workflow_name, workflow_id}"""
        try:
            cache_key = f"invocation_workflow_map:{api_key}"
            cached_data = self.redis.hgetall(cache_key)
            if cached_data:
                # Convert bytes keys/values to strings and parse JSON values
                return {
                    k: json.loads(v) 
                    for k, v in cached_data.items()
                }
            return {}
        except Exception as e:
            self.log.error(f"Error getting invocation workflow mapping: {e}")
            return {}
    
    async def set_invocation_workflow_mapping(self, api_key: str, mapping: Dict[str, Dict], ttl: int = 300):
        """Cache invocation-workflow mapping with 5-minute TTL"""
        try:
            cache_key = f"invocation_workflow_map:{api_key}"
            # Convert to JSON strings for Redis storage
            redis_mapping = {
                inv_id: json.dumps(wf_info) 
                for inv_id, wf_info in mapping.items()
            }
            
            # Use pipeline for atomic operations
            pipe = self.redis.pipeline()
            pipe.delete(cache_key)  # Clear existing data
            if redis_mapping:  # Only set if we have data
                pipe.hset(name = cache_key, mapping = redis_mapping)
            pipe.expire(cache_key, ttl)
            pipe.execute()
        except Exception as e:
            self.log.error(f"Error setting invocation workflow mapping: {e}")
    
    # Response caching for common filter combinations
    async def get_response_cache(self, api_key: str, workflow_id: str = None, history_id: str = None) -> Optional[Dict]:
        """Get cached response for specific filter combination"""
        try:
            cache_key = f"invocations_response:{api_key}:{workflow_id or 'all'}:{history_id or 'all'}"
            cached_data = self.redis.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception as e:
            self.log.error(f"Error getting response cache: {e}")
            return None
    
    async def set_response_cache(self, api_key: str, response: Dict, workflow_id: str = None, 
                               history_id: str = None, ttl: int = 20):
        """Cache response for specific filter combination with 20 seconds TTL"""
        try:
            cache_key = f"invocations_response:{api_key}:{workflow_id or 'all'}:{history_id or 'all'}"
            self.redis.setex(cache_key, ttl, json.dumps(response))
        except Exception as e:
            self.log.error(f"Error setting response cache: {e}")
    
    # Deleted invocations caching (with local memory cache)
    async def get_deleted_invocations(self, api_key: str) -> set:
        """Get deleted invocations with local caching"""
        try:
            now = datetime.now()
            cache_key = f"deleted_invocations_local:{api_key}"
            
            # Check if we have fresh local cache (30 seconds)
            if (cache_key in self.local_cache and 
                cache_key in self.last_deleted_refresh and
                (now - self.last_deleted_refresh[cache_key]).seconds < 30):
                return self.local_cache[cache_key]
            
            # Refresh from Redis
            redis_key = f"deleted_invocations:{api_key}"
            deleted_set = {inv_id.decode() if isinstance(inv_id, bytes) else inv_id 
                          for inv_id in self.redis.smembers(redis_key)}
            
            # Update local cache
            self.local_cache[cache_key] = deleted_set
            self.last_deleted_refresh[cache_key] = now
            
            return deleted_set
        except Exception as e:
            self.log.error(f"Error getting deleted invocations: {e}")
            return set()
    
    # Invocation list caching
    async def get_invocations_cache(self, api_key: str, filters: Dict) -> Optional[List[Dict]]:
        """Get cached invocations list"""
        try:
            # Create cache key based on filters
            filter_str = "_".join([f"{k}:{v}" for k, v in sorted(filters.items()) if v is not None])
            cache_key = f"invocations_raw:{api_key}:{filter_str or 'all'}"
            
            cached_data = self.redis.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception as e:
            self.log.error(f"Error getting invocations cache: {e}")
            return None
    
    async def set_invocations_cache(self, api_key: str, invocations: List[Dict], 
                                  filters: Dict, ttl: int = 60):
        """Cache invocations list with 1-minute TTL"""
        try:
            filter_str = "_".join([f"{k}:{v}" for k, v in sorted(filters.items()) if v is not None])
            cache_key = f"invocations_raw:{api_key}:{filter_str or 'all'}"
            self.redis.setex(cache_key, ttl, json.dumps(invocations))
        except Exception as e:
            self.log.error(f"Error setting invocations cache: {e}")
    
    async def add_to_deleted_invocations(self, api_key: str, invocation_ids: List[str]):
        """Add invocation IDs to the deleted set"""
        try:
            redis_key = f"deleted_invocations:{api_key}"
            if invocation_ids:
                self.redis.sadd(redis_key, *invocation_ids)
            # Refresh local cache if needed
            await self.get_deleted_invocations(api_key)  # Forces refresh
        except Exception as e:
            self.log.error(f"Error adding to deleted invocations: {e}")
            
    # Request deduplication
    async def is_duplicate_request(self, api_key: str, request_hash: str, ttl: int = 3) -> bool:
        """Check if this is a duplicate request within TTL seconds"""
        try:
            cache_key = f"request_dedup:{api_key}:{request_hash}"
            exists = self.redis.exists(cache_key)
            if not exists:
                self.redis.setex(cache_key, ttl, "1")
                return False
            return True
        except Exception as e:
            self.log.error(f"Error checking duplicate request: {e}")
            return False
    
    async def is_duplicate_workflow_request(self, api_key: str, request_hash: str, ttl: int = 10) -> bool:
        """Check if this is a duplicate workflow request within TTL seconds"""
        try:
            cache_key = f"workflow_request_dedup:{api_key}:{request_hash}"
            exists = self.redis.exists(cache_key)
            if not exists:
                self.redis.setex(cache_key, ttl, "1")
                return False
            return True
        except Exception as e:
            self.log.error(f"Error checking duplicate workflow request: {e}")
            return False
    
    # Health check and cache warming
    async def warm_cache(self, api_key: str, workflow_manager: WorkflowManager):
        """Warm up the cache with essential data"""
        try:
            # Warm workflows cache
            workflows = await workflow_manager.gi_object.gi.workflows.get_workflows()
            await self.set_workflows_cache(api_key, workflows)
            
            mapping = {}          
            # Use parallel processing to build mapping efficiently
            for workflow in workflows:
                try:
                    wf_invocations = await workflow_manager.gi_object.gi.invocations.get_invocations(workflow_id=workflow['id'])
                    for inv in wf_invocations:
                        mapping[inv['id']] = {
                            'workflow_name': workflow['name'],
                            'workflow_id': workflow['id']
                        }
                except Exception as e:
                    self.log.warning(f"Failed to get invocations for workflow {workflow['id']}: {e}")
                    continue
            
            await self.set_invocation_workflow_mapping(api_key, mapping)
            self.log.info(f"Cache warmed for user ******{api_key[-5]}: {len(workflows)} workflows, {len(mapping)} invocation mappings")
            
        except Exception as e:
            self.log.error(f"Error warming cache: {e}")

    async def get_invocation_result(self, api_key: str, invocation_id: str) -> Optional[Dict]:
        """Get cached full invocation result"""
        try:
            cache_key = f"invocation_result:{api_key}:{invocation_id}"
            cached_data = self.redis.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception as e:
            self.log.error(f"Error getting invocation result: {e}")
            return None

    async def set_invocation_result(self, api_key: str, invocation_id: str, result: Dict, ttl: int = 86400):
        """Cache full invocation result with 1-day TTL"""
        try:
            cache_key = f"invocation_result:{api_key}:{invocation_id}"
            self.redis.setex(cache_key, ttl,  json.dumps(result))
        except Exception as e:
            self.log.error(f"Error setting invocation result: {e}")
            
    async def set_invocation_state(self, api_key: str, invocation_id: str, state: str):
        """Set or update an invocation state inside the hash for a given API key"""
        try:
            if state not in {"Pending", "Failed", "Complete"}:
                raise ValueError(f"Invalid state: {state}")
            cache_key = f"invocation_states:{api_key}"
            self.redis.hset(cache_key, invocation_id, state)
            self.log.info(f" new invocation state has been set for user: ****{api_key[-5]} invocation id: {invocation_id}")
        except Exception as e:
            self.log.error(f"Error setting invocation state: {e}")

    async def get_invocation_state(self, api_key: str, invocation_id: str) -> Optional[str]:
        """Get a single invocation state"""
        try:
            cache_key = f"invocation_states:{api_key}"
            raw = self.redis.hget(cache_key, invocation_id)
            return raw if raw else None
        except Exception as e:
            self.log.error(f"Error getting invocation state: {e}")
            return None

    async def delete_invocation_state(self, api_key: str, invocation_id: str):
        """Delete a single invocation state"""
        try:
            cache_key = f"invocation_states:{api_key}"
            await asyncio.to_thread(self.redis.hdel, cache_key, invocation_id)
        except Exception as e:
            self.log.error(f"Error deleting invocation state: {e}")