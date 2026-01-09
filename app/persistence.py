import os
import logging
from dotenv import load_dotenv
from typing import Any, Optional, List, Iterable

from pymongo import AsyncMongoClient, errors
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from app.log_setup import configure_logging

load_dotenv()
configure_logging()

class MongoStore:
    
    """
    Async MongoDB key-value store

    Supports flexible connection via URI string or individual parameters.
    Includes retry mechanisms for connection and operations.

    Args:
        connection_string: Optional MongoDB URI (e.g., 'mongodb://user:pass@host:port/db').
        host: MongoDB host (ignored if connection_string provided).
        port: MongoDB port (ignored if connection_string provided).
        database_name: Database name.
        collection_name: Collection name for key-value storage.
        username: Username for authentication (ignored if connection_string provided).
        password: Password for authentication (ignored if connection_string provided).
        auth_source: Authentication source database.
        server_selection_timeout_ms: Timeout for server selection in milliseconds.
        max_retries: Maximum retries for connection and operations.
        retry_delay_ms: Delay between retries in milliseconds.

    """
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        host: str = os.getenv("MONGO_HOST", "localhost"),
        port: int = os.getenv("MONGO_PORT", 27017),
        database_name: str = "Galaxy_Integration",
        username: Optional[str] = None,
        password: Optional[str] = None,
        auth_source: str = "admin",
        server_selection_timeout_ms: int = 5000,
        max_retries: int = 3,
        retry_delay_ms: int = 1000,
    ):
        
        self._max_retries = max_retries
        self._retry_delay_seconds = retry_delay_ms / 1000.0
        
        self.log = logging.getLogger(__class__.__name__)

        if connection_string:
            self._client = AsyncMongoClient(
                connection_string,
                server_selection_timeout_ms=server_selection_timeout_ms,
            )
        else:
            self._client = AsyncMongoClient(
                host=host,
                port=port,
                username=username,
                password=password,
                auth_source=auth_source,
                server_selection_timeout_ms=server_selection_timeout_ms,
            )

        self._db = self._client[database_name]

    # Class async instantiator.
    
    @classmethod
    async def create(
        cls,
        connection_string: Optional[str] = None,
        host: str = "localhost",
        port: int = 27017,
        database_name: str = "Galaxy_Integration",
        username: Optional[str] = None,
        password: Optional[str] = None,
        auth_source: str = "admin",
        server_selection_timeout_ms: int = 5000,
        max_retries: int = 3,
        retry_delay_ms: int = 1000,
    ):
        """ Asynchronous classmethod to create and initialize the store instance. """
        
        self = cls(
            connection_string=connection_string,
            host=host,
            port=port,
            database_name=database_name,
            username=username,
            password=password,
            auth_source=auth_source,
            server_selection_timeout_ms=server_selection_timeout_ms,
            max_retries=max_retries,
            retry_delay_ms=retry_delay_ms,
        )
        await self._ping_with_retry()
        return self
    
    # Lifecycle helpers

    async def __aenter__(self):
        """ Async context manager entry. """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """ Async context manager exit: close connection. """
        self.close()

    def close(self) -> None:
        """ Cleanly close MongoDB connection. """
        try:
            self._client.close()
            self.log.info("MongoDB connection closed.")
        except Exception as exc:
            self.log.warning(f"Error closing MongoDB connection: {exc}")
            
            
    async def _ping_with_retry(self):
        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_fixed(self._retry_delay_seconds),
            retry=retry_if_exception_type(errors.PyMongoError),
            reraise=True,
        )
        async def ping():
            await self._client.admin.command("ping")

        try:
            await ping()
        except errors.PyMongoError as exc:
            self.log.error("Failed to ping MongoDB after retries.")
            raise RuntimeError("Failed to verify MongoDB connection.") from exc

    async def _ensure_indexes(self, collection_name: str):
        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_fixed(self._retry_delay_seconds),
            retry=retry_if_exception_type(errors.PyMongoError),
            reraise=True,
        )
        async def create():
            await self._db[collection_name].create_index("key", unique=True)
            
        try:
            await create()
        except errors.PyMongoError as exc:
            self.log.error("Failed to create unique index on 'key' after retries.")
            raise RuntimeError("Failed to create index.") from exc

    # Public API

    async def set(self, collection_name: str, key: str, value: Any) -> None:
        """ Asynchronously store or update a key-value pair with retry. Uses upsert semantics (idempotent). Supports any JSON-serializable value. """
        
        await self._ensure_indexes(collection_name)
        
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Key must be a non-empty string.")

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_fixed(self._retry_delay_seconds),
            retry=retry_if_exception_type(errors.PyMongoError),
            reraise=True,
        )
        async def upsert():
            await self._db[collection_name].update_one(
                {"key": key},
                {"$set": {"value": value}},
                upsert=True,
            )

        try:
            await upsert()
            self.log.debug(f"Successfully set key '{key}'.")
        except errors.PyMongoError as exc:
            self.log.error(f"Failed to set key '{key}' after retries.")
            raise RuntimeError(f"Failed to store key '{key}'.") from exc

    async def get(self, collection_name: str, key: str) -> Optional[Any]:
        """ Asynchronously retrieve a value by key with retry. Returns None if key does not exist. """
        
        await self._ensure_indexes(collection_name)
        
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Key must be a non-empty string.")

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_fixed(self._retry_delay_seconds),
            retry=retry_if_exception_type(errors.PyMongoError),
            reraise=True,
        )
        async def fetch():
            doc = await self._db[collection_name].find_one(
                {"key": key},
                {"_id": 0, "value": 1},
            )
            return doc["value"] if doc else None

        try:
            return await fetch()
        except errors.PyMongoError as exc:
            self.log.error(f"Failed to fetch key '{key}' after retries.")
            raise RuntimeError(f"Failed to fetch key '{key}'.") from exc
    
    async def add_to_set(self, collection_name: str, key: str, elements: Iterable[Any]) -> None:
        """
        Atomically add one or more elements to a set-like array value.
        - Uses MongoDB $addToSet semantics (no duplicates)
        - Upserts if key does not exist
        - Safe under concurrency
        """

        await self._ensure_indexes(collection_name)
         
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Key must be a non-empty string.")

        elements = list(elements)
        if not elements:
            return  # no-op by design

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_fixed(self._retry_delay_seconds),
            retry=retry_if_exception_type(errors.PyMongoError),
            reraise=True,
        )
        async def update():
            await self._db[collection_name].update_one(
                {"key": key},
                {"$addToSet": {"value": {"$each": elements}}},
                upsert=True,
            )

        try:
            await update()
            self.log.debug(f"Added {len(elements)} element(s) to set '{key}'.")
        except errors.PyMongoError as exc:
            self.log.error(f"Failed to add elements to set '{key}' after retries.")
            raise RuntimeError(f"Failed to update set for key '{key}'.") from exc
    
    async def delete(self, collection_name: str, key: str) -> bool:
        """ Asynchronously delete a key-value pair if it exists. Returns True if deleted, False if key did not exist. """
        
        await self._ensure_indexes(collection_name)
        
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Key must be a non-empty string.")

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_fixed(self._retry_delay_seconds),
            retry=retry_if_exception_type(errors.PyMongoError),
            reraise=True,
        )
        async def remove():
            result = await self._db[collection_name].delete_one({"key": key})
            return result.deleted_count > 0

        try:
            deleted = await remove()
            if deleted:
                self.log.debug(f"Successfully deleted key '{key}'.")
                
        except errors.PyMongoError as exc:
            self.log.error(f"Failed to delete key '{key}' after retries.")
            raise RuntimeError(f"Failed to delete key '{key}'.") from exc

    async def exists(self, key: str) -> bool:
        """ Asynchronously check if a key exists. """
        return await self.get(key) is not None

    async def list_keys(self, collection_name: str, prefix: Optional[str] = None, limit: int = 100) -> List[str]:
        """
        Asynchronously list keys, optionally filtered by prefix.

        Args:
            prefix: Optional prefix to filter keys.
            limit: Maximum number of keys to return.

        Returns:
            List of matching keys.
        """
        await self._ensure_indexes(collection_name)
         
        query = {"key": {"$regex": f"^{prefix}"}} if prefix else {}
        
        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_fixed(self._retry_delay_seconds),
            retry=retry_if_exception_type(errors.PyMongoError),
            reraise=True,
        )
        async def fetch_keys():
            keys = []
            cursor = self._db[collection_name].find(query, {"_id": 0, "key": 1}).limit(limit)
            async for doc in cursor:
                keys.append(doc["key"])
            return keys

        try:
            return await fetch_keys()
        except errors.PyMongoError as exc:
            self.log.error("Failed to list keys after retries.")
            raise RuntimeError("Failed to list keys.") from exc
            
    async def update_value_element(
        self,
        collection_name: str,
        key: str,
        match_field: str,
        match_value: Any,
        update_field: str,
        new_value: Any,
    ) -> None:
        """
        Update a field inside a matched object in the `value` array.
        """

        await self._ensure_indexes(collection_name)

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_fixed(self._retry_delay_seconds),
            retry=retry_if_exception_type(errors.PyMongoError),
            reraise=True,
        )
        async def update():
            await self._db[collection_name].update_one(
                {
                    "key": key,
                    f"value.{match_field}": match_value,
                },
                {
                    "$set": {
                        f"value.$.{update_field}": new_value
                    }
                },
            )

        try:
            await update()
        except errors.PyMongoError as exc:
            self.log.error("Failed to update a selected element.")
            raise RuntimeError("Failed to update a selected element.") from exc
        

    async def remove_from_value_array(
        self,
        collection_name: str,
        key: str,
        match_field: str,
        match_value: Any,
    ) -> bool:
        """
        Remove element(s) from the `value` array where match_field == match_value.

        Returns True if at least one element was removed.
        """

        await self._ensure_indexes(collection_name)

        if not isinstance(key, str) or not key.strip():
            raise ValueError("Key must be a non-empty string.")

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_fixed(self._retry_delay_seconds),
            retry=retry_if_exception_type(errors.PyMongoError),
            reraise=True,
        )
        async def pull():
            result = await self._db[collection_name].update_one(
                {"key": key},
                {
                    "$pull": {
                        "value": {match_field: match_value}
                    }
                },
            )
            return result.modified_count > 0

        try:
            removed = await pull()
            if removed:
                self.log.debug(
                    f"Removed element(s) from value array where {match_field}={match_value}."
                )
            return removed
        except errors.PyMongoError as exc:
            self.log.error("Failed to remove element(s) from value array after retries.")
            raise RuntimeError("Failed to update value array.") from exc