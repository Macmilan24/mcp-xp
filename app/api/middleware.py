import os
import logging
from dotenv import load_dotenv
import httpx
import json
import asyncio

from cryptography.fernet import Fernet
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from fastapi import Request, status

from sys import path
path.append('.')

from app.context import current_api_key


load_dotenv()

# Same as the one used in your register_key module
FERNET_SECRET = os.getenv("SECRET_KEY")
fernet = Fernet(FERNET_SECRET)

class GalaxyAPIKeyMiddleware(BaseHTTPMiddleware):
    """
    Reject any request whose USER-API-TOKEN header is missing
    or not accepted by the Galaxy server and sets apikey as a context.
    """
    # One shared httpx.AsyncClient for connection reuse
    def __init__(self, app):
        super().__init__(app)
        self.client = httpx.AsyncClient(timeout=5.0)
        self.GALAXY_URL = os.getenv("GALAXY_URL")
        self.log = logging.getLogger(__class__.__name__)


    async def dispatch(self, request: Request, call_next):
        
        public_paths = {"/", "/docs", "/redoc", "/openapi.json", "/register-key"}
        if request.url.path in public_paths or request.url.path.startswith("/static/"):
            return await call_next(request)
        
        api_key = request.headers.get("USER-API-TOKEN")
        if not api_key:
            self.log.error("missing USER_API_KEY header")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "API key header 'USER-API-TOKEN' is required"},
            )

        # Validate asynchronously (non-blocking)
        try:
            valid_key = await self.validate_decrypt_key(api_key)
        except httpx.RequestError as e:
            return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": f"Galaxy server unavailable: {e}"},
            )
        
        if valid_key :
            # Save key in context so downstream code can access it
            current_api_key.set(valid_key)

        else:
            self.log.error("Invalid USER_API_KEY header")
            return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Invalid Galaxy API key."},
            )

        return await call_next(request)

    async def validate_decrypt_key(self, token: str) -> bool:
        """Decrypts the encrypted token and returns the original Galaxy API key."""
        try:
            # Run the decryption and JSON parsing in a thread pool to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            decrypted = await loop.run_in_executor(None, fernet.decrypt, token.encode("utf-8"))
            data = await loop.run_in_executor(None, json.loads, decrypted.decode("utf-8"))
            return data["apikey"]
        except Exception as e:
            self.log.error(f"decryption failed: {e}")
            return None