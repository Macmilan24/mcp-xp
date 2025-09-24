import os
import logging
from dotenv import load_dotenv
import httpx
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from fastapi import Request, status

from sys import path

path.append(".")

from app.context import current_api_key

from app.api.security import create_mcp_access_token

load_dotenv()


class GalaxyAPIKeyMiddleware(BaseHTTPMiddleware):
    """
    Reject any request whose USER-API-KEY header is missing
    or not accepted by the Galaxy server and sets apikey as a context.
    """

    # One shared httpx.AsyncClient for connection reuse
    def __init__(self, app):
        super().__init__(app)
        self.client = httpx.AsyncClient(timeout=5.0)
        self.GALAXY_URL = os.getenv("GALAXY_URL")
        self.log = logging.getLogger(__class__.__name__)

    async def dispatch(self, request: Request, call_next):

        public_paths = {"/docs", "/redoc", "/openapi.json"}
        if request.url.path in public_paths or request.url.path.startswith("/static/"):
            return await call_next(request)

        api_key = request.headers.get("USER-API-KEY")

        if not api_key:
            self.log.error("missing USER_API_KEY header")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "API key header 'USER-API-KEY' is required"},
            )

        # Validate asynchronously (non-blocking)
        try:
            valid = await self._is_valid_key(api_key)
        except httpx.RequestError:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": "Galaxy server unavailable."},
            )

        if not valid:
            self.log.error("Invalid USER_API_KEY header")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid Galaxy API key."},
            )

        try:
            mcp_token = create_mcp_access_token(galaxy_api_key=api_key)
            request.state.mcp_token = mcp_token
        except Exception as e:
            self.log.error(f"Failed to create internal MCP token: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Could not generate internal security token."},
            )

        # Save key in context so downstream code can access it
        current_api_key.set(api_key)

        return await call_next(request)

    async def _is_valid_key(self, key: str) -> bool:
        """
        Lightweight Galaxy key check.
        Returns True only if the key is accepted.
        """
        url = f"{self.GALAXY_URL.rstrip('/')}/api/users/current"
        # headers = {"x-api-key": key}
        try:
            r = await self.client.get(url)
            return r.status_code == 200

        except httpx.RequestError as e:
            self.log.error(f"Galaxy not reachable: {e}")
            return False
