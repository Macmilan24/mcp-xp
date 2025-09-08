import os
import json
import logging
import asyncio
from typing import Optional

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from cryptography.fernet import Fernet, InvalidToken
import jwt  # PyJWT

from app.context import current_api_key

# Environment / secrets
FERNET_SECRET = os.getenv("SECRET_KEY")
if not FERNET_SECRET:
    raise RuntimeError("SECRET_KEY (Fernet secret) is required in env")
fernet = Fernet(FERNET_SECRET)


GALAXY_API_TOKEN = "galaxy_api_token"


class JWTGalaxyKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware that expects:
      Authorization: Bearer <JWT>
    The JWT must contain a claim ('galaxy_api_token') that is either:
      - a fernet-encrypted JSON payload like {"apikey": "<actual_key>"} (what the register user produces)
    The middleware will set current_api_key to the final plain api key string.
    """

    def __init__(self, app):
        super().__init__(app)
        self.log = logging.getLogger(self.__class__.__name__)

    async def dispatch(self, request: Request, call_next):
        # Allow public paths if you need them (adjust to your app). Remove if unwanted.
        public_paths = {"/", "/docs", "/redoc", "/openapi.json", "/register-user"}
        if request.url.path in public_paths or request.url.path.startswith("/static/"):
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Authorization header with Bearer token is required."},
            )

        token = auth.split(" ")[1].strip()
        try:
            payload = self._decode_jwt(token)
        except HTTPException as e:
            return JSONResponse(status_code=e.status_code, content={"detail": e.detail})

        # Extract the API token claim
        if GALAXY_API_TOKEN not in payload:
            self.log.error("JWT missing API key claim '%s'", GALAXY_API_TOKEN)
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": f"JWT missing required claim '{GALAXY_API_TOKEN}'."},
            )

        galaxy_jwt_token = payload[GALAXY_API_TOKEN]
        if not galaxy_jwt_token:
            self.log.error("Empty API key claim")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "API key claim is empty."},
            )

        # Try to decrypt claim_value (it might be the fernet token string produced by register-user)
        apikey = await self._decrypt_api_token(galaxy_jwt_token)
        if not apikey:
            # fallback: treat as raw api key string
            if isinstance(galaxy_jwt_token, str) and galaxy_jwt_token.strip():
                apikey = galaxy_jwt_token.strip()
            else:
                self.log.error("Unable to obtain Galaxy API key from JWT claim")
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Invalid API key in JWT claim."},
                )

        # set the context for downstream handlers
        current_api_key.set(apikey)
        return await call_next(request)

    def _decode_jwt(self, token: str) -> dict:
        """
        Decode/verify JWT synchronously (PyJWT). Raises HTTPException(401) on invalid token.
        For RS-based tokens, JWT_SECRET should contain the public key.
        """
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
            return payload
        except jwt.InvalidTokenError as e:
            self.log.error("Invalid JWT: %s", e)
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid JWT: {e}")

    async def _decrypt_api_token(self, token_str: str) -> Optional[str]:
        """
        If token_str is a fernet-encrypted payload (bytes when encoded),
        decrypt and parse JSON for {"apikey": "<value>"} and return the value.
        Returns None if decryption/parsing fails so caller can fallback to raw token.
        """
        if not isinstance(token_str, str) or not token_str:
            return None
        loop = asyncio.get_running_loop()
        try:
            decrypted = await loop.run_in_executor(None, fernet.decrypt, token_str.encode("utf-8"))
            parsed = await loop.run_in_executor(None, json.loads, decrypted.decode("utf-8"))
            apikey = parsed.get("apikey")
            if apikey and isinstance(apikey, str):
                return apikey
            self.log.error("Decrypted JWT galaxy api-key payload missing 'apikey' field")
            return None
        except (InvalidToken, Exception) as e:
            # Not a fernet payload or parse failed; return None so fallback can apply
            self.log.debug("Fernet decryption/parsing failed for JWT claim: %s", e)
            return None