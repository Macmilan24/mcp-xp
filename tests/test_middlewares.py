import os
import jwt
import json
import pytest
import redis
import logging
from datetime import datetime
from sys import path
path.append(".")

from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient as FastAPITestClient
from fastmcp.server.middleware import MiddlewareContext
from unittest.mock import Mock, patch, MagicMock
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_429_TOO_MANY_REQUESTS

from app.log_setup import configure_logging
from app.api.middleware import JWTGalaxyKeyMiddleware, RateLimiterMiddleware
from app.context import current_api_key
from app.bioblend_server.utils import current_api_key_server
from app.bioblend_server.utils import JWTGalaxyKeyMiddleware as FastMCPJWTGalaxyKeyMiddleware



# Configure test logging
configure_logging()


@pytest.fixture
def jwt_logger():
    return logging.getLogger("TestJWTGalaxyKeyMiddleware")

@pytest.fixture
def rate_logger():
    return logging.getLogger("TestRateLimiterMiddleware")

@pytest.fixture
def mcp_jwt_logger():
    return logging.getLogger("TestFastMCPJWTGalaxyKeyMiddleware")

@pytest.fixture
def fernet_secret():
    """Fixture for Fernet secret key."""
    return Fernet.generate_key()


@pytest.fixture
def fernet(fernet_secret):
    """Fixture for Fernet instance."""
    return Fernet(fernet_secret)


@pytest.fixture
def test_app():
    """Basic FastAPI app for testing middlewares."""
    app = FastAPI()
    
    @app.get("/")
    async def root():
        return {"message": "Hello World", "api_key": current_api_key.get(None)}
       
    @app.get("/protected")
    async def protected():
        return {"message": "Protected endpoint", "api_key": current_api_key.get(None)}
    
    @app.get("/health")
    async def health():
        return {"status": "ok"}
    
    return app
class TestJWTGalaxyKeyMiddleware:
    """Unit tests for JWTGalaxyKeyMiddleware."""
    
    @pytest.fixture
    def app_with_jwt_middleware(self, test_app, fernet_secret,jwt_logger):
        """App with JWT middleware added."""
        jwt_logger.debug("Setting up app_with_jwt_middleware")
        os.environ["SECRET_KEY"] = fernet_secret.decode()  # Set env for middleware init
        test_app.add_middleware(JWTGalaxyKeyMiddleware)
        client = FastAPITestClient(test_app)
        jwt_logger.debug("app_with_jwt_middleware setup complete")
        yield client
        del os.environ["SECRET_KEY"]
        jwt_logger.debug("app_with_jwt_middleware teardown complete")
    
    def test_public_paths_allowed_without_auth(self, app_with_jwt_middleware, caplog, jwt_logger):
        """Test that public paths are allowed without authorization."""
        jwt_logger.info("TEST: test_public_paths_allowed_without_auth starting")
        caplog.set_level(logging.INFO)
        jwt_logger.debug("Making request to public path /")
        response = app_with_jwt_middleware.get("/")
        assert response.status_code == 200
        assert "api_key" in response.json()
        jwt_logger.info("TEST: test_public_paths_allowed_without_auth PASSED")

    def test_missing_authorization_header(self, app_with_jwt_middleware, caplog, jwt_logger):
        """Test 401 response for missing Authorization header."""
        jwt_logger.info("TEST: test_missing_authorization_header starting")
        caplog.set_level(logging.ERROR)
        jwt_logger.debug("Making request without Authorization header to /protected")
        response = app_with_jwt_middleware.get("/protected")
        assert response.status_code == HTTP_401_UNAUTHORIZED
        assert response.json()["detail"] == "Authorization header with Bearer token is required."
        jwt_logger.info("TEST: test_missing_authorization_header PASSED")

    def test_invalid_jwt_token(self, app_with_jwt_middleware, caplog, jwt_logger):
        """Test 401 response for invalid JWT token."""
        jwt_logger.info("TEST: test_invalid_jwt_token starting")
        caplog.set_level(logging.ERROR)
        invalid_token = "invalid.token.here"
        jwt_logger.debug(f"Making request with invalid token: {invalid_token} to /protected")
        response = app_with_jwt_middleware.get(
            "/protected",
            headers={"Authorization": f"Bearer {invalid_token}"}
        )
        assert response.status_code == HTTP_401_UNAUTHORIZED
        assert "Invalid JWT" in response.json()["detail"]
        assert "Invalid JWT" in caplog.text
        jwt_logger.info("TEST: test_invalid_jwt_token PASSED")

    def test_jwt_missing_galaxy_api_token_claim(self, app_with_jwt_middleware, caplog, jwt_logger):
        """Test 401 response for JWT missing the required claim."""
        jwt_logger.info("TEST: test_jwt_missing_galaxy_api_token_claim starting")
        caplog.set_level(logging.ERROR)
        payload = {"other_claim": "value"}
        token = jwt.encode(payload, key=None, algorithm="none")
        jwt_logger.debug(f"Making request with missing claim token: {token} to /protected")
        response = app_with_jwt_middleware.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == HTTP_401_UNAUTHORIZED
        assert "JWT missing required claim" in response.json()["detail"]
        jwt_logger.info("TEST: test_jwt_missing_galaxy_api_token_claim PASSED")

    def test_jwt_with_empty_galaxy_api_token_claim(self, app_with_jwt_middleware, caplog, jwt_logger):
        """Test 401 response for empty galaxy_api_token claim."""
        jwt_logger.info("TEST: test_jwt_with_empty_galaxy_api_token_claim starting")
        caplog.set_level(logging.ERROR)
        payload = {"galaxy_api_token": ""}
        token = jwt.encode(payload, key=None, algorithm="none")
        jwt_logger.debug(f"Making request with empty claim token: {token} to /protected")
        response = app_with_jwt_middleware.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == HTTP_401_UNAUTHORIZED
        assert "API key claim is empty." == response.json()["detail"]
        jwt_logger.info("TEST: test_jwt_with_empty_galaxy_api_token_claim PASSED")

    def test_jwt_with_raw_api_key_claim(self, app_with_jwt_middleware, caplog,jwt_logger):
        """Test successful handling of raw API key in claim."""
        jwt_logger.info("TEST: test_jwt_with_raw_api_key_claim starting")
        caplog.set_level(logging.INFO)
        raw_apikey = "test-api-key-123"
        payload = {"galaxy_api_token": raw_apikey}
        token = jwt.encode(payload, key=None, algorithm="none")
        jwt_logger.debug(f"Making request with raw apikey token: {token} to /protected")
        response = app_with_jwt_middleware.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        assert response.json()["api_key"] == raw_apikey
        jwt_logger.info("TEST: test_jwt_with_raw_api_key_claim PASSED")

    def test_jwt_with_encrypted_api_key_claim(self, fernet, app_with_jwt_middleware, caplog, jwt_logger):
        """Test successful decryption of encrypted API key in claim."""
        jwt_logger.info("TEST: test_jwt_with_encrypted_api_key_claim starting")
        caplog.set_level(logging.INFO)
        raw_apikey = "test-encrypted-api-key-456"
        encrypted_payload = json.dumps({"apikey": raw_apikey}).encode()
        encrypted_token = fernet.encrypt(encrypted_payload)
        payload = {"galaxy_api_token": encrypted_token.decode()}
        token = jwt.encode(payload, key=None, algorithm="none")
        jwt_logger.debug(f"Making request with encrypted apikey token: {token} to /protected")
        response = app_with_jwt_middleware.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        assert response.json()["api_key"] == raw_apikey
        jwt_logger.info("TEST: test_jwt_with_encrypted_api_key_claim PASSED")

    def test_jwt_with_invalid_encrypted_claim_falls_back_to_raw(self, app_with_jwt_middleware, caplog, jwt_logger):
        """Test fallback to raw claim when decryption fails."""
        jwt_logger.info("TEST: test_jwt_with_invalid_encrypted_claim_falls_back_to_raw starting")
        caplog.set_level(logging.INFO)
        invalid_encrypted = "invalid-encrypted-token"
        payload = {"galaxy_api_token": invalid_encrypted}
        token = jwt.encode(payload, key=None, algorithm="none")
        jwt_logger.debug(f"Making request with invalid encrypted token: {token} to /protected")
        response = app_with_jwt_middleware.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        assert response.json()["api_key"] == invalid_encrypted
        jwt_logger.info("TEST: test_jwt_with_invalid_encrypted_claim_falls_back_to_raw PASSED")


class TestRateLimiterMiddleware:
    """Unit tests for RateLimiterMiddleware."""
    
    
    @pytest.fixture
    def mock_redis_client(self,rate_logger):
        """Mock Redis client."""
        rate_logger.debug("Creating mock_redis_client")
        mock_redis = Mock(spec=redis.Redis)
        mock_redis.pipeline.return_value = MagicMock()
        rate_logger.debug("mock_redis_client setup complete")
        return mock_redis
    
    @pytest.fixture
    def app_with_rate_limiter(self, test_app, mock_redis_client, rate_logger):
        """App with RateLimiter middleware added."""
        rate_logger.debug("Setting up app_with_rate_limiter")
        test_app.add_middleware(RateLimiterMiddleware, redis_client=mock_redis_client)
        client = FastAPITestClient(test_app)
        rate_logger.debug("app_with_rate_limiter setup complete")
        return client
    
    def test_skip_rate_limiting_for_health_paths(self, app_with_rate_limiter, caplog,rate_logger):
        """Test that rate limiting is skipped for health check paths."""
        rate_logger.info("TEST: test_skip_rate_limiting_for_health_paths starting")
        caplog.set_level(logging.INFO)
        rate_logger.debug("Making request to /health")
        response = app_with_rate_limiter.get("/health")
        assert response.status_code == 200
        rate_logger.info("TEST: test_skip_rate_limiting_for_health_paths PASSED")

    def test_no_api_key_skips_rate_limiting(self, app_with_rate_limiter, mock_redis_client, caplog, rate_logger):
        """Test that requests without API key skip rate limiting."""
        rate_logger.info("TEST: test_no_api_key_skips_rate_limiting starting")
        caplog.set_level(logging.INFO)
        with patch.object(mock_redis_client, 'pipeline') as mock_pipeline:
            rate_logger.debug("Making request without API key to /")
            response = app_with_rate_limiter.get("/")
            assert response.status_code == 200
            mock_pipeline.assert_not_called()  # No Redis interaction
        rate_logger.info("TEST: test_no_api_key_skips_rate_limiting PASSED")

    @patch('time.time')
    def test_rate_limit_exceeded_for_endpoint(self, mock_time, app_with_rate_limiter, mock_redis_client, caplog, rate_logger):
        """Test 429 response when endpoint rate limit is exceeded."""
        rate_logger.info("TEST: test_rate_limit_exceeded_for_endpoint starting")
        caplog.set_level(logging.ERROR)
        mock_time.return_value = 1000

        pipeline_mock = MagicMock()
        pipeline_mock.execute.return_value = [31, 1, 1, 1, 1, 1]
        mock_redis_client.pipeline.return_value = pipeline_mock
        
        rate_logger.debug("Making request to /invocations/ with exceeded endpoint limit")
        response = app_with_rate_limiter.get(
            "/invocations/",
            headers={"Authorization": "Bearer test-key"}
        )
        assert response.status_code == HTTP_429_TOO_MANY_REQUESTS
        assert response.json()["limit_type"] == "endpoint"
        assert int(response.headers["X-RateLimit-Limit"]) == 30
        assert "endpoint" in caplog.text
        rate_logger.info("TEST: test_rate_limit_exceeded_for_endpoint PASSED")

    @patch('time.time')
    def test_rate_limit_exceeded_for_user(self, mock_time, app_with_rate_limiter, mock_redis_client, caplog, rate_logger):
        """Test 429 response when user global rate limit is exceeded."""
        rate_logger.info("TEST: test_rate_limit_exceeded_for_user starting")
        caplog.set_level(logging.ERROR)
        mock_time.return_value = 1000
        # endpoint ok (1), user=101 >100
        pipeline_mock = MagicMock()
        pipeline_mock.execute.return_value = [1, 1, 101, 1, 1, 1]
        mock_redis_client.pipeline.return_value = pipeline_mock
        
        rate_logger.debug("Making request to /protected with exceeded user limit")
        response = app_with_rate_limiter.get(
            "/protected",
            headers={"Authorization": "Bearer test-user-key"}
        )
        assert response.status_code == HTTP_429_TOO_MANY_REQUESTS
        assert response.json()["limit_type"] == "user"
        assert int(response.headers["X-RateLimit-Limit"]) == 100
        assert "user" in caplog.text
        rate_logger.info("TEST: test_rate_limit_exceeded_for_user PASSED")

    @patch('time.time')
    def test_rate_limit_exceeded_for_global(self, mock_time, app_with_rate_limiter, mock_redis_client, caplog, rate_logger):
        """Test 429 response when global rate limit is exceeded."""
        rate_logger.info("TEST: test_rate_limit_exceeded_for_global starting")
        caplog.set_level(logging.ERROR)
        mock_time.return_value = 1000
        pipeline_mock = MagicMock()
        pipeline_mock.execute.return_value = [1, 1, 1, 1, 2001, 1]
        mock_redis_client.pipeline.return_value = pipeline_mock
        
        rate_logger.debug("Making request to /protected with exceeded global limit")
        response = app_with_rate_limiter.get(
            "/protected",
            headers={"Authorization": "Bearer test-global-key"}
        )
        assert response.status_code == HTTP_429_TOO_MANY_REQUESTS
        assert response.json()["limit_type"] == "global"
        assert int(response.headers["X-RateLimit-Limit"]) == 2000
        assert "global" in caplog.text
        rate_logger.info("TEST: test_rate_limit_exceeded_for_global PASSED")

    @patch('time.time')
    def test_within_rate_limits_adds_headers(self, mock_time, app_with_rate_limiter, mock_redis_client, caplog, rate_logger):
        """Test that successful requests add rate limit headers."""
        rate_logger.info("TEST: test_within_rate_limits_adds_headers starting")
        caplog.set_level(logging.INFO)
        mock_time.return_value = 1000
        # All counters within limits
        pipeline_mock = MagicMock()
        pipeline_mock.execute.return_value = [5, 1, 10, 1, 500, 1]
        mock_redis_client.pipeline.return_value = pipeline_mock
        
        rate_logger.debug("Making request to /protected within limits")
        response = app_with_rate_limiter.get(
            "/protected",
            headers={"Authorization": "Bearer test-within-limit-key"}
        )
        assert response.status_code == 200
        assert int(response.headers["X-RateLimit-Limit"]) == 100  # Default limit
        assert int(response.headers["X-RateLimit-Remaining"]) == 95
        assert "X-RateLimit-Reset" in response.headers
        rate_logger.info("TEST: test_within_rate_limits_adds_headers PASSED")

    @patch('time.time')
    def test_redis_error_allows_request(self, mock_time, app_with_rate_limiter, mock_redis_client, caplog, rate_logger):
        """Test that Redis errors allow the request to proceed."""
        rate_logger.info("TEST: test_redis_error_allows_request starting")
        caplog.set_level(logging.ERROR)
        mock_time.return_value = 1000
        mock_redis_client.pipeline.side_effect = redis.RedisError("Connection error")
        
        rate_logger.debug("Making request to /protected with Redis error")
        response = app_with_rate_limiter.get(
            "/protected",
            headers={"Authorization": "Bearer test-redis-error-key"}
        )
        assert response.status_code == 200  # Proceeds despite error
        assert "Connection error" in caplog.text
        rate_logger.info("TEST: test_redis_error_allows_request PASSED")

    @pytest.mark.asyncio
    async def test_mark_user_active(self, mock_redis_client, caplog, rate_logger):
        """Test marking user as active in Redis sorted set."""
        rate_logger.info("TEST: test_mark_user_active starting")
        caplog.set_level(logging.INFO)
        middleware = RateLimiterMiddleware(FastAPI(), redis_client = mock_redis_client)
        rate_logger.debug("Marking user active: test-active-key")
        await middleware._mark_user_active("test-active-key")
        mock_redis_client.zadd.assert_called_once_with(
            "rate_limit:active_users_last_10_minutes",
            {"test-active-key": pytest.approx(datetime.now().timestamp())}
        )
        rate_logger.info("TEST: test_mark_user_active PASSED")
        
        
class TestFastMCPJWTGalaxyKeyMiddleware:
    """Unit tests for FastMCP JWTGalaxyKeyMiddleware."""
           
    @pytest.fixture
    def middleware(self, fernet_secret, mcp_jwt_logger):
        """Fixture for JWT middleware instance."""
        mcp_jwt_logger.debug("Setting up middleware")
        os.environ["SECRET_KEY"] = fernet_secret.decode()
        middleware_instance = FastMCPJWTGalaxyKeyMiddleware()
        mcp_jwt_logger.debug("middleware setup complete")
        yield middleware_instance
        del os.environ["SECRET_KEY"]
        mcp_jwt_logger.debug("middleware teardown complete")
    
    @pytest.fixture
    def mock_context(self):
        """Mock MiddlewareContext."""
        return Mock(spec=MiddlewareContext)
    
    @pytest.fixture
    def mock_call_next(self):
        """Mock CallNext that returns a response including the current API key."""
        async def mock_next(context):
            return {"message": "Protected endpoint", "api_key": current_api_key_server.get(None)}
        return MagicMock(wraps=mock_next)
    
    @pytest.mark.asyncio
    async def test_missing_authorization_header(self, middleware, mock_context, mock_call_next, caplog, mcp_jwt_logger):
        """Test unauthorized response for missing Authorization header."""
        mcp_jwt_logger.info("TEST: test_missing_authorization_header starting")
        caplog.set_level(logging.ERROR)
        with patch('app.bioblend_server.utils.get_http_headers') as mock_headers:
            mock_headers.return_value = {}
            result = await middleware.on_request(mock_context, mock_call_next)
            assert result == {"error": "Unauthorized"}
            mock_call_next.assert_not_called()
            assert "Authorization header with Bearer token is required" in caplog.text
        mcp_jwt_logger.info("TEST: test_missing_authorization_header PASSED")

    @pytest.mark.asyncio
    async def test_invalid_jwt_token(self, middleware, mock_context, mock_call_next, caplog, mcp_jwt_logger):
        """Test unauthorized response for invalid JWT token."""
        mcp_jwt_logger.info("TEST: test_invalid_jwt_token starting")
        caplog.set_level(logging.ERROR)
        invalid_token = "invalid.token.here"
        with patch('app.bioblend_server.utils.get_http_headers') as mock_headers:
            mock_headers.return_value = {"Authorization": f"Bearer {invalid_token}"}
            result = await middleware.on_request(mock_context, mock_call_next)
            assert result == {"error": "Unauthorized"}
            mock_call_next.assert_not_called()
            assert "Invalid JWT" in caplog.text
        mcp_jwt_logger.info("TEST: test_invalid_jwt_token PASSED")

    @pytest.mark.asyncio
    async def test_jwt_missing_galaxy_api_token_claim(self, middleware, mock_context, mock_call_next, caplog, mcp_jwt_logger):
        """Test unauthorized response for JWT missing the required claim."""
        mcp_jwt_logger.info("TEST: test_jwt_missing_galaxy_api_token_claim starting")
        caplog.set_level(logging.ERROR)
        payload = {"other_claim": "value"}
        token = jwt.encode(payload, key=None, algorithm="none")
        with patch('app.bioblend_server.utils.get_http_headers') as mock_headers:
            mock_headers.return_value = {"Authorization": f"Bearer {token}"}
            result = await middleware.on_request(mock_context, mock_call_next)
            assert result == {"error": "Unauthorized"}
            mock_call_next.assert_not_called()
            assert "JWT missing API key claim 'galaxy_api_token'" in caplog.text
        mcp_jwt_logger.info("TEST: test_jwt_missing_galaxy_api_token_claim PASSED")

    @pytest.mark.asyncio
    async def test_jwt_with_empty_galaxy_api_token_claim(self, middleware, mock_context, mock_call_next, caplog, mcp_jwt_logger):
        """Test unauthorized response for empty galaxy_api_token claim."""
        mcp_jwt_logger.info("TEST: test_jwt_with_empty_galaxy_api_token_claim starting")
        caplog.set_level(logging.ERROR)
        payload = {"galaxy_api_token": ""}
        token = jwt.encode(payload, key=None, algorithm="none")
        with patch('app.bioblend_server.utils.get_http_headers') as mock_headers:
            mock_headers.return_value = {"Authorization": f"Bearer {token}"}
            result = await middleware.on_request(mock_context, mock_call_next)
            assert result == {"error": "Unauthorized"}
            mock_call_next.assert_not_called()
            assert "Empty API key claim" in caplog.text
        mcp_jwt_logger.info("TEST: test_jwt_with_empty_galaxy_api_token_claim PASSED")

    @pytest.mark.asyncio
    async def test_jwt_with_encrypted_api_key_claim(self, fernet, middleware, mock_context, mock_call_next, caplog, mcp_jwt_logger):
        """Test successful decryption of encrypted API key in claim."""
        mcp_jwt_logger.info("TEST: test_jwt_with_encrypted_api_key_claim starting")
        caplog.set_level(logging.INFO)
        raw_apikey = "test-encrypted-api-key-456"
        encrypted_payload = json.dumps({"apikey": raw_apikey}).encode()
        encrypted_token = fernet.encrypt(encrypted_payload)
        payload = {"galaxy_api_token": encrypted_token.decode()}
        token = jwt.encode(payload, key=None, algorithm="none")
        with patch('app.bioblend_server.utils.get_http_headers') as mock_headers:
            mock_headers.return_value = {"Authorization": f"Bearer {token}"}
            result = await middleware.on_request(mock_context, mock_call_next)
            
            mock_call_next.assert_called_once()
            assert result["message"] == "Protected endpoint"
            assert result["api_key"] == raw_apikey
            assert "Incoming request to MCP server validated." in caplog.text
        mcp_jwt_logger.info("TEST: test_jwt_with_encrypted_api_key_claim PASSED")