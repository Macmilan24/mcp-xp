from sys import path
path.append(".")

import pytest
import asyncio
import json
from unittest.mock import MagicMock, patch, AsyncMock
from app.orchestration.invocation_cache import InvocationCache
from app.orchestration.invocation_tasks import InvocationBackgroundTasks
from app.orchestration.utils import TTLiveConfig

@pytest.fixture
def mock_redis():
    return MagicMock()

@pytest.fixture
def cache_instance(mock_redis):
    """Fixture to provide an instance of InvocationCache with a mocked Redis client."""
    return InvocationCache(mock_redis)

@pytest.fixture
def mock_cache():
    return MagicMock()

@pytest.fixture
def background_tasks(mock_redis, mock_cache):
    return InvocationBackgroundTasks(cache=mock_cache, redis_client=mock_redis)

class TestBackgroundClass:
    @pytest.mark.asyncio
    async def test_start_cache_warming_task(self, background_tasks, caplog):
        with patch.object(background_tasks, 'warm_all_user_caches', new_callable=AsyncMock) as mock_warm:
            mock_warm.return_value = None
            task = asyncio.create_task(background_tasks.start_cache_warming_task())
            await asyncio.sleep(0.1)  # Allow task to run briefly
            task.cancel()
            assert mock_warm.called
            assert "Error in cache warming task" not in caplog.text

    @pytest.mark.asyncio
    async def test_warm_all_user_caches(self, background_tasks, mock_redis):
        mock_redis.zrange.return_value = ["token1", "token2"]
        with patch.object(background_tasks, 'warm_user_cache', new_callable=AsyncMock) as mock_warm:
            mock_warm.return_value = None
            await background_tasks.warm_all_user_caches()
            assert mock_warm.call_count == 2

    @pytest.mark.asyncio
    async def test_get_active_users(self, background_tasks, mock_redis):
        mock_redis.zrange.return_value = ["user1", "user2"]
        mock_redis.zremrangebyscore.return_value = 1
        active_users = await background_tasks.get_active_users()
        assert active_users == ["user1", "user2"]
        assert mock_redis.zremrangebyscore.called

    @pytest.mark.asyncio
    async def test_decrypt_api_token_success(self, background_tasks):
        with patch('app.orchestration.invocation_tasks.Fernet') as mock_fernet:
            mock_instance = mock_fernet.return_value
            mock_instance.decrypt.return_value = json.dumps({"apikey": "decrypted_key"}).encode()
            result = await background_tasks._decrypt_api_token("encrypted_token")
            assert result == "decrypted_key"

    @pytest.mark.asyncio
    async def test_decrypt_api_token_failure(self, background_tasks):
        with patch('app.orchestration.invocation_tasks.Fernet') as mock_fernet:
            mock_instance = mock_fernet.return_value
            mock_instance.decrypt.side_effect = Exception("Invalid token")
            result = await background_tasks._decrypt_api_token("invalid_token")
            assert result is None


    @pytest.mark.asyncio
    async def test_fetch_workflows_safely_success(self, background_tasks):
        mock_gi = MagicMock()
        mock_gi.workflows.get_workflows.return_value = [
            {"id": "wf1", "name": "Workflow1"},
            {"id": "wf2", "name": "Workflow2"}
        ]
        mock_gi.workflows.show_workflow.side_effect = [
            {"id": "wf1", "name": "Workflow1", "annotation": "desc1", "tags": []},
            {"id": "wf2", "name": "Workflow2", "annotation": "desc2", "tags": []}
        ]

        mock_workflow_manager = MagicMock()
        mock_workflow_manager.gi_object.gi = mock_gi

        with patch('app.bioblend_server.executor.workflow_manager.WorkflowManager', return_value=mock_workflow_manager):
            result = await background_tasks.fetch_workflows_safely(mock_workflow_manager, fetch_details=True)
            assert len(result) == 2
            assert all("id" in wf and "name" in wf for wf in result)

    @pytest.mark.asyncio
    async def test_build_invocation_workflow_mapping(self, background_tasks):
        mock_gi = MagicMock()
        mock_gi.invocations.get_invocations.side_effect = [
            [{"id": "inv1", "workflow_id": "wf1"}],
            [{"id": "inv2", "workflow_id": "wf2"}]
        ]

        mock_workflow_manager = MagicMock()
        mock_workflow_manager.gi_object.gi = mock_gi

        with patch('app.bioblend_server.executor.workflow_manager.WorkflowManager', return_value=mock_workflow_manager):
            workflows = [
                {"id": "wf1", "name": "Workflow1"},
                {"id": "wf2", "name": "Workflow2"},
            ]
            mapping, invocations = await background_tasks.build_invocation_workflow_mapping(
                mock_workflow_manager, workflows
            )
            assert len(mapping) == 2
            assert len(invocations) == 2

    @pytest.mark.asyncio
    async def test_cleanup_expired_cache_entries(self, background_tasks, mock_redis):
        """ Test background expired cache entries """
        mock_redis.zremrangebyscore.return_value = 1
        await background_tasks.cleanup_expired_cache_entries()
        assert mock_redis.zremrangebyscore.called


class TestInvocationCache:
    @pytest.mark.asyncio
    async def test_get_workflows_cache_hit(self, cache_instance, mock_redis):
        """Test retrieving workflows from cache when data exists."""
        username = "test_user"
        expected_data = [{"id": "wf1"}]
        mock_redis.get.return_value = json.dumps(expected_data).encode()

        result = await cache_instance.get_workflows_cache(username)

        assert result == expected_data
        mock_redis.get.assert_called_once_with(f"workflows:{username}")


    @pytest.mark.asyncio
    async def test_get_workflows_cache_miss(self, cache_instance, mock_redis):
        """Test retrieving workflows from cache when no data exists."""
        username = "test_user"
        mock_redis.get.return_value = None

        result = await cache_instance.get_workflows_cache(username)

        assert result is None
        mock_redis.get.assert_called_once_with(f"workflows:{username}")


    @pytest.mark.asyncio
    async def test_set_workflows_cache_success(self, cache_instance, mock_redis):
        """Test setting workflows cache successfully."""
        username = "test_user"
        workflows = [{"id": "wf1"}]
        ttl = 3600

        await cache_instance.set_workflows_cache(username, workflows, ttl)

        mock_redis.setex.assert_called_once_with(
            f"workflows:{username}", ttl, json.dumps(workflows)
        )


    @pytest.mark.asyncio
    async def test_get_invocation_workflow_mapping_hit(self, cache_instance, mock_redis):
        """Test retrieving invocation workflow mapping when data exists."""
        username = "test_user"
        expected_data = {"inv1": {"workflow_name": "wf1", "workflow_id": "wf_id1"}}
        mock_redis.hgetall.return_value = {
            b"inv1": json.dumps({"workflow_name": "wf1", "workflow_id": "wf_id1"}).encode()
        }

        result = await cache_instance.get_invocation_workflow_mapping(username)

        assert result == expected_data
        mock_redis.hgetall.assert_called_once_with(f"invocation_workflow_map:{username}")

    @pytest.mark.asyncio
    async def test_get_invocation_workflow_mapping_miss(self, cache_instance, mock_redis):
        """Test retrieving invocation workflow mapping when no data exists."""
        username = "test_user"
        mock_redis.hgetall.return_value = {}

        result = await cache_instance.get_invocation_workflow_mapping(username)

        assert result == {}
        mock_redis.hgetall.assert_called_once_with(f"invocation_workflow_map:{username}")

    @pytest.mark.asyncio
    async def test_set_invocation_workflow_mapping_success(self, cache_instance, mock_redis):
        """Test setting invocation workflow mapping successfully."""
        username = "test_user"
        mapping = {"inv1": {"workflow_name": "wf1", "workflow_id": "wf_id1"}}
        ttl = 60
        mock_pipeline = MagicMock()
        mock_redis.pipeline.return_value = mock_pipeline
        mock_pipeline.delete.return_value = mock_pipeline
        mock_pipeline.hset.return_value = mock_pipeline
        mock_pipeline.expire.return_value = mock_pipeline

        await cache_instance.set_invocation_workflow_mapping(username, mapping, ttl)

        mock_redis.pipeline.assert_called_once()
        mock_pipeline.delete.assert_called_once_with(f"invocation_workflow_map:{username}")
        mock_pipeline.hset.assert_called_once_with(
            name=f"invocation_workflow_map:{username}",
            mapping={"inv1": json.dumps(mapping["inv1"])},
        )
        mock_pipeline.expire.assert_called_once_with(f"invocation_workflow_map:{username}", ttl)
        mock_pipeline.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_response_cache_hit(self, cache_instance, mock_redis):
        """Test retrieving response cache when data exists."""
        username = "test_user"
        workflow_id = "wf1"
        history_id = "hist1"
        expected_data = {"response": "data"}
        mock_redis.get.return_value = json.dumps(expected_data).encode()

        result = await cache_instance.get_response_cache(username, workflow_id, history_id)

        assert result == expected_data
        mock_redis.get.assert_called_once_with(
            f"invocations_response:{username}:{workflow_id}:{history_id}"
        )

    @pytest.mark.asyncio
    async def test_get_response_cache_miss(self, cache_instance, mock_redis):
        """Test retrieving response cache when no data exists."""
        username = "test_user"
        workflow_id = "wf1"
        history_id = "hist1"
        mock_redis.get.return_value = None

        result = await cache_instance.get_response_cache(username, workflow_id, history_id)

        assert result is None
        mock_redis.get.assert_called_once_with(
            f"invocations_response:{username}:{workflow_id}:{history_id}"
        )

    @pytest.mark.asyncio
    async def test_set_response_cache_success(self, cache_instance, mock_redis):
        """Test setting response cache successfully."""
        username = "test_user"
        response = {"response": "data"}
        workflow_id = "wf1"
        history_id = "hist1"
        ttl = 10

        await cache_instance.set_response_cache(username, response, workflow_id, history_id, ttl)

        mock_redis.setex.assert_called_once_with(
            f"invocations_response:{username}:{workflow_id}:{history_id}",
            ttl,
            json.dumps(response),
        )

    @pytest.mark.asyncio
    async def test_get_deleted_invocations_success(self, cache_instance, mock_redis):
        """Test retrieving deleted invocations successfully."""
        username = "test_user"
        expected_data = {"inv1", "inv2"}
        mock_redis.smembers.return_value = {b"inv1", b"inv2"}

        result = await cache_instance.get_deleted_invocations(username)

        assert result == expected_data
        mock_redis.smembers.assert_called_once_with(f"deleted_invocations:{username}")


    @pytest.mark.asyncio
    async def test_get_invocations_cache_hit(self, cache_instance, mock_redis):
        """Test retrieving invocations cache when data exists."""
        username = "test_user"
        filters = {"workflow_id": "wf1"}
        expected_data = [{"id": "inv1"}]
        mock_redis.get.return_value = json.dumps(expected_data).encode()

        result = await cache_instance.get_invocations_cache(username, filters)

        assert result == expected_data
        mock_redis.get.assert_called_once_with(f"invocations_raw:{username}:workflow_id:wf1")


    @pytest.mark.asyncio
    async def test_get_invocations_cache_miss(self, cache_instance, mock_redis):
        """Test retrieving invocations cache when no data exists."""
        username = "test_user"
        filters = {"workflow_id": "wf1"}
        mock_redis.get.return_value = None

        result = await cache_instance.get_invocations_cache(username, filters)

        assert result is None
        mock_redis.get.assert_called_once_with(f"invocations_raw:{username}:workflow_id:wf1")


    @pytest.mark.asyncio
    async def test_set_invocations_cache_success(self, cache_instance, mock_redis):
        """Test setting invocations cache successfully."""
        username = "test_user"
        invocations = [{"id": "inv1"}]
        filters = {"workflow_id": "wf1"}
        ttl = 20

        await cache_instance.set_invocations_cache(username, invocations, filters, ttl)

        mock_redis.setex.assert_called_once_with(
            f"invocations_raw:{username}:workflow_id:wf1", ttl, json.dumps(invocations)
        )

    @pytest.mark.asyncio
    async def test_add_to_deleted_invocations_success(self, cache_instance, mock_redis):
        """Test adding invocation IDs to deleted set successfully."""
        username = "test_user"
        invocation_ids = ["inv1", "inv2"]

        await cache_instance.add_to_deleted_invocations(username, invocation_ids)

        mock_redis.sadd.assert_called_once_with(f"deleted_invocations:{username}", *invocation_ids)

    @pytest.mark.asyncio
    async def test_is_duplicate_request_false(self, cache_instance, mock_redis):
        """Test checking for non-duplicate request."""
        username = "test_user"
        request_hash = "hash1"
        mock_redis.exists.return_value = False

        result = await cache_instance.is_duplicate_request(username, request_hash)

        assert result is False
        mock_redis.exists.assert_called_once_with(f"request_dedup:{username}:{request_hash}")
        mock_redis.setex.assert_called_once_with(f"request_dedup:{username}:{request_hash}", 3, "1")


    @pytest.mark.asyncio
    async def test_is_duplicate_request_true(self, cache_instance, mock_redis):
        """Test checking for duplicate request."""
        username = "test_user"
        request_hash = "hash1"
        mock_redis.exists.return_value = True

        result = await cache_instance.is_duplicate_request(username, request_hash)

        assert result is True
        mock_redis.exists.assert_called_once_with(f"request_dedup:{username}:{request_hash}")
        assert not mock_redis.setex.called


    @pytest.mark.asyncio
    async def test_is_duplicate_workflow_request_false(self, cache_instance, mock_redis):
        """Test checking for non-duplicate workflow request."""
        username = "test_user"
        request_hash = "hash1"
        mock_redis.exists.return_value = False

        result = await cache_instance.is_duplicate_workflow_request(username, request_hash)

        assert result is False
        mock_redis.exists.assert_called_once_with(f"workflow_request_dedup:{username}:{request_hash}")
        mock_redis.setex.assert_called_once_with(
            f"workflow_request_dedup:{username}:{request_hash}", TTLiveConfig.DUPLICATE_CHECK.value, "1"
        )


    @pytest.mark.asyncio
    async def test_is_duplicate_workflow_request_true(self, cache_instance, mock_redis):
        """Test checking for duplicate workflow request."""
        username = "test_user"
        request_hash = "hash1"
        mock_redis.exists.return_value = True

        result = await cache_instance.is_duplicate_workflow_request(username, request_hash)

        assert result is True
        mock_redis.exists.assert_called_once_with(f"workflow_request_dedup:{username}:{request_hash}")
        assert not mock_redis.setex.called


    @pytest.mark.asyncio
    async def test_get_invocation_result_hit(self, cache_instance, mock_redis):
        """Test retrieving invocation result when data exists."""
        username = "test_user"
        invocation_id = "inv1"
        expected_data = {"result": "data"}
        mock_redis.get.return_value = json.dumps(expected_data).encode()

        result = await cache_instance.get_invocation_result(username, invocation_id)

        assert result == expected_data
        mock_redis.get.assert_called_once_with(f"invocation_result:{username}:{invocation_id}")


    @pytest.mark.asyncio
    async def test_get_invocation_result_miss(self, cache_instance, mock_redis):
        """Test retrieving invocation result when no data exists."""
        username = "test_user"
        invocation_id = "inv1"
        mock_redis.get.return_value = None

        result = await cache_instance.get_invocation_result(username, invocation_id)

        assert result is None
        mock_redis.get.assert_called_once_with(f"invocation_result:{username}:{invocation_id}")


    @pytest.mark.asyncio
    async def test_set_invocation_result_success(self, cache_instance, mock_redis):
        """Test setting invocation result successfully."""
        username = "test_user"
        invocation_id = "inv1"
        result = {"result": "data"}
        ttl = 86400

        await cache_instance.set_invocation_result(username, invocation_id, result, ttl)

        mock_redis.setex.assert_called_once_with(
            f"invocation_result:{username}:{invocation_id}", ttl, json.dumps(result)
        )


    @pytest.mark.asyncio
    async def test_set_invocation_state_success(self, cache_instance, mock_redis):
        """Test setting invocation state successfully."""
        username = "test_user"
        invocation_id = "inv1"
        state = "Complete"

        with patch.object(cache_instance, "log") as mock_log:
            await cache_instance.set_invocation_state(username, invocation_id, state)
            mock_redis.hset.assert_called_once_with(
                f"invocation_states:{username}", invocation_id, state
            )
            mock_log.info.assert_called_once_with(
                f" new invocation state has been set for user: {username} invocation id: {invocation_id}"
            )


    @pytest.mark.asyncio
    async def test_set_invocation_state_invalid_state(self, cache_instance, mock_redis):
        """Test setting invalid invocation state."""
        username = "test_user"
        invocation_id = "inv1"
        state = "Invalid"

        with patch.object(cache_instance, "log") as mock_log:
            await cache_instance.set_invocation_state(username, invocation_id, state)
            assert not mock_redis.hset.called
            mock_log.error.assert_called_once_with("Error setting invocation state: Invalid state: Invalid")


    @pytest.mark.asyncio
    async def test_get_invocation_state_hit(self, cache_instance, mock_redis):
        """Test retrieving invocation state when data exists."""
        username = "test_user"
        invocation_id = "inv1"
        mock_redis.hget.return_value = b"Complete"

        result = await cache_instance.get_invocation_state(username, invocation_id)

        assert result == "Complete"
        mock_redis.hget.assert_called_once_with(f"invocation_states:{username}", invocation_id)


    @pytest.mark.asyncio
    async def test_get_invocation_state_miss(self, cache_instance, mock_redis):
        """Test retrieving invocation state when no data exists."""
        username = "test_user"
        invocation_id = "inv1"
        mock_redis.hget.return_value = None

        result = await cache_instance.get_invocation_state(username, invocation_id)

        assert result is None
        mock_redis.hget.assert_called_once_with(f"invocation_states:{username}", invocation_id)


    @pytest.mark.asyncio
    async def test_delete_invocation_state_success(self, cache_instance, mock_redis):
        """Test deleting invocation state successfully."""
        username = "test_user"
        invocation_id = "inv1"

        await cache_instance.delete_invocation_state(username, invocation_id)

        mock_redis.hdel.assert_called_once_with(f"invocation_states:{username}", invocation_id)


    @pytest.mark.asyncio
    async def test_add_deleted_workflows_success(self, cache_instance, mock_redis):
        """Test adding deleted workflow IDs successfully."""
        username = "test_user"
        workflow_ids = ["wf1", "wf2"]
        mock_pipeline = MagicMock()
        mock_redis.pipeline.return_value = mock_pipeline
        mock_pipeline.sadd.return_value = mock_pipeline
        mock_pipeline.expire.return_value = mock_pipeline

        await cache_instance.add_deleted_workflows(username, workflow_ids)

        mock_redis.pipeline.assert_called_once()
        mock_pipeline.sadd.assert_called_once_with(f"deleted_workflows:{username}", *workflow_ids)
        mock_pipeline.expire.assert_called_once_with(f"deleted_workflows:{username}", TTLiveConfig.WORKFLOW_CACHE.value)
        mock_pipeline.execute.assert_called_once()


    @pytest.mark.asyncio
    async def test_get_deleted_workflows_success(self, cache_instance, mock_redis):
        """Test retrieving deleted workflows successfully."""
        username = "test_user"
        expected_data = [b"wf1", b"wf2"]
        mock_redis.smembers.return_value = expected_data

        result = await cache_instance.get_deleted_workflows(username)

        assert result == expected_data
        mock_redis.smembers.assert_called_once_with(f"deleted_workflows:{username}")


    @pytest.mark.asyncio
    async def test_clear_deleted_workflows_success(self, cache_instance, mock_redis):
        """Test clearing deleted workflows successfully."""
        username = "test_user"

        await cache_instance.clear_deleted_workflows(username)

        mock_redis.delete.assert_called_once_with(f"deleted_workflows:{username}")