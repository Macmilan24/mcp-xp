import os
from fastmcp import FastMCP
import httpx
import logging

from app.log_setup import configure_logging
from app.bioblend_server.utils import (
    ExecutorToolResponse,
    ApiKeyMiddleware,
    current_api_key_server,
)


configure_logging()
logger = logging.getLogger("fastmcp_bioblend_server")

MAIN_APP_URL = os.getenv("MAIN_APP_URL", "http://host.docker.internal:8000")
API_BASE = f"{MAIN_APP_URL}/api"


if not os.environ.get("GALAXY_API_KEY"):
    logger.warning(
        "GALAXY_API_KEY environment variable is not set. GalaxyClient functionality may fail."
    )


bioblend_app = FastMCP(
    name="galaxyTools",
    instructions="Provides tools and resources for interacting with Galaxy instances via BioBlend. "
    "Tools that allow querying Galaxy information and retrieving any sort of information on a galaxy instance "
    "Tools that allow execution of a tools and workflows within a galaxy instance."
    "Make sure to specify 'tool', 'dataset', or 'workflow' for 'query_type'. "
    "Use 'entity_id' only if the user explicitly provides an ID.",
    middleware=[ApiKeyMiddleware()],
)


@bioblend_app.tool()
async def execute_galaxy_tool_workflow(
    entity: str, name: str = None, entity_id: str = None
) -> ExecutorToolResponse:
    """
    Execute a Galaxy tool or workflow and return its metadata.

    Args:
        entity(str): The type of Galaxy entity to execute. Must be either `"tool"` or `"workflow"`.
        name(str, optional): The human-readable name of the tool or workflow. Used to locate the entity if `entity_id` is not provided.
        entity_id(str, optional): The unique Galaxy ID of the tool or workflow. Used to locate the entity if `name` is not provided.

    Returns:
        ExecutorToolResponse
            A structured response containing:
            - Entity type (`tool` or `workflow`)
            - Name
            - Galaxy ID
            - Description/annotation
            - Action link to the execution form endpoint
    """
    logger.info(
        f"Calling execute_galaxy_tool_workflow with entity='{entity}', name='{name}', entity_id='{entity_id}'"
    )

    if name is None and entity_id is None:
        raise ValueError("Neither the name or the id is inputted for the execution")
    try:
        # Get current user
        user_api_key = current_api_key_server.get()
        if user_api_key is None:
            raise ValueError("current user api-key is missing")

        headers = {"USER-API-KEY": user_api_key}
        params = {"name": name} if name else {}
        base_path = f"{API_BASE}/internal/executor"

        logger.info(f"current mcp user: ************{user_api_key[-4:]}")

        if entity == "workflow":
            endpoint_path = (
                f"/workflows/get-by-name" if name else f"/workflows/{entity_id}"
            )
        elif entity == "tool":
            endpoint_path = f"/tools/get-by-name" if name else f"/tools/{entity_id}"
        else:
            raise ValueError(f"Invalid entity type: {entity}")

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{base_path}{endpoint_path}",
                headers=headers,
                params=params,
                timeout=30.0,
            )
            response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
            data = response.json()

        return ExecutorToolResponse(
            entity=entity,
            name=data.get("name"),
            id=data.get("id"),
            description=data.get("description"),
            action_link=f"/api/{entity}s/{data.get('id')}/form",
        )

    except httpx.HTTPStatusError as e:
        logger.error(
            f"Error from internal executor service: {e.response.status_code} - {e.response.text}"
        )
        raise RuntimeError(
            f"Failed to execute Galaxy {entity}: Upstream service returned an error."
        )
    except Exception as e:
        logger.error(f"Error in gateway during execute_galaxy_tool_workflow: {e}")
        raise RuntimeError(f"An unexpected error occurred: {e}")


@bioblend_app.tool()
async def get_galaxy_information_tool(
    query: str, query_type: str, entity_id: str = None
) -> str:
    """
    Fetch detailed information on Galaxy tools, workflows, datasets, and invocations.

    This tool handles all information requests about Galaxy entities, based on
    the `query_type` (tool, workflow, dataset) and the user's `query`.
    Use `entity_id` only when the user's query explicitly includes an ID.

    Args:
        query: The user's query message that needs a response, accompanied by full and detailed contextual information.
        query_type: The type of Galaxy entity the query needs a response for, with one of three values: "tool", "dataset", or "workflow".
                    Select "workflow" for general workflow details and specific workflow invocation details.
        entity_id: Optional parameter. Provide this only when the user's query explicitly includes an ID,
                   allowing retrieval of information by that specific entity ID.

    Returns:
        A string containing the detailed Galaxy information and the response to the user's query.
    """
    logger.info(
        f"Calling get_galaxy_information with query='{query}', query_type='{query_type}', entity_id='{entity_id}'"
    )
    try:
        # Get current user
        user_api_key = current_api_key_server.get()
        if user_api_key is None:
            raise ValueError("Current user API key is missing from context.")

        headers = {"USER-API-KEY": user_api_key}
        payload = {"query": query, "query_type": query_type, "entity_id": entity_id}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE}/internal/informer/get-entity-info",
                headers=headers,
                json=payload,
                timeout=120.0,  # Give the informer more time for potential LLM calls
            )
            response.raise_for_status()
            data = response.json()

        return data.get(
            "response",
            "Error: No 'response' field in the result from the Informer service.",
        )

    except httpx.HTTPStatusError as e:
        logger.error(
            f"Error from internal informer service: {e.response.status_code} - {e.response.text}"
        )
        return f"An error occurred while fetching Galaxy information: The information service is unavailable or returned an error."
    except Exception as e:
        logger.error(f"Error in gateway during get_galaxy_information_tool: {e}")
        return f"An unexpected error occurred in the gateway: {str(e)}"
