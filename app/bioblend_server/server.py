import os
from fastmcp import FastMCP
import logging
from typing import Literal

from app.log_setup import configure_logging
from app.bioblend_server.utils import JWTGalaxyKeyMiddleware, current_api_key_server

from app.bioblend_server.galaxy import GalaxyClient
from app.bioblend_server.informer.informer import GalaxyInformer

configure_logging()
logger = logging.getLogger("fastmcp_bioblend_server")

if not os.environ.get("GALAXY_API_KEY"):
    logger.warning("GALAXY_API_KEY environment variable is not set. GalaxyClient functionality may fail.")


bioblend_app = FastMCP(
                        name="galaxyTools",
                        instructions="""
                                    You have tools for interacting with a Galaxy instance.
                                    These tools support these distinct capabilities:
                                    
                                    (1) (**get_galaxy_information_tool**) Querying **Galaxy information** to retrieve **accurate and upto date details(information)** about available tools, datasets, or workflows in **Galaxy**. 
                                        Used for tool/workflow recommendations, To respond accurately when asked about the available tools, datasets, or workflows in the galaxy instance.
                                        used only for gathering information to **respond to galaxy related queries**.
                                        
                                    (2) (**execute_galaxy_tool_workflow**) Executing Galaxy operations to run a specific tool or execute a workflow (used only when the user explicitly requests this operation).
                                    
                                    Always keep querying and executing strictly separate in purpose and usage.
                                    """,
                        middleware=[JWTGalaxyKeyMiddleware()]
                    )

@bioblend_app.tool()
async def get_galaxy_information_tool(
    query: str,
    query_type: str,
    entity_id: str = None
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
    logger.info(f"Calling get_galaxy_information with query='{query}', query_type='{query_type}', entity_id='{entity_id}'")
    try:
        # Get current user
        user_api_key = current_api_key_server.get()
        if user_api_key is None:
            raise ValueError("current user api-key is missing")        

        # Create galaxy instances
        galaxy_client = GalaxyClient(user_api_key)
        logger.info( f"current Galaxy MCP server user: {galaxy_client.whoami}")
        # Create GalaxyInformer object and execute informer
        informer = await GalaxyInformer.create(galaxy_client=galaxy_client, entity_type=query_type)
        result = await informer.get_entity_info(search_query = query, entity_id = entity_id)
        return result
    
    except Exception as e:
        logger.error(f"Error in get_galaxy_information_tool: {e}", exc_info=True)
        return f"An error occurred while fetching Galaxy information: {str(e)}"