import os
from fastmcp import FastMCP
import logging

# Import core logic from galaxy.py
from app.bioblend_server.galaxy import get_galaxy_information, GalaxyClient
from app.log_setup import configure_logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
configure_logging()
logger = logging.getLogger("fastmcp_bioblend_server")

# Initialize FastMCP server
# You can set environment variables for Galaxy URL and API Key
GALAXY_URL = os.environ.get("GALAXY_URL", "http://localhost:8080") # Provide a default or raise error if not set
GALAXY_API_KEY = os.environ.get("GALAXY_API_KEY")

if not GALAXY_API_KEY:
    logger.warning("GALAXY_API_KEY environment variable is not set. GalaxyClient functionality may fail.")


bioblend_app = FastMCP(
                        name="galaxyTools",
                        instructions="Provides tools and resources for interacting with Galaxy instances via BioBlend. "
                                    "Tools allow querying Galaxy information and retrieving any sort of information on a galaxy instance "
                                    "Make sure to specify 'tool', 'dataset', or 'workflow' for 'query_type'. "
                                    "Use 'entity_id' only if the user explicitly provides an ID."
                    )



@bioblend_app.tool()
async def get_galaxy_information_tool(
    query: str,
    query_type: str,
    entity_id: str = None
) -> dict:
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
    # Directly call the original function from galaxy.py
    try:
        result = await get_galaxy_information(query=query, query_type=query_type, entity_id=entity_id)
        return result
    except Exception as e:
        logger.error(f"Error in get_galaxy_information_tool: {e}", exc_info=True)
        return f"An error occurred while fetching Galaxy information: {str(e)}"


@bioblend_app.resource("galaxy://whoami")
def get_galaxy_whoami() -> dict:
    """
    Retrieves the current user's details from the Galaxy instance.
    """
    logger.info("Calling get_galaxy_whoami")
    try:
        galaxy_client = GalaxyClient()
        return galaxy_client.whoami()
    except Exception as e:
        logger.error(f"Error in get_galaxy_whoami: {e}", exc_info=True)
        return {"error": f"Failed to retrieve user details: {str(e)}"}

# TODO: Tool execution handler, lets start simple with just the tools
#  For tools there needs to be a state input builder, that builds the state from the 
# io_details of the tools and then use that structure to execute a file from that.
# def execute_tool(): # Starting simple with the tool.