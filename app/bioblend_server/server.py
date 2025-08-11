import os
from fastmcp import FastMCP
import logging
from pydantic import BaseModel, Field
from typing import Literal, Optional

# Import core logic from galaxy.py
from app.bioblend_server.galaxy import get_galaxy_information, GalaxyClient
from app.log_setup import configure_logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

configure_logging()
logger = logging.getLogger("fastmcp_bioblend_server")


if not os.environ.get("GALAXY_API_KEY"):
    logger.warning("GALAXY_API_KEY environment variable is not set. GalaxyClient functionality may fail.")


bioblend_app = FastMCP(
                        name="galaxyTools",
                        instructions="Provides tools and resources for interacting with Galaxy instances via BioBlend. "
                                    "Tools allow querying Galaxy information and retrieving any sort of information on a galaxy instance "
                                    "Tools that allow execution of a tools and workflows within a galaxy instance."
                                    "Make sure to specify 'tool', 'dataset', or 'workflow' for 'query_type'. "
                                    "Use 'entity_id' only if the user explicitly provides an ID."
                    )

# Structure for the executor tool to respond with.
class ExecutorToolResponse(BaseModel):
    entity: Literal["tool", "workflow"] = Field(..., title="Entity")
    name: str = Field(..., title="Name")
    id: str = Field(..., title="Id")
    description: Optional[str] = Field(default=None, title="Description")
    action_link: str = Field(..., title="Action Link")


@bioblend_app.tool()
async def execute_galaxy_tool_workflow(
    entity: str,
    name: str = None,
    entity_id: str = None
)-> ExecutorToolResponse : 
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

    from app.bioblend_server.executor.tool_manager import ToolManager
    from app.bioblend_server.executor.workflow_manager import WorkflowManager

    tool_manager=ToolManager()
    workflow_manager= WorkflowManager()

    if name is None and entity_id is  None:
        raise ValueError("Neither the name or the id is inputted for the execution")
    try:
        if entity== "workflow":
            if name:
                workflow_result = workflow_manager.get_worlflow_by_name(name)
            elif entity_id:
                workflow_result = workflow_manager.get_workflow_by_id(entity_id)

            workflow = workflow_manager.gi_object.gi.workflows.show_workflow(workflow_result.id)

            return ExecutorToolResponse(
                entity= "workflow",
                name = workflow.get("name"),
                id = workflow.get("id"),
                description = workflow.get("annotation") or workflow.get("description"),
                action_link = f"/api/workflows/{workflow.get('id')}/form"
            )


        elif entity == "tool":
            if name:
                tool_result = tool_manager.get_tool_by_name(name)
            elif entity_id:
                tool_result = tool_manager.get_tool_by_id(entity_id)

            tool = tool_manager.gi_object.gi.tools.show_tool(tool_result.id)

            return ExecutorToolResponse(
                entity= "tool",
                name = tool.get("name"),
                id = tool.get("id"),
                description = tool.get("description") or tool.get("annotation"),
                action_link = f"/api/tools/{tool.get('id')}/form"
            )

        else:
            raise ValueError(f"Invalid entity type: {entity}")
    except Exception as e:
        logger.error(f"error occured when executing execute_galaxy_tool_workflow: {e}")
        raise RuntimeError(f"Failed to execute Galaxy {entity}: {e}")
    

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


# TODO: Galaxy interaction should be based of off diffrent APIs.

# How wwould multiple histories be handled if multiple user were to query this ar onece. 
# If Diffrent having diffrent APIs and diffrent galaxy account???


# TODO: IDEA
# Pass APis to the MCP as env variable?
# Take APIs as inputs as well.


# TODO: Galaxy interface and galaxy integration issues???