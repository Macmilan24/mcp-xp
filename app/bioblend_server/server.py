import os
from fastmcp import FastMCP
import logging

from app.log_setup import configure_logging
from app.bioblend_server.utils import ExecutorToolResponse, ApiKeyMiddleware, current_api_key_server

from app.bioblend_server.galaxy import GalaxyClient
from app.bioblend_server.informer.informer import GalaxyInformer

from app.bioblend_server.executor.tool_manager import ToolManager
from app.bioblend_server.executor.workflow_manager import WorkflowManager


configure_logging()
logger = logging.getLogger("fastmcp_bioblend_server")


if not os.environ.get("GALAXY_API_KEY"):
    logger.warning("GALAXY_API_KEY environment variable is not set. GalaxyClient functionality may fail.")


bioblend_app = FastMCP(
                        name="galaxyTools",
                        instructions="Provides tools and resources for interacting with Galaxy instances via BioBlend. "
                                    "Tools that allow querying Galaxy information and retrieving any sort of information on a galaxy instance "
                                    "Tools that allow execution of a tools and workflows within a galaxy instance."
                                    "Make sure to specify 'tool', 'dataset', or 'workflow' for 'query_type'. "
                                    "Use 'entity_id' only if the user explicitly provides an ID.",
                        middleware=[ApiKeyMiddleware()]
                    )



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
    logger.info(f"Calling execute_galaxy_tool_workflow with entity='{entity}', name='{name}', entity_id='{entity_id}'")

    if name is None and entity_id is  None:
        raise ValueError("Neither the name or the id is inputted for the execution")
    try:
        # Get current user
        user_api_key = current_api_key_server.get()
        if user_api_key is None:
            raise ValueError("current user api-key is missing")
        
        logger.info( f"current mcp user: ************{user_api_key[-4:]}")
        
        # Create galaxy instance
        galaxy_client = GalaxyClient(user_api_key)

        # Create tool and workflow manager objects
        tool_manager=ToolManager(galaxy_client)
        workflow_manager= WorkflowManager(galaxy_client)

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
    try:
        # Get current user
        user_api_key = current_api_key_server.get()
        if user_api_key is None:
            raise ValueError("current user api-key is missing")
        
        logger.info( f"current mcp user: ************{user_api_key[-4:]}")

        # Create galaxy instances
        galaxy_client = GalaxyClient(user_api_key)

        # Create GalaxyInformer object and execute informer
        informer = await GalaxyInformer.create(galaxy_client=galaxy_client, entity_type=query_type)
        galaxy_response= await informer.get_entity_info(search_query = query, entity_id = entity_id)

        result= galaxy_response.get('response')
        return result
    
    except Exception as e:
        logger.error(f"Error in get_galaxy_information_tool: {e}", exc_info=True)
        return f"An error occurred while fetching Galaxy information: {str(e)}"