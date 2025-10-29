import os
from fastmcp import FastMCP
import logging

from contextlib import asynccontextmanager
from apscheduler.schedulers.background import BackgroundScheduler

from app.log_setup import configure_logging
from app.bioblend_server.utils import ExecutorToolResponse, ApiKeyMiddleware, current_api_key_server

from app.bioblend_server.galaxy import GalaxyClient
from app.bioblend_server.informer.informer import GalaxyInformer

from app.bioblend_server.executor.tool_manager import ToolManager
from app.bioblend_server.executor.workflow_manager import WorkflowManager

from app.bioblend_server.informer.pipeline.tool_pipeline import main as run_tool_pipeline
from app.bioblend_server.informer.pipeline.workflow_pipeline import main as run_workflow_pipeline


configure_logging()
logger = logging.getLogger("fastmcp_bioblend_server")


if not os.environ.get("GALAXY_API_KEY"):
    logger.warning("GALAXY_API_KEY environment variable is not set. GalaxyClient functionality may fail.")


PIPELINE_DATA_DIR = os.path.join(os.path.dirname(__file__), "informer", "pipeline", "data")
PROCESSED_TOOLS_PATH = os.path.join(PIPELINE_DATA_DIR, "processed_tools.json")
PROCESSED_WORKFLOWS_PATH = os.path.join(PIPELINE_DATA_DIR, "processed_workflows.json")

# --- NEW: Create a scheduler instance ---
scheduler = BackgroundScheduler(daemon=True)

def run_pipelines_if_needed():
    """Checks for data files and runs pipelines if they are missing."""
    logger.info("Checking for processed data files on startup...")
    if not os.path.exists(PROCESSED_TOOLS_PATH):
        logger.warning("Processed tools file not found. Running the tool pipeline now...")
        try:
            run_tool_pipeline()
        except Exception as e:
            logger.error(f"Failed to run tool pipeline on startup: {e}", exc_info=True)
    else:
        logger.info("Found existing processed tools file.")

    if not os.path.exists(PROCESSED_WORKFLOWS_PATH):
        logger.warning("Processed workflows file not found. Running the workflow pipeline now...")
        try:
            run_workflow_pipeline()
        except Exception as e:
            logger.error(f"Failed to run workflow pipeline on startup: {e}", exc_info=True)
    else:
        logger.info("Found existing processed workflows file.")

def schedule_monthly_jobs():
    """Schedules the pipelines to run monthly."""
    logger.info("Scheduling monthly data refresh jobs.")
    # We use 'weeks=4' as a simple proxy for 'monthly'
    scheduler.add_job(run_tool_pipeline, 'interval', weeks=4, id='monthly_tool_pipeline', replace_existing=True)
    scheduler.add_job(run_workflow_pipeline, 'interval', weeks=4, id='monthly_workflow_pipeline', replace_existing=True)
    scheduler.start()
    logger.info("Scheduler started. Pipeline jobs are scheduled to run every 4 weeks.")


@asynccontextmanager
async def lifespan(app: FastMCP):
    # --- NEW: This block runs on server startup ---
    run_pipelines_if_needed()
    schedule_monthly_jobs()
    yield
    # --- NEW: This block runs on server shutdown ---
    logger.info("Shutting down the scheduler.")
    scheduler.shutdown()
    
bioblend_app = FastMCP(
                        lifespan=lifespan,
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