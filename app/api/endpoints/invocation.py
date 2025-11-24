import re
import os
import tempfile
import redis
import pathlib
import shutil
import asyncio
from anyio.to_thread import run_sync
from typing import Dict, List, Union
import logging
from datetime import datetime
import hashlib
import json

from sys import path
path.append(".")

from fastapi import APIRouter, Path, Query, HTTPException, BackgroundTasks, Response
from fastapi.responses import  FileResponse
from fastapi.concurrency import run_in_threadpool
from bioblend.galaxy.objects.wrappers import HistoryDatasetAssociation, HistoryDatasetCollectionAssociation, Invocation
from starlette.status import HTTP_204_NO_CONTENT

from app.context import current_api_key
from app.bioblend_server.galaxy import GalaxyClient
from app.bioblend_server.executor.workflow_manager import WorkflowManager
from app.api.schemas import invocation, workflow
from app.api.socket_manager import ws_manager, SocketMessageEvent, SocketMessageType
from app.orchestration.invocation_cache import InvocationCache
from app.orchestration.invocation_tasks import InvocationBackgroundTasks
from app.orchestration.utils import NumericLimits

from exceptions import InternalServerErrorException, NotFoundException

# Helper functions and redis instantiation
logger = logging.getLogger("invocation")
redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=os.environ.get("REDIS_PORT"), db=0, decode_responses=True)
invocation_cache = InvocationCache(redis_client)
invocation_background = InvocationBackgroundTasks(cache = invocation_cache, redis_client=redis_client)

# Helper functions
def _rmtree_sync(path: pathlib.Path):
    shutil.rmtree(path, ignore_errors=True)
    
def log_task_error(task: asyncio.Task, *, task_name: str) -> None:
    """Log errors from completed background tasks."""
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        logger.info(f"Background task '{task_name}' was cancelled.")
        return
    
    if exc is not None:
        logger.error(
            f"Background task '{task_name}' failed",
            exc_info=exc,
        )

router = APIRouter()

@router.get(
    "/",
    response_model=invocation.InvocationList,
    summary="List all workflow invocations",
    tags=["Invocation"]
)
async def list_invocations(
    workflow_id: str | None = Query(None, description="Filter by workflow ID"),
    history_id: str | None = Query(None, description="Filter by History ID"),
):
    """
    Retrieves a list of workflow invocations from the Galaxy instance.
    
    Features:
    - Multi-level caching for optimal performance
    - Request deduplication to prevent duplicate processing
    - Parallel processing of data fetching
    - Graceful error handling and partial results
    """
    api_key = current_api_key.get()
    
    # Initialize cache and clients
    try:
        galaxy_client = GalaxyClient(api_key)
        username = galaxy_client.whoami
        # Get deleted invocations list.
        deleted_invocation_ids= await invocation_cache.get_deleted_invocations(username)
        
        # Step 1: Request deduplication check and filter out deleted invocations
        request_hash = _generate_request_hash(api_key, workflow_id, history_id)
        if await invocation_cache.is_duplicate_request(username, request_hash):
            logger.info("Duplicate request detected, serving from cache")
        
        # Step 2: Try to get cached response first
        cached_response = await invocation_cache.get_response_cache(
            username, workflow_id, history_id
        )
        if cached_response:
            logger.info("Filtering and serving response from cache")
            invocations = cached_response.get("invocations", [])
            filtered_cache_response = [
                inv for inv in invocations if inv["id"] not in deleted_invocation_ids
            ]
            response_data = {
                "invocations": filtered_cache_response,
            }

            return invocation.InvocationList(**response_data)
        
        # Step 3: Initialize Galaxy client and workflow manager give it time for new invocations to be registerd under list
        # TODO: sleeping is a temporary solution, find a permanent solution for this.
        await asyncio.sleep(NumericLimits.SHORT_SLEEP)
        workflow_manager = WorkflowManager(galaxy_client)
        
        # Step 4: Fetch data with parallel processing and caching
        invocations_data, workflows_data = await _fetch_core_data(
            invocation_cache, username, workflow_manager, workflow_id, history_id
        )
        
        if not invocations_data:
            logger.warning("No invocations data retrieved")
            return invocation.InvocationList(invocations=[], total_count=0)
        
        # Step 5: Get invocation-workflow mapping (cached or build)
        workflow_mapping, all_invocations = await invocation_background.build_invocation_workflow_mapping(workflow_manager,workflows_data)
        
        if workflow_mapping:
            await invocation_cache.set_invocation_workflow_mapping(username, workflow_mapping)
        if not workflow_id and not history_id:
            if all_invocations:
                await invocation_cache.set_invocations_cache(username, all_invocations, filters={"workflow_id": None, "history_id": None})
        
            # Step 6: Filter out deleted invocations
            invocations_data = [
                inv for inv in all_invocations 
                if inv.get('id') not in deleted_invocation_ids
            ]
        else:
            invocations_data = [
                inv for inv in invocations_data 
                if inv.get('id') not in deleted_invocation_ids
            ]
        logger.info(f"length after filteration {len(invocations_data)}")
        
        # Step 7: Format invocations and filter deleted invocatoins optimized processing
        invocation_list = await _format_invocations(
            username, invocations_data, workflow_mapping
        )
        

        # Step 8: Build response
        response_data = {
            "invocations": [inv.model_dump() for inv in invocation_list],
        }
        
        # Step 9: Cache the response
        await invocation_cache.set_response_cache(
            username, response_data, workflow_id, history_id
        )
        
        logger.info(f"Successfully retrieved {len(invocation_list)} invocations (total: {len(invocations_data)})")
        return invocation.InvocationList(**response_data)
        
    except Exception as e:
        logger.error(f"Error in list_invocations: {e}", exc_info=True)
        # Try to return partial results if possible
        try:
            return await _handle_partial_failure(api_key, workflow_id, history_id)
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
            raise InternalServerErrorException("Failed to list invocations")


async def _fetch_core_data(cache: InvocationCache, username: str, workflow_manager: WorkflowManager, workflow_id: str, history_id: str):
    """Fetch core data (invocations, workflows, deleted set) with parallel processing and caching"""
    
    # Prepare filter dictionary for caching
    filters = {"workflow_id": workflow_id, "history_id": history_id}
    
    async def fetch_invocations():
        """Fetch invocations with caching"""
        cached_invocations = await cache.get_invocations_cache(username, filters)
        if not cached_invocations:
            # Fetch all from Galaxy API based on filters
            if history_id and workflow_id:
                invocations = await run_in_threadpool(
                    workflow_manager.gi_object.gi.invocations.get_invocations,
                    workflow_id=workflow_id,
                    history_id=history_id,
                    limit = NumericLimits.INVOCATION_LIMIT
                )
            elif history_id:
                invocations = await run_in_threadpool(
                    workflow_manager.gi_object.gi.invocations.get_invocations,
                    history_id=history_id,
                    limit = NumericLimits.INVOCATION_LIMIT
                )
            elif workflow_id:
                invocations = await run_in_threadpool(
                    workflow_manager.gi_object.gi.invocations.get_invocations,
                    workflow_id=workflow_id,
                    limit = NumericLimits.INVOCATION_LIMIT
                )
            else:
                invocations = await run_in_threadpool(
                    workflow_manager.gi_object.gi.invocations.get_invocations,
                    limit = NumericLimits.INVOCATION_LIMIT
                )

            # Cache the results
            await cache.set_invocations_cache(username, invocations, filters)
            return invocations
        
        # Have cache, check for new invocations
        parsed_times = [datetime.fromisoformat(inv['create_time']) for inv in cached_invocations if inv.get('create_time')]
        max_create = max(parsed_times) if parsed_times else datetime.min
        cached_ids = {inv['id'] for inv in cached_invocations}
        
        # Determine filter params
        filter_params = {}
        if history_id:
            filter_params['history_id'] = history_id
        if workflow_id:
            filter_params['workflow_id'] = workflow_id
        
        # Fetch recent few to check for new
        recent = await run_in_threadpool(
            workflow_manager.gi_object.gi.invocations.get_invocations,
            **filter_params,
            limit = NumericLimits.INVOCATION_LIMIT
        )
        
        has_new_or_updated = False
        for inv in recent:
            inv_id = inv.get('id')
            inv_create = datetime.fromisoformat(inv['create_time']) if inv.get('create_time') else datetime.min
            if inv_id not in cached_ids or inv_create > max_create:
                has_new_or_updated = True
                break
        
        if not has_new_or_updated:
            logger.info("No new invocations, using cache")
            return cached_invocations
        
        # Has new or updated, fetch all
        invocations = await run_in_threadpool(
            workflow_manager.gi_object.gi.invocations.get_invocations,
            **filter_params
        )
        
        # Cache the results
        await cache.set_invocations_cache(username, invocations, filters)
        return invocations
    
    async def fetch_workflows():
        """Fetch workflows with caching"""
        cached_workflows = await cache.get_workflows_cache(username)
       
        if cached_workflows:
            logger.info("Using cached workflows")
            return cached_workflows
        
        workflows = await invocation_background.fetch_workflows_safely(workflow_manager=workflow_manager, fetch_details = True)
        # Cache the results
        await cache.set_workflows_cache(username, workflows)
        return workflows
    
    
    # Execute all fetches in parallel
    try:
        results = await asyncio.gather(
            fetch_invocations(),
            fetch_workflows(),
            return_exceptions=True
        )
        
        # Handle results and exceptions
        invocations_data = results[0] if not isinstance(results[0], Exception) else []
        workflows_data = results[1] if not isinstance(results[1], Exception) else []
        
        # Log any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                operation = ["invocations", "workflows"][i]
                logger.error(f"Failed to fetch {operation}: {result}")
        
        return invocations_data, workflows_data
        
    except Exception as e:
        logger.error(f"Error in parallel data fetching: {e}")
        return [], []

async def _format_invocations(username, invocations_data: list[dict], workflow_mapping: dict[str, dict]):
    """Format invocations with optimized state mapping"""
    
    invocation_list = []
    logger.info(f"length og mappings: {len(workflow_mapping)}  and invocations: {len(invocations_data)}")
    length = 0
    for inv in invocations_data:
        inv_id = inv.get('id')
        
        # Get workflow info from mapping
        workflow_info = workflow_mapping.get(inv_id, {})
        workflow_name = workflow_info.get('workflow_name', None)
        workflow_id = workflow_info.get("workflow_id", None)
        
        # Skip if no workflow name : Likely due to the invocation being a subworkflow invocation or workflow being deleted.
        # TODO: Maybe in the future if invocation are still needed even after the workflow has been deleted, we can refactor implemenation.
        if not workflow_name:
            length+=1
            continue
        
        # Optimized state mapping
        raw_state = inv.get('state')
        inv_state = await _map_invocation_state(username, raw_state, inv.get("id"))
        
        invocation_list.append(
            invocation.InvocationListItem(
                id=inv_id,
                workflow_name=workflow_name,
                workflow_id=workflow_id,
                history_id=inv.get('history_id'),
                state=inv_state,
                create_time=inv.get('create_time'),
                update_time=inv.get('update_time')
            )
        )
        
    logger.debug(f"skipped {length} invocation becasue workflowname was missing.")
    
    return invocation_list


async def _map_invocation_state(username, raw_state: str, invocation_id: str) -> str:
    """Map Galaxy invocation states to user-friendly states"""
    
    # Get invocation state if saved for accurate state.
    cached_state = await invocation_cache.get_invocation_state(username, invocation_id)
    if cached_state:
        logger.info(f"Getting cached invocation state: {cached_state}")
        return cached_state

    
    if not raw_state:
        logger.warning("Invocation state is None or empty")
        return "Unknown"
    
    # Define state mappings
    failed_states = {"cancelled", "failed", "cancelling"}
    pending_states = {"requires_materialization", "ready", "new", "scheduled"}
    
    if raw_state in failed_states:
        return "Failed"

    elif raw_state in pending_states:
        logger.warning("invocation state might be inaccurate.")
        return "Pending"
    else:
        logger.warning(f"Unknown invocation state: {raw_state}")
        return "Unknown"


def _generate_request_hash(*args) -> str:
    """Generate hash for request deduplication"""
    request_string = "|".join(str(arg) if arg is not None else "None" for arg in args)
    return hashlib.md5(request_string.encode()).hexdigest()


async def _handle_partial_failure(api_key, workflow_id, history_id):
    """Handle partial failures by returning what we can"""
    try:
        # Try to get basic invocation data without workflows
        galaxy_client = GalaxyClient(api_key)
        username = galaxy_client.whoami
        workflow_manager = WorkflowManager(galaxy_client)
        
        if workflow_id and history_id:
            invocations = await run_in_threadpool(
                workflow_manager.gi_object.gi.invocations.get_invocations,
                workflow_id=workflow_id,
                history_id = history_id
            )  
        elif workflow_id:
            invocations = await run_in_threadpool(
                workflow_manager.gi_object.gi.invocations.get_invocations,
                workflow_id=workflow_id
            )
        elif history_id:
            invocations = await run_in_threadpool(
                workflow_manager.gi_object.gi.invocations.get_invocations,
                history_id=history_id
            )
        else:
            invocations = await run_in_threadpool(
                workflow_manager.gi_object.gi.invocations.get_invocations
            )
        
        # Create minimal invocation list without workflow names
        invocation_list = []
        for inv in invocations:
            invocation_list.append(
                invocation.InvocationListItem(
                    id=inv.get('id'),
                    workflow_name="Unknown (partial failure)",
                    workflow_id='Unknown (partial failure)',
                    history_id=inv.get('history_id'),
                    state=await _map_invocation_state(username, inv.get('state'), inv.get('id')),
                    create_time=inv.get('create_time'),
                    update_time=inv.get('update_time')
                )
            )
        
        return invocation.InvocationList(
            invocations=invocation_list
        )
        
    except Exception as e:
        logger.error(f"Partial failure handling also failed: {e}")
        raise


@router.get(
    "/{invocation_id}/invocation_pdf",
    response_class = FileResponse,
    summary= "Get invocation pdf report",
    tags=["Invocation", "Workflows"]
)
async def invocation_report_pdf(
    invocation_id: str = Path(..., description="The ID of the invocation from a certain workflow")
):

    galaxy_client = GalaxyClient(current_api_key.get())
    workflow_manager = WorkflowManager(galaxy_client)

    tmpdir = tempfile.mkdtemp(prefix="galaxy_pdf_")
    tmpdir_path = pathlib.Path(tmpdir)

    try:

        inv = await run_in_threadpool(
            workflow_manager.gi_object.gi.invocations.show_invocation,
            invocation_id=invocation_id
        )
        
        try:
            # get workflow object 
            workflow_obj = await run_in_threadpool(
                workflow_manager.gi_object.workflows.get,
                id_ = inv.get("workflow_id")
            )
            workflow_name = workflow_obj.name
        except Exception as e:
            logger.error(f"error finding workflow name: {e} defaulting to id.")
            workflow_name = inv.get("workflow_id")

        pdf_report_name = re.sub(r'[\\/*?:"<>|]', '', f"{workflow_name}_invocation_report.pdf")
        pdf_path = tmpdir_path / pdf_report_name

        await run_in_threadpool(
            workflow_manager.gi_object.gi.invocations.get_invocation_report_pdf,
            invocation_id=invocation_id,
            file_path=str(pdf_path)
        )
        background = BackgroundTasks()
        background.add_task(run_sync, _rmtree_sync, tmpdir)

        return FileResponse(
            path=pdf_path,
            filename=pdf_report_name,
            media_type="application/octet-stream",
            background=background
        )
    except Exception as exc:
        # Clean up immediately on error
        await run_sync(_rmtree_sync, tmpdir)
        raise InternalServerErrorException("Failed to get PDF invocation report")
  
  
async def structure_outputs(_invocation: Invocation, outputs: Dict[str, list], workflow_manager: WorkflowManager, failure: bool = False):
    """Strcucture invocation result outputs, datasets, collection and reports"""
    # Prepare workflow invocation results
    invocation_report = workflow_manager.gi_object.gi.invocations.get_invocation_report(_invocation.id)
    
    collection_outputs = outputs.get( "collection_datasets")
    dataset_ouputs = outputs.get("output_datasets")
    
    logger.info(f"Structuring invocation outputs for result.")

    try:
        # Format the outputs (Pydantic schemas could be added here for validation)
        final_output_dataset = []
        final_collection_dataset = []
        semaphore = asyncio.Semaphore(NumericLimits.SEMAPHORE_LIMIT)
        
        async def structure_and_append(output_id: str, store_list: list, collection: bool):
            async with semaphore:
                try:
                    if not collection:
                        dataset_info = await asyncio.to_thread(
                            workflow_manager.gi_object.gi.datasets.show_dataset,
                            dataset_id=output_id
                        )
                                        
                        store_list.append({
                            "type": "dataset",
                            "id": dataset_info.get("id"),
                            "name": dataset_info.get("name"),
                            "visible": dataset_info.get("visible"),
                            "file_path": dataset_info.get('file_name'),
                            "peek": dataset_info.get('peek'),
                            "data_type": dataset_info.get('extension', 'file_ext'),
                            "is_intermediate": not dataset_info.get("visible")
                        })
                    else:
                        output = await asyncio.to_thread(
                            workflow_manager.gi_object.dataset_collections.get, output_id
                        )
                        store_list.append({
                            "type": "collection",
                            "id": output.id,
                            "name": output.name,
                            "visible": output.visible,
                            "collection_type": output.collection_type,
                            "elements": [
                                {
                                    "identifier": e.get("element_identifier", ""),
                                    "name": e.get("object", {}).get("name", ""),
                                    "id": e.get("object", {}).get("id", ""),
                                    "peek": e.get("object", {}).get("peek", ""),
                                    "data_type": (await asyncio.to_thread(
                                        workflow_manager.gi_object.gi.datasets.show_dataset,
                                        dataset_id=e.get("object", {}).get("id", "")
                                    )).get('extension', 'file_ext') if e.get("object", {}).get("id", "") else ''
                                }
                                for e in output.elements
                            ],
                            "is_intermediate": not output.visible
                        })
                        
                except Exception as e:
                    logger.error(f"Error when structuring inputs: {e}")

        # Create coroutines for formatting
        dataset_tasks = [structure_and_append(output, final_output_dataset, False) for output in dataset_ouputs]
        collection_tasks = [structure_and_append(output, final_collection_dataset, True) for output in collection_outputs]
        
        # Run them concurrently
        await asyncio.gather(
                    *dataset_tasks,
                    *collection_tasks
                )
        
        logger.info(f"output datasets formatted: {len(final_output_dataset)}, collection datasets formatted: {len(final_collection_dataset)}")
        
        if failure:
            return final_output_dataset + final_collection_dataset, None
        else:
            # Prepare workflow invocation results
            invocation_report_dict = await asyncio.to_thread(
                    workflow_manager.gi_object.gi.invocations.get_invocation_report,
                    invocation_id = _invocation.id
                    )
            invocation_report = (
                f"### {invocation_report_dict.get('title', '')}\n\n"
                f"{invocation_report_dict.get('markdown', '')}"
            )
                
        return final_output_dataset + final_collection_dataset, invocation_report

    except Exception as e:
        logger.error(f"Error formatting outputs: {e}")
        raise

        
async def structure_inputs(inv: Dict, workflow_manager: WorkflowManager):
    inputs_formatted = {}
    try:
        # Process dataset and collection inputs
        if inv['inputs'] is not None:
            async def process_input(step_index, input_value):
                label = input_value.get('label', f"Input_{step_index}")
                
                if input_value.get('src') == 'hda':  # Dataset
                    dataset_id = input_value['id']
                    dataset_info = await asyncio.to_thread(
                        workflow_manager.gi_object.gi.datasets.show_dataset,
                        dataset_id
                    )
                    return label, {
                        "type": "dataset",
                        "id": dataset_id,
                        "name": dataset_info.get('name', ''),
                        "visible": dataset_info.get('visible', False),
                        "file_path": dataset_info.get('file_name'),
                        "peek": dataset_info.get('peek', ''),
                        "data_type": dataset_info.get('extension', 'file_ext'),
                        "step_id": input_value.get('workflow_step_id', '')
                    }
                    
                elif input_value.get('src') == 'hdca':  # Collection
                    collection_id = input_value['id']
                    collection = await asyncio.to_thread(
                        workflow_manager.gi_object.dataset_collections.get,
                        collection_id
                    )
                    
                    async def get_element_data(e):
                        element_obj = e.get('object', {})
                        element_id = element_obj.get('id')
                        if element_id:
                            element_dataset = await asyncio.to_thread(
                                workflow_manager.gi_object.gi.datasets.show_dataset,
                                element_id
                            )
                            return {
                                "identifier": e.get('element_identifier', ''),
                                "name": element_dataset.get('name', ''),
                                "id": element_id,
                                "peek": element_dataset.get('peek', ''),
                                "data_type": element_dataset.get('data_type', '')
                            }
                        return None
                    
                    element_coros = [get_element_data(e) for e in collection.elements]
                    elements_formatted = [el for el in await asyncio.gather(*element_coros) if el is not None]
                    
                    return label, {
                        "type": "collection",
                        "id": collection_id,
                        "name": collection.name,
                        "visible": collection.visible,
                        "collection_type": collection.collection_type,
                        "elements": elements_formatted,
                    }
                
                return label, None
            
            input_coros = [process_input(step_index, input_value) for step_index, input_value in inv['inputs'].items()]
            input_results = await asyncio.gather(*input_coros)
            for label, value in input_results:
                if value:
                    inputs_formatted[label] = value
        
        # Process parameter inputs
        if inv["input_step_parameters"] is not None:
            for step_index, param_value in inv["input_step_parameters"].items():
                label = param_value.get('label', f"Parameter_{step_index}")
                inputs_formatted[label] = {
                    "type": "parameter",
                    "value": param_value.get('parameter_value', ''),
                }
        
        logger.info("Structuring invocation inputs for result.")
    except Exception as e:
        raise InternalServerErrorException("structuring input structure failed")
    
    return inputs_formatted

async def report_invocation_failure(galaxy_client: GalaxyClient, invocation_id: str)-> str:
    """Report falure error for an invocation"""
    
    explanation = ""
    failed_job_descriptions = []
    
    # Get invocation jobs
    try:
        invocation_jobs = galaxy_client.gi_client.jobs.get_jobs(invocation_id=invocation_id)
    except Exception as e:
        logger.error(f"Error retrieving invocation jobs: {str(e)}")
        explanation += f"Error retrieving invocation jobs: {str(e)}\n"
        return explanation
    
    # Prepare concurrent tasks for failed jobs
    failed_job_tasks = []
    for inv_job in invocation_jobs:
        if inv_job.get("state") == 'error':
            job_id = inv_job.get("id")
            if job_id:
                failed_job_tasks.append(
                            asyncio.to_thread(galaxy_client.gi_client.jobs.show_job, job_id = job_id, full_details=True)
                        )
        
    if failed_job_tasks:
        try:
            job_details_list = await asyncio.gather(*failed_job_tasks)
            logger.info(f"Number of failed jobs in invocation {len(job_details_list)}")
        except Exception as e:
            logger.error(f"Error retrieving job details concurrently: {str(e)}")
            explanation += f"Error retrieving failed job details: {str(e)}\n"
            return explanation
        
        for job_details in job_details_list:
            try:
                # Extract the key error describers
                tool_id = job_details.get('tool_id', "Unknown")
                stderr_output = job_details.get('stderr') or job_details.get('tool_stderr') or job_details.get('job_stderr') or ""
                std_out = job_details.get('stdout') or job_details.get('tool_stdout') or ""
                exit_code = job_details.get('exit_code')

                # Create a structured failure summary
                error_description = f"Tool `{tool_id}` failed during execution.\n"
                
                if std_out:
                    error_description += f"  - Job Message Logs: {std_out}\n"
                if exit_code:
                    error_description += f"  - Exit code: {exit_code}\n"
                if stderr_output:
                    error_description += f"  - Error message: {stderr_output.strip()}\n"

                
                failed_job_descriptions.append(error_description)
            except Exception as e:
                logger.error(f"Error processing job details: {str(e)}")
                explanation += f"Error processing failed job details: {str(e)}\n"
    
    if failed_job_descriptions:
        explanation += "\n\nFailed Jobs Detected:\n" + "\n".join(failed_job_descriptions) + "\n"
    
    else:
        explanation += "\nNo failed jobs detected in this invocation.\n"
        
    return explanation

async def background_track_and_cache(
    invocation_id: str,
    galaxy_client: GalaxyClient,
    workflow_manager: WorkflowManager,
    history_id,
    create_time,
    last_update_time,
    inputs_formatted,
    workflow_description,
    ws_manager,
    tracking_key
    ):
    """Background task to track invocation, structure results, and cache when ready"""

    username = galaxy_client.whoami
    
    try:
   
        _invocation = await run_in_threadpool(
            galaxy_client.gi_object.invocations.get,
            id_=invocation_id
        )
        if not _invocation:
            logger.error(f"Invocation {invocation_id} not found in background")
            return

        outputs, inv_state, update_time = await workflow_manager.track_invocation(
            invocation=_invocation,
            tracker_id=invocation_id,
            ws_manager=ws_manager,
            invocation_check=True
        )
        
        # Set invocation to cache
        await invocation_cache.set_invocation_state(username, invocation_id, inv_state)

        if inv_state == "Failed":
            
            invocation_report = await report_invocation_failure(galaxy_client, invocation_id)
            workflow_result, _ = await structure_outputs(
            _invocation=_invocation, outputs=outputs, workflow_manager=workflow_manager, failure = True
            )
        else:
            workflow_result, invocation_report = await structure_outputs(
            _invocation=_invocation, outputs=outputs, workflow_manager=workflow_manager
            )
            
        result_dict = {
            "invocation_id": invocation_id,
            "state": inv_state,
            "history_id": history_id,
            "create_time": create_time,
            "update_time": update_time if update_time else last_update_time,
            "inputs": inputs_formatted,
            "result": workflow_result,
            "workflow": workflow_description.model_dump(),
            "report": invocation_report
        }
        await invocation_cache.set_invocation_result(username, invocation_id, result_dict)
        logger.info("Invocation results are complete and ready.")
        if ws_manager:
            ws_data = {
                "type": SocketMessageType.INVOCATION_COMPLETE,
                "payload": {"message": "Invocation results are complete and ready."}
            }
            await ws_manager.broadcast(
                event=SocketMessageEvent.workflow_execute,
                data=ws_data,
                tracker_id=invocation_id
            )


    except Exception as e:
        logger.error(f"Error in background tracking for {invocation_id}: {e}")
        result_dict = {
            "invocation_id": invocation_id,
            "state": "Failed",
            "history_id": history_id,
            "create_time": create_time,
            "update_time": last_update_time,
            "inputs": inputs_formatted,
            "result": [],
            "workflow": workflow_description.model_dump(),
            "report": None
        }
        await invocation_cache.set_invocation_result(username, invocation_id, result_dict)

    finally:
        await run_in_threadpool(redis_client.delete, tracking_key)


@router.get(
    "/{invocation_id}/result",
    response_model=invocation.InvocationResult,
    summary="Result of a certain workflow invocations",
    tags=["Invocation"]
)
async def show_invocation_result(
    invocation_id: str = Path(..., description=""),
    internal_api: str | None = None
):
    try:
        if internal_api:
            logger.info("Calling invocation result internally.")
            api_key = internal_api
        else:
            api_key = current_api_key.get()
            
        galaxy_client = GalaxyClient(api_key)
        username = galaxy_client.whoami

        # Step 1: Check if deleted
        deleted = await invocation_cache.get_deleted_invocations(username)
        if invocation_id in deleted:
            raise NotFoundException("Invocation not found")

        # Step 2: Check cache for full result
        cached_result = await invocation_cache.get_invocation_result(username, invocation_id)
        if cached_result:
            return invocation.InvocationResult(**cached_result)

        # Step 3: Fetch invocation for preliminary info
        workflow_manager = WorkflowManager(galaxy_client)

        _invocation, invocation_details = await asyncio.gather(
            asyncio.to_thread(
                galaxy_client.gi_object.invocations.get,
                id_=invocation_id
            ),
            asyncio.to_thread(
                galaxy_client.gi_object.gi.invocations.show_invocation,
                invocation_id =invocation_id
            )
        )
        if not _invocation:
            raise NotFoundException("Invocation not found")

        # Fetch common details for partial response
        mapping = await invocation_cache.get_invocation_workflow_mapping(username)
        if not mapping:
            logger.warning("workflow to invocation map not found, warming user cache.")
            await invocation_background.warm_user_cache(token = "dummytoken", api_key=api_key) # using a dummy token to fill the functionality
            mapping = await invocation_cache.get_invocation_workflow_mapping(username)
            
        if not mapping or invocation_id not in mapping:
            raise InternalServerErrorException("could not find workflow details with the invocation id inputted.")
        
        stored_workflow_id = mapping.get(invocation_id, {}).get('workflow_id')

        # Retrieve workflow description
        workflow_description_list = await invocation_cache.get_workflows_cache(username)
        workflow_description = None
        
        if workflow_description_list:
            for _workflow in workflow_description_list:
                try:
                    if stored_workflow_id == _workflow.get("id"):
                        workflow_description = workflow.WorkflowListItem(**_workflow)
                except Exception as e:
                    logger.warning(f"coudn't retreive from cache: {e}")
                    
        if workflow_description is None:
            raise InternalServerErrorException("Could not locate workflow description in cache.")

        inputs_formatted = await structure_inputs(inv=invocation_details, workflow_manager=workflow_manager)
        
        # Compute preliminary state (without full check)
        state = invocation_details.get("state")
        if state in ["cancelled", "failed"]:
            inv_state = "Failed"
        elif state in ["requires_materialization", "ready", "new", "scheduled"]:
            inv_state = "Pending"
        else:
            logger.warning(f"Unknown invocation state: {state}")
            inv_state = "Failed"
        
        # If pending-like, trigger background tracking if not already
        tracking_key = f"tracking:{api_key}:{invocation_id}"
        
        if inv_state == "Pending":
            
            invocation_result = {
                "invocation_id" : invocation_details.get("id"),
                "state" : inv_state,
                "history_id" : invocation_details.get("history_id"),
                "create_time" : invocation_details.get("create_time"),
                "update_time" :invocation_details.get("update_time"),
                "inputs" : inputs_formatted,
                "result" : [],
                "workflow" : workflow_description.model_dump(),
                "report" : None 
            }
            
            await invocation_cache.set_invocation_result(
                username = username, 
                invocation_id = invocation_details.get("id"),
                result = invocation_result,
                )
            
            set_result = await run_in_threadpool(
                redis_client.set, tracking_key, "1", ex=NumericLimits.BACKGROUND_INVOCATION_TRACK, nx=True
            )
            if set_result: 
                
                background_task = asyncio.create_task(
                    background_track_and_cache(
                        invocation_id = invocation_id,
                        galaxy_client = galaxy_client,
                        workflow_manager = workflow_manager,
                        history_id = invocation_details.get("history_id"),
                        create_time = invocation_details.get("create_time"),
                        last_update_time = invocation_details.get("update_time"),
                        inputs_formatted = inputs_formatted,
                        workflow_description= workflow_description,
                        ws_manager = ws_manager,
                        tracking_key = tracking_key
                        )
                    )
                background_task.add_done_callback(lambda t: log_task_error(t, task_name="track invocation"))
        else:
            logger.info(f"Loading invocation failure report for invocation {invocation_id}")
            invocation_report = await report_invocation_failure(galaxy_client = galaxy_client, invocation_id = invocation_id)
            invocation_result = {
                "invocation_id" : invocation_details.get("id"),
                "state" : inv_state,
                "history_id" : invocation_details.get("history_id"),
                "create_time" : invocation_details.get("create_time"),
                "update_time" :invocation_details.get("update_time"),
                "inputs" : inputs_formatted,
                "result" : [],
                "workflow" : workflow_description.model_dump(),
                "report" : invocation_report 
            }
            
            await invocation_cache.set_invocation_result(
                username = username, 
                invocation_id = invocation_details.get("id"),
                result = invocation_result,
                )
        
        return invocation.InvocationResult(**invocation_result)
    except Exception as e:
        logger.error(f"Error getting invocation result: {e}")
        raise InternalServerErrorException("Error getting invocation result")


async def _cancel_invocation_and_delete_data(invocation_ids: List[str], workflow_manager: WorkflowManager, username: str):
    """Cancel running invocaitons and delete data in the background."""
    try:    
        for invocation_id in invocation_ids:

            # Get the invocation object
            _invocation = await run_in_threadpool(
                workflow_manager.gi_object.invocations.get,
                id_=invocation_id
            )

            # Cancel if not in a terminal state
            state = _invocation.state
            terminal_states = {'cancelled', 'failed', 'scheduled', 'error'}  # 'scheduled' often means completed
            if state not in terminal_states:
                await run_in_threadpool(
                    workflow_manager.gi_object.gi.invocations.cancel_invocation,
                    invocation_id=invocation_id
                )
                # Short wait for cancellation to propagate (or implement polling for state change)
                await asyncio.sleep(NumericLimits.SHORT_SLEEP)

            # Get outputs and purge datasets/collections to free space
            outputs, _, _ = await workflow_manager.track_invocation(
                invocation=_invocation,
                tracker_id=invocation_id,
                ws_manager=None,
                invocation_check=True
            )
            
            # Concurrently delete invication output datasets.
            delete_datasets = [asyncio.to_thread(ds.delete, purge= True) for ds in outputs if isinstance(ds, HistoryDatasetAssociation)]
            delete_collections = [asyncio.to_thread(ds.delete) for ds in outputs if isinstance(ds, HistoryDatasetCollectionAssociation)]       
            await asyncio.gather(*delete_datasets, *delete_collections, return_exceptions= False)
            
            # Delete invocation states saved earlier.
            await asyncio.gather(*[invocation_cache.delete_invocation_state(username, inv_id) for inv_id in invocation_ids])
            
        
        logger.info(f"Invocation {invocation_ids} deletion complete.")
    except Exception as e:
            logger.error(f"Error while deleting invocation: {e}")
        
@router.delete(
    "/DELETE",
    summary="Delete workflow invocations",
    tags=["Invocation"],
    status_code=HTTP_204_NO_CONTENT
)
async def delete_invocations(
    invocation_ids: str = Query(..., description="Comma-separated IDs of the workflow invocations to delete")
) -> Response:
    """
    Simulates deletion of workflow invocations in the middleware layer since Galaxy does not support
    permanent deletion of invocation records via API. Cancels the invocation(s) if running, purges
    associated datasets to free space, and marks them as deleted in persistent storage to filter
    from listings.
    """

    api_key = current_api_key.get()
    galaxy_client = GalaxyClient(api_key)
    username = galaxy_client.whoami
    
    workflow_manager = WorkflowManager(galaxy_client)

    try:
        # Parse comma-separated string into list
        ids_list = [i.strip() for i in invocation_ids.split(",") if i.strip()]

        # Mark as deleted using cache method
        await invocation_cache.add_to_deleted_invocations(username, ids_list)
        

        # Spawn async deletion tasks
        background_task = asyncio.create_task(_cancel_invocation_and_delete_data(ids_list, workflow_manager, username))
        background_task.add_done_callback(lambda t: log_task_error(t, task_name="Invocation cancelling and data deletion"))

        return Response(status_code=HTTP_204_NO_CONTENT)
    except Exception as e:
        raise InternalServerErrorException("Failed to delete invocations")