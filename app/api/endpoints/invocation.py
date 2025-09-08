import re
import tempfile
import uuid
import pathlib
import shutil
from anyio.to_thread import run_sync
from typing import Dict
import logging

from sys import path
path.append(".")

from fastapi import APIRouter, Path, Query, HTTPException, BackgroundTasks
from fastapi.responses import  FileResponse
from fastapi.concurrency import run_in_threadpool
from bioblend.galaxy.objects.wrappers import HistoryDatasetAssociation, HistoryDatasetCollectionAssociation, Invocation

from app.context import current_api_key
from app.bioblend_server.galaxy import GalaxyClient
from app.bioblend_server.executor.workflow_manager import WorkflowManager
from app.api.schemas import invocation, workflow
from app.api.socket_manager import ws_manager

# Helper functions
logger = logging.getLogger("invocation")

def _rmtree_sync(path: pathlib.Path):
    shutil.rmtree(path, ignore_errors=True)

async def structure_ouptuts(_invocation: Invocation, outputs: list, workflow_manager: WorkflowManager):

    # TODO : Optimization of intermediate and final output separation.
    # prepare workflow invocation results
    invocation_id, invocation_report, intermediate_outputs, final_outputs = await run_in_threadpool(
        workflow_manager._make_result,
        invocation= _invocation,
        outputs = outputs
        )
    logger.info(f"itermediate: {len(intermediate_outputs)}, final: {len(final_outputs)}")

    try:
        # Format the outputs using Pydantic schemas
        intermediate_outputs_formatted = []
        final_outputs_formatted = []

        for ds in final_outputs:
            if isinstance(ds, HistoryDatasetAssociation):
                final_outputs_formatted.append(
                            {
                            "type" : "dataset",
                            "id": ds.id,
                            "name": ds.name,
                            "visible": ds.visible,
                            "peek": workflow_manager.gi_object.gi.datasets.show_dataset(ds.id)['peek'],
                            "data_type": workflow_manager.gi_object.gi.datasets.show_dataset(ds.id)['data_type'],
                            "is_intermediate": False
                        }
                    )
            elif isinstance(ds, HistoryDatasetCollectionAssociation):
                final_outputs_formatted.append(
                        {
                            "type" : "collection",
                            "id": ds.id,
                            "name": ds.name,
                            "visible": ds.visible,
                            "collection_type": ds.collection_type,
                            "elements": [
                                {
                                    "identifier": e["element_identifier"],
                                    "name": e["object"]["name"],
                                    "id": e["object"]["id"],
                                    "peek": e["object"]["peek"],
                                    "data_type": workflow_manager.gi_object.gi.datasets.show_dataset(e["object"]["id"])['data_type']
                                }

                                for e in ds.elements
                            ],
                            "is_intermediate": False
                        }
                    )
            else:
                logger.error("result is neither a collection nor a dataset")
                
        for ds in intermediate_outputs:
            if isinstance(ds, HistoryDatasetAssociation):
                intermediate_outputs_formatted.append(
                            {
                            "type" : "dataset",
                            "id": ds.id,
                            "name": ds.name,
                            "visible": ds.visible,
                            "peek": workflow_manager.gi_object.gi.datasets.show_dataset(ds.id)['peek'],
                            "data_type": workflow_manager.gi_object.gi.datasets.show_dataset(ds.id)['data_type'],
                            "is_intermediate": True
                        }
                    )
            elif isinstance(ds, HistoryDatasetCollectionAssociation):
                intermediate_outputs_formatted.append(
                        {
                            "type" : "collection",
                            "id": ds.id,
                            "name": ds.name,
                            "visible": ds.visible,
                            "collection_type": ds.collection_type,
                            "elements": [
                                {
                                    "identifier": e["element_identifier"],
                                    "name": e["object"]["name"],
                                    "id": e["object"]["id"],
                                    "peek": e["object"]["peek"],
                                    "data_type": workflow_manager.gi_object.gi.datasets.show_dataset(e["object"]["id"])['data_type']
                                }

                                for e in ds.elements
                            ],
                            "is_intermediate": True
                        }
                    )
            else:
                logger.error("result is neither a collection nor a dataset")
                
        logger.info(f"itermediate formatted: {len(intermediate_outputs_formatted)}, final formatted : {len(final_outputs_formatted)}")
                
        combined_final_result = [*intermediate_outputs_formatted, *final_outputs_formatted]
                                        
        # workflow_result = invocation.WorkflowExecutionResponse(
        #                 invocation_id =invocation_id,
        #                 history_id= _invocation.history_id,
        #                 report = invocation_report,
        #                 result = combined_final_result
        #         )
        
        return combined_final_result, invocation_report
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list workflows: {e}")

        
async def structure_inputs(inv: Dict, workflow_manager: WorkflowManager):
    inputs_formatted = {}
    try:
        # Process dataset and collection inputs
        if inv['inputs'] is not None:
            for step_index, input_value in inv['inputs'].items():
                label = input_value.get('label', f"Input_{step_index}")
                
                if input_value.get('src') == 'hda':  # Dataset
                    dataset_id = input_value['id']
                    dataset_info = await run_in_threadpool(
                        workflow_manager.gi_object.gi.datasets.show_dataset,
                        dataset_id
                    )
                    inputs_formatted[label] = {
                        "type": "dataset",
                        "id": dataset_id,
                        "name": dataset_info.get('name', ''),
                        "visible": dataset_info.get('visible', False),
                        "peek": dataset_info.get('peek', ''),
                        "data_type": dataset_info.get('data_type', ''),
                        "step_id": input_value.get('workflow_step_id', '')
                    }
                    
                elif input_value.get('src') == 'hdca':  # Collection
                    collection_id = input_value['id']
                    collection = await run_in_threadpool(
                        workflow_manager.gi_object.gi.datasets.show_dataset,
                        collection_id
                    )
                    elements_formatted = []
                    for e in collection.get('elements', []):
                        element_obj = e.get('object', {})
                        element_id = element_obj.get('id')
                        if element_id:
                            element_dataset = await run_in_threadpool(
                                workflow_manager.gi_object.gi.datasets.show_dataset,
                                element_id
                            )
                            elements_formatted.append({
                                "identifier": e.get('element_identifier', ''),
                                "name": element_dataset.get('name', ''),
                                "id": element_id,
                                "peek": element_dataset.get('peek', ''),
                                "data_type": element_dataset.get('data_type', '')
                            })
                    inputs_formatted[label] = {
                        "type": "collection",
                        "id": collection_id,
                        "name": collection.get('name', ''),
                        "visible": collection.get('visible', False),
                        "collection_type": collection.get('collection_type', ''),
                        "elements": elements_formatted,
                    }
        
        # Process parameter inputs
        if inv["input_step_parameters"] is not None:
            for step_index, param_value in inv["input_step_parameters"].items():
                label = param_value.get('label', f"Parameter_{step_index}")
                inputs_formatted[label] = {
                    "type": "parameter",
                    "value": param_value.get('parameter_value', ''),
                }
    except Exception as e:
        raise HTTPException(status_code= 500, detail= f"structuring input structure failed: {e}")          
                              
    return inputs_formatted

router = APIRouter()

@router.get(
    "/",
    response_model=invocation.InvocationList,
    summary="List all workflow invocations",
    tags=["Invocation"]
)
async def list_invocations(
    workflow_id: str | None = Query(None, description="Filter by workflow ID"),
    history_id: str | None = Query(None, description="Filter by History ID")
):
    """
    Retrieves a list of all workflow invocations in the Galaxy instance.
    Optionally filter results to only show invocations for a specific workflow, and history.
    Returns basic information including invocation ID, workflow ID, history ID, state, and timestamps.
    """
    galaxy_client = GalaxyClient(current_api_key.get())
    workflow_manager = WorkflowManager(galaxy_client)
    try:

        if history_id and workflow_id:
            invocations = await run_in_threadpool(
                workflow_manager.gi_object.gi.invocations.get_invocations,
                workflow_id = workflow_id,
                history_id = history_id
            )
            logger.info("invocation retreived with history and workflow filter")
        elif history_id:
            invocations = await run_in_threadpool(
                workflow_manager.gi_object.gi.invocations.get_invocations,
                history_id = history_id
            )
            logger.info("invocation retreived with history filter")
        # Filter by workflow_id if provided
        elif workflow_id:
            invocations = await run_in_threadpool(
                workflow_manager.gi_object.gi.invocations.get_invocations,
                workflow_id = workflow_id
            )
            logger.info("invocation retreived with workflow filter")
        else:
            # Get all invocations from Galaxy instance
            invocations = await run_in_threadpool(
                workflow_manager.gi_object.gi.invocations.get_invocations
            )
            logger.info("invocation retreived with no filter")
        # Get list of workflows
        workflows = await run_in_threadpool(
            workflow_manager.gi_object.gi.workflows.get_workflows
        )

        # format and map invocation to workflow_id
        invocation_list = []
        for inv in invocations:
            stored_workflow_id = None
            for wf in workflows:
                for inv_ in invocations:
                    if inv['id'] == inv_["id"]:
                        stored_workflow_id = wf["id"]
                        workflow_name = wf["name"]
                        break
                if stored_workflow_id:
                    break

            invocation_list.append(
                invocation.InvocationListItem(
                    id=inv.get("id"),
                    workflow_name= workflow_name,
                    workflow_id=inv.get("workflow_id"),
                    history_id=inv.get("history_id"),
                    state=inv.get("state"),
                    create_time=inv.get("create_time"),
                    update_time=inv.get("update_time")
                )
            )
        
        return invocation.InvocationList(invocations=invocation_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list invocations: {e}")



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
        raise HTTPException(status_code=500, detail=f"Failed to get PDF invocation report: {exc}")
    
@router.get(
    "/{invocation_id}/result",
    response_model=invocation.InvocationResult,
    summary="Result of a certain workflow invocations",
    tags=["Invocation"]
)
async def show_invocation_result(
    invocation_id = Path(..., description= "")
):
    galaxy_client = GalaxyClient(current_api_key.get())
    workflow_manager = WorkflowManager(galaxy_client)

    # TODO: Refactoring websocket connection by using the invocation id as the the room id
    # Generate a random tracker_id for now
    # Set websocket to use the invocation id as the room id
    # Get invocation object
    _invocation = await run_in_threadpool(
        galaxy_client.gi_object.invocations.get,
        id_ = invocation_id
        )
    outputs, invocation_completed = await workflow_manager.track_invocation(
        invocation = _invocation, 
        tracker_id = invocation_id, 
        ws_manager = ws_manager,
        invocation_check = True
        )
    
    logger.info(len(outputs))

    stored_workflow_id = None
    workflows = workflow_manager.gi_object.gi.workflows.get_workflows()
    for wf in workflows:
        wf_invocations = workflow_manager.gi_object.gi.workflows.get_invocations(wf['id'])
        for inv in wf_invocations:
            if inv['id'] == invocation_id:
                stored_workflow_id = wf['id']
                break
        if stored_workflow_id:
            break
        
    # Get workflow Name.
    try:
        workflow_ = await run_in_threadpool(
            workflow_manager.gi_object.gi.workflows.show_workflow,
            workflow_id = stored_workflow_id
            )
        
    except Exception as e:
        logger.error(f"workflow removed or unknown: {e}")

    # Structure workflow details
    workflow_description = workflow.WorkflowListItem(
        id = workflow_.get("id", "Unknown"),
        name = workflow_.get("name", "Unknown"),
        description = workflow_.get("annotation") or workflow_.get("description", None),
        tags = workflow_.get("tags", "Unknown")
    )

    #Strucutre inputs into appropriate format
    inputs_formatted = await structure_inputs(inv=inv, workflow_manager=workflow_manager)

     # Wrap invocaion id into invocation object
    _invocation = await run_in_threadpool(
        workflow_manager.gi_object.invocations.get, 
        id_ = invocation_id
        )

    # Structure invocation results
    workflow_result, invocation_report = await structure_ouptuts(_invocation= _invocation, outputs = outputs, workflow_manager= workflow_manager)

    # Strucutre Invocation state.
    if inv.get("state") == "cancelled" or inv.get("state") == "cancelled" or inv.get("state") == "failed":
        inv_state = "Failed"
    elif invocation_completed and inv.get("state") == "scheduled":
        inv_state = "Complete"
    elif inv.get("state") == "requires_materialization" or inv.get("state") == "ready" or inv.get("state") == "new" or inv.get("state") == "scheduled":
        inv_state = "Pending"
    else:
        logger.warning(f"invocation state unknown: {inv.get('state')}") # Flag unknown invocation state or if it is none
        inv_state == "Failed"

    return invocation.InvocationResult(
        invocation_id = inv.get("id"),
        state = inv_state,
        history_id = inv.get("history_id"),
        create_time = inv.get("create_time"),
        update_time = inv.get("update_time"),
        inputs = inputs_formatted,
        result = workflow_result,
        workflow= workflow_description,
        report = invocation_report
    )