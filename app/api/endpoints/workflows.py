from sys import path
path.append('.')
import tempfile, pathlib, shutil, re, os
from anyio.to_thread import run_sync
import logging
import uuid

from fastapi import APIRouter, UploadFile,File, Path, Query, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.concurrency import run_in_threadpool
from bioblend.galaxy.objects.wrappers import HistoryDatasetAssociation, HistoryDatasetCollectionAssociation, Workflow

from app.context import current_api_key
from app.bioblend_server.galaxy import GalaxyClient
from app.bioblend_server.executor.workflow_manager import WorkflowManager
from app.api.schemas import workflow
from app.api.socket_manager import ws_manager, SocketMessageEvent, SocketMessageType

logger = logging.getLogger('workflow_endpoint')
router = APIRouter()


@router.get(
    "/",
    response_model=workflow.WorkflowList,
    summary="List all workflows",
    tags=["Workflows"]
)
async def list_workflows():
    """
    Retrieves a list of all workflows available in the Galaxy instance.
    Returns basic information including workflow ID, name, and description.
    """

    galaxy_client = GalaxyClient(current_api_key.get())
    workflow_manager = WorkflowManager(galaxy_client)

    try:
        # Get all workflows from Galaxy instance with full details
        workflows = await run_in_threadpool(
            workflow_manager.gi_object.gi.workflows.get_workflows
         )
        
        # Extract only the required fields
        workflow_list = []
        for wf in workflows:
            # Get full workflow details to access annotations
            full_workflow: dict = await run_in_threadpool(
                workflow_manager.gi_object.gi.workflows.show_workflow,
                workflow_id=wf['id']
            )
            
            workflow_list.append(
                workflow.WorkflowListItem(
                    id=full_workflow["id"],
                    name=full_workflow["name"],
                    description=full_workflow.get("annotation") or full_workflow.get("description", None),
                    tags = full_workflow.get("tags")
                )
            )
        
        return workflow.WorkflowList(workflows=workflow_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list workflows: {e}")
    
@router.post(
        "/upload-workflow",
        response_model = workflow.WorkflowDetails,
        summary = "Upload an external workflow ga file",
        tags=["Workflows"]
)
async def upload_workflow(
        file: UploadFile = File(..., description="The Workflow ga file to upload."),
        tracker_id: str | None = Query(None, description="Client-supplied tracker ID for WebSocket updates"),
):
    """Uploads an external workflow from a ga file into the galaxy instance."""

    galaxy_client = GalaxyClient(current_api_key.get())
    workflow_manager = WorkflowManager(galaxy_client)
    tracker_id = tracker_id or str(uuid.uuid4())

    try:
        # Use a temporary file to handle the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            tmp.write(await file.read())    
            tmp_path = tmp.name

        # Upload the galaxt workflow into the instance
        workflow = await workflow_manager.upload_workflow(path = tmp_path,
                                                          ws_manager = ws_manager, 
                                                          tracker_id = tracker_id
                                                          )
        # Clean up the temporary file
        os.remove(tmp_path)

        if not isinstance(workflow, Workflow):
            raise HTTPException(status_code=500, detail = workflow.get('error', "workflow uploading failed."))
        
        workflow_details  = await run_in_threadpool(workflow_manager.gi_object.gi.workflows.show_workflow, workflow.id)
       
        return {
            "id": workflow_details.get("id"),
            "tags": workflow_details.get("tags", None),
            "create_time": workflow_details.get("create_time"),
            "annotations": workflow_details.get("annotations", None),
            "published": workflow_details.get("published"),
            "license": workflow_details.get("license", None),
            "galaxy_url": workflow_details.get("url"),
            "creator": workflow_details.get("creator", None),
            "steps": workflow_details.get("steps"),
            "inputs": workflow_details.get("inputs"),
        }
    except Exception as e:
        # Clean up in case of error
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
 
@router.get(
    "/{workflow_id}/form",
    response_class=HTMLResponse,
    summary="Get Dynamic Workflow Form",
    tags=["Workflows"]
)
async def get_workflow_form(
    workflow_id: str = Path(..., description="The ID of the Galaxy workflow."),
    history_id: str = Query(..., description="The ID of the history to select inputs from.")
):
    """
    Generates and returns a dynamic HTML form for a specific workflow.
    The form is populated with available datasets from the provided history.
    """

    galaxy_client = GalaxyClient(current_api_key.get())
    workflow_manager = WorkflowManager(galaxy_client)

    try:
        # Get the wrapper objects needed by your service method
        workflow_obj = await run_in_threadpool(workflow_manager.gi_object.workflows.get, workflow_id)
        history_obj = await run_in_threadpool(workflow_manager.gi_object.histories.get, history_id)

        # Run the synchronous HTML build in a thread pool
        html_form = await run_in_threadpool(
            workflow_manager.build_input,
            workflow=workflow_obj,
            history=history_obj
        )
        return HTMLResponse(content=html_form)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build workflow form: {e}")

@router.post(
    "/{workflow_id}/histories/{history_id}/execute",
    response_model=workflow.WorkflowExecutionResponse,
    summary="Execute a Workflow",
    tags=["Workflows"]
)
async def execute_workflow(
    request: Request,
    dummy_input: str = Form(None, description="Dummy input to force form rendering the form input for the galaxy execution"), # Adding dummy form value to input request form.
    workflow_id: str = Path(..., description="The ID of the Galaxy workflow to execute."),
    history_id: str = Path(..., description="The ID of the history for execution."),
    tracker_id: str | None = Query(None, description="Client-supplied tracker ID for WebSocket updates"),
):
    """
    Executes a workflow using input data submitted via a form.

    This endpoint expects a `multipart/form-data` submission, as generated by the `/form` endpoint.
    It parses the form, invokes the workflow, tracks it to completion, and returns the results.
    """

    galaxy_client = GalaxyClient(current_api_key.get())
    workflow_manager = WorkflowManager(galaxy_client)
    tracker_id = tracker_id or str(uuid.uuid4())

    try:
        await ws_manager.broadcast(event = SocketMessageEvent.workflow_execute,
                             data = {
                                 "type": SocketMessageType.WORKFLOW_EXECUTE,
                                 "payload": {"message" : "Execution started."}   
                                },
                             tracker_id=tracker_id
        )
        form_data = await request.form()
        workflow_obj = await run_in_threadpool(workflow_manager.gi_object.workflows.get, workflow_id)
        history_obj = await run_in_threadpool(workflow_manager.gi_object.histories.get, history_id)
        workflow_details = await run_in_threadpool(workflow_manager.gi_object.gi.workflows.show_workflow, workflow_id)

        # Reconstruct the 'inputs' dictionary for BioBlend
        inputs = {}
        steps: dict = workflow_details['steps']
        for step_id, step_details in steps.items():

            # Check if this step is an input step and is in the form data
            form_value = form_data.get(step_id)
            if form_value is None:
                continue

            # Differentiate between data inputs and parameter inputs
            if step_details['type'] in ['data_input', 'data_collection_input']:
                src = 'hdca' if step_details['type'] == 'data_collection_input' else 'hda'
                inputs[step_id] = {'src': src, 'id': form_value}

            elif step_details['type'] == 'parameter_input':
                inputs[step_id] = form_value

        logger.info(f"inputs applied: {inputs}")

        # Run the entire synchronous workflow execution and tracking in a thread pool
        invocation_id, report, intermediate_outputs, final_outputs = await workflow_manager.run_track_workflow(
            inputs=inputs,
            workflow=workflow_obj,
            history=history_obj,
            ws_manager = ws_manager,
            tracker_id = tracker_id
            )

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
                            "data_type": workflow_manager.gi_object.gi.datasets.show_dataset(ds.id)['data_type']
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
                            ]
                        }
                    )
                
        for ds in intermediate_outputs:
            if isinstance(ds, HistoryDatasetAssociation):
                intermediate_outputs_formatted.append(
                        {
                            "type" : "dataset",
                            "id": ds.id,
                            "name": ds.name,
                            "visible": ds.visible,
                            "peek": workflow_manager.gi_object.gi.datasets.show_dataset(ds.id)['peek'],
                            "data_type": workflow_manager.gi_object.gi.datasets.show_dataset(ds.id)['data_type']
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
                            ]
                        }
                    )
                                        
        return {
                "invocation_id": invocation_id,
                "history_id": history_id,
                "report": report,
                "final_outputs": final_outputs_formatted,
                "intermediate_outputs": intermediate_outputs_formatted
            }
    except Exception as e:
        await ws_manager.broadcast(
            event= SocketMessageEvent.workflow_execute,
            data = {
                "type": SocketMessageType.WORKFLOW_FAILURE,
                "payload" : f"Workflow execution failed: {e}"
            },
            tracker_id = tracker_id
        )
        # Provide a detailed error message for debugging
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {e}")
    
@router.get(
    "/{workflow_id}/details",
    response_model=workflow.WorkflowDetails,
    summary="Get detailed information about the Galaxy workflow",
    tags=["Workflows"]
)
async def get_workflow_details(
    workflow_id: str = Path(..., description="The ID of the Galaxy workflow.")
):
    """
    Shows the details of a workflow including id, created time,
    annotations, published or not, creator, url, license, steps and inputs
    """

    galaxy_client = GalaxyClient(current_api_key.get())
    workflow_manager = WorkflowManager(galaxy_client)

    try:
        workflow_details  = await run_in_threadpool(workflow_manager.gi_object.gi.workflows.show_workflow, workflow_id)
       
        return {
            "id": workflow_details.get("id"),
            "tags": workflow_details.get("tags", None),
            "create_time": workflow_details.get("create_time"),
            "annotations": workflow_details.get("annotations", None),
            "published": workflow_details.get("published"),
            "license": workflow_details.get("license", None),
            "galaxy_url": workflow_details.get("url"),
            "creator": workflow_details.get("creator", None),
            "steps": workflow_details.get("steps"),
            "inputs": workflow_details.get("inputs"),
        }
                
    except Exception as e:
        # detailed error responses
        raise HTTPException(status_code = 500 , detail= f'Show workflow failed {e}')