from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.responses import HTMLResponse


from app.api.dependencies import get_galaxy_client
from app.bioblend_server.galaxy import GalaxyClient
from app.bioblend_server.executor.tool_manager import ToolManager
from app.bioblend_server.executor.workflow_manager import WorkflowManager

router = APIRouter()


@router.get("/tools/get-by-name", summary="Internal: Find a tool by its name")
async def find_tool_by_name(
    name: str, client: GalaxyClient = Depends(get_galaxy_client)
):
    tool_manager = ToolManager(client)
    tool = tool_manager.get_tool_by_name(name)
    if not tool:
        raise HTTPException(
            status_code=404, detail=f"Tool with name '{name}' not found."
        )

    return {"id": tool.id, "name": tool.name, "description": tool.description}


@router.get(
    "/tools/{tool_id}/form",
    response_class=HTMLResponse,
    summary="Internal: Get HTML form for a tool",
)
async def get_tool_form(
    tool_id: str, history_id: str, client: GalaxyClient = Depends(get_galaxy_client)
):
    tool_manager = ToolManager(client)
    tool = tool_manager.get_tool_by_id(tool_id)
    history = client.gi_object.histories.get(history_id)
    if not tool or not history:
        raise HTTPException(status_code=404, detail="Tool or History not found.")
    return await tool_manager.build_html_form(tool=tool, history=history)


@router.get("/workflows/get-by-name", summary="Internal: Find a workflow by its name")
async def find_workflow_by_name(
    name: str, client: GalaxyClient = Depends(get_galaxy_client)
):
    workflow_manager = WorkflowManager(client)
    workflow = workflow_manager.get_worlflow_by_name(name)
    if not workflow:
        raise HTTPException(
            status_code=404, detail=f"Workflow with name '{name}' not found."
        )
    return {"id": workflow.id, "name": workflow.name}


@router.get(
    "/workflows/{workflow_id}/form",
    response_class=HTMLResponse,
    summary="Internal: Get HTML form for a workflow",
)
async def get_workflow_form(
    workflow_id: str, history_id: str, client: GalaxyClient = Depends(get_galaxy_client)
):
    workflow_manager = WorkflowManager(client)
    workflow = workflow_manager.get_workflow_by_id(workflow_id)
    history = client.gi_object.histories.get(history_id)
    if not workflow or not history:
        raise HTTPException(status_code=404, detail="Workflow or History not found.")
    return await workflow_manager.build_input(workflow=workflow, history=history)
