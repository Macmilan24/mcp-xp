from __future__ import annotations

import logging
import time
from typing import List, Union, Tuple, Optional, Literal, Dict
import json
import asyncio
import traceback

from rapidfuzz import process, fuzz
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
from enum import Enum

load_dotenv()

from sys import path
path.append('.')

from bioblend.galaxy.objects.wrappers import Workflow, Invocation, History, Dataset, DatasetCollection
from bioblend.galaxy.toolshed import ToolShedClient

from app.bioblend_server.galaxy import GalaxyClient
from app.bioblend_server.executor.tool_manager import ToolManager
from app.bioblend_server.executor.form_generator import WorkflowFormGenerator

from app.api.socket_manager import SocketManager, SocketMessageType, SocketMessageEvent

class JobState(str, Enum):
    NEW = "new"
    RESUBMITTED = "resubmitted"
    UPLOAD = "upload"
    WAITING = "waiting"
    QUEUED = "queued"
    RUNNING = "running"
    OK = "ok"
    ERROR = "error"
    FAILED = "failed"
    PAUSED = "paused"
    DELETING = "deleting"
    DELETED = "deleted"
    STOPPING = "stop"
    STOPPED = "stopped"
    SKIPPED = "skipped"

class WorkflowManager:
    """Workflow manager class for managing Galaxy workflows"""

    def __init__(self, galaxy_client: GalaxyClient):
        self.galaxy_client = galaxy_client

        self.gi_object=self.galaxy_client.gi_object 
        self.gi_admin = self.galaxy_client.gi_admin # For administrative functionalitites like toolshed instantiation and tool installing
        self.toolshed=ToolShedClient(self.gi_admin.gi)    # Toolshed instance

        self.tool_manager = ToolManager(galaxy_client = self.galaxy_client)
        self.data_manager = self.tool_manager.data_manager
        self.log = logging.getLogger(self.__class__.__name__)

    async def upload_workflow(self, workflow_json: dict, ws_manager: SocketManager = None, tracker_id: str = None):
        """Upload workflow from a ga file json."""

        if ws_manager:
            await ws_manager.broadcast(
                event = SocketMessageEvent.workflow_upload,
                data = {
                    "type": SocketMessageType.UPLOAD_WORKFLOW,
                    "payload": {"message": "Workflow upload started, checking and installing missing tools."}
                    },
                tracker_id=tracker_id
            )
        self.log.info("Workflow upload started, checking and installing missing tools.")
        # Check if the tools are installed and install all missing tools
        try:
            workflow_steps=workflow_json.get('steps', None)
            if workflow_steps:
                for step in workflow_steps.values():
                    await self.tool_check_install(step, ws_manager, tracker_id)
        except Exception as e:
            if ws_manager:
                await ws_manager.broadcast(
                    event = SocketMessageEvent.workflow_upload,
                    data = {
                        "type": SocketMessageType.UPLOAD_FAILURE,
                        "payload": {"message": f"Error installing missing tools in the uploaded workflow: {e}"}
                        },
                    tracker_id=tracker_id
                )
            self.log.error(f"Error installing missing tools in the uploaded workflow: {e} traceback:{traceback.format_exc()}")
            
            # return {"error": f"Error installing missing tools in the uploaded workflow: {e}"}
            
        # Reload the tool box after tools are installed
        await asyncio.to_thread(self.gi_admin.gi.config.reload_toolbox)

        workflow = await asyncio.to_thread(
            self.gi_object.workflows.import_new, 
            src=workflow_json, 
            publish=False
            )

        # Check if the workflow is considered runnable by the instance
        if workflow.is_runnable:
            if ws_manager:
                self.log.info("workflow Uploaded successfully")
                await ws_manager.broadcast(
                    event = SocketMessageEvent.workflow_upload,
                    data = {
                        "type": SocketMessageType.UPLOAD_COMPLETE,
                        "payload": {"message": "Workflow successfully uploaded."}
                        },
                    tracker_id=tracker_id
                    )
        else:
            self.log.error("Workflow is not runnable.")
    
    def get_worlflow_by_name(self, name: str, score_cutoff: int = 70) -> Workflow | None:
        """Get workflow by its name (fuzzy match)."""
        workflow_list = self.gi_object.gi.workflows.get_workflows()
        # Build a dict that maps each workflow name to the full dict
        name_to_wf = {wf["name"]: wf for wf in workflow_list}

        match = process.extractOne(
            name,
            name_to_wf.keys(),          # list of names
            scorer=fuzz.partial_ratio,
            score_cutoff=score_cutoff
        )

        if match is None:               # nothing above the cutoff
            return None

        wf_dict = name_to_wf[match[0]]
        return self.gi_object.workflows.get(wf_dict["id"])
    
    def get_workflow_by_id(self, id: str)-> Workflow | None:
        """Get tool by its id"""
        
        try:
            workflow= self.gi_object.workflows.get(id)
            return workflow 
        except:
            return None
        
    def tool_exists(self, step: dict) -> bool:
        """Checks if a specific version of a tool is installed within the galaxy instance"""

        tool_id = step.get('tool_id')
        if not tool_id:
            return True

        try:
            tool = self.gi_admin.gi.tools.show_tool(tool_id)
        except Exception:
            return False
        if not tool:
            return False

        # Grab repository info (None if local tool)
        step_repo = step.get('tool_shed_repository')
        tool_repo = tool.get('tool_shed_repository')
        # If the step was defined to come from a Tool Shed, enforce that
        if step_repo:
            # tool must also be from a Tool Shed
            if not tool_repo:
                return False
            # revisions must match exactly
            if tool_repo.get('changeset_revision') != step_repo.get('changeset_revision'):
                return False

        # If step_repo is None, we’re happy with any existing tool (shed or local)
        return True

    # Function that installs tools missing in the galaxy instance for the workflow invocation
    # Need  administrator api        
    async def tool_check_install(self, step: dict, ws_manager: SocketManager, tracker_id: str):
        """Check and install if a tool in a workflow is missing"""

        # Recurse into subworkflow steps
        if step.get('type') == 'subworkflow':
            for sub_step in step['subworkflow']['steps'].values():
                await self.tool_check_install(sub_step, ws_manager, tracker_id)
            return  # Skip install for subworkflow container itself

        # Skip steps without a tool_id
        if not step.get('tool_id'):
            return

        # Check if tool is already installed
        if not self.tool_exists(step):
            self.log.info(f"Installing tool for step {step['id']}")
            toolshed_info = step['tool_shed_repository']
            try:
                install_result = self.toolshed.install_repository_revision(
                    tool_shed_url=f'https://{toolshed_info["tool_shed"]}',
                    name=toolshed_info["name"],
                    owner=toolshed_info["owner"],
                    changeset_revision=toolshed_info["changeset_revision"],
                    install_tool_dependencies=True,
                    install_repository_dependencies=True,
                    install_resolver_dependencies=True,
                    tool_panel_section_id=None,
                    new_tool_panel_section_label=None
                )

                # Since reloading specific tools is not working, gone with refreshing the full toolbox after all the tools are installed 
                # self.gi_object.gi.tools.reload(step['tool_id'])

                if isinstance(install_result, dict):
                    self.log.info(f"status: {install_result.get('status')}, message: {install_result.get('message')}")
                    if ws_manager:
                        await ws_manager.broadcast(
                            event= SocketMessageEvent.workflow_upload,
                            data = {
                                "type": SocketMessageType.TOOL_INSTALL,
                                "payload": {"message": f"{install_result.get('message')}"}
                            },
                            tracker_id = tracker_id
                        )

                elif isinstance(install_result, list):
                    for repo_info in install_result:
                        status = repo_info.get('status', 'unknown')
                        error_msg = repo_info.get('error_message', 'None') if status != 'installed' else 'None'

                        self.log.info(
                            f"Tool install result - Name: {repo_info.get('name', 'N/A')}, "
                            f"Owner: {repo_info.get('owner', 'N/A')}, "
                            f"Status: {status}, "
                            f"Error: {error_msg}"
                        )
                        if ws_manager:
                            await ws_manager.broadcast(
                                event=SocketMessageEvent.workflow_upload,
                                data = {
                                    "type" : SocketMessageType.TOOL_INSTALL,
                                    "payload" : {
                                        "name": repo_info.get('name', 'N/A'),
                                        "owner": repo_info.get('owner', 'N/A'),
                                        "status": status,
                                        "error" : error_msg
                                                }
                                },
                                tracker_id = tracker_id
                                )

            except Exception as e:
                self.log.error(f"Failed to install tool '{toolshed_info['name']}': {str(e)}  traceback:{traceback.format_exc()}")
                raise
        else:
            self.log.info(f"Tool found for step {step['id']}, skipping installation")

    def get_workflow_io(self, workflow: Workflow)-> dict:
        """get input structure of a workflow to be filled"""
        
        return dict.fromkeys(workflow.inputs.keys())

    def _map_inputs_to_steps(self, name: str, inputs: dict, steps: dict) -> dict:
        """
        Map workflow inputs to their corresponding input steps.

        Args:
            inputs (dict): workflow["inputs"]
            steps (dict): workflow["steps"]

        Returns:
            dict: A mapping of input labels to detailed step info.
        """
        
        input_map = {
            "name": name
        }

        for input_id, input_info in inputs.items():
            step = steps.get(str(input_id))
            input_map[input_id] = {
                "Label": input_info.get("label"),
                "step_id": step.get('id'),
                "input_type": step.get('type'),
                "annotation": step.get('annotation', ''),
                "tool_inputs": step.get('tool_inputs', {}),
                "uuid": input_info.get("uuid", "")
            }

        return input_map
    
    ## TODO : VALIDATION STEP needs improvement.
    def validate_input_to_tool(self, mapped_input: dict, workflow: dict) -> dict:
        """Validate the input HTML form against the first tool it is connected to."""
        steps = workflow.get("steps", {})
        input_tools = set()
        tool_xml_cache = {}

        # Map source step ID to connected tool params and tool IDs
        input_connections_map = {}  # {source_step_id: [(tool_id, param_name)]}

        for step in steps.values():
            tool_id = step.get("tool_id")
            if not tool_id:
                continue

            input_steps = step.get("input_steps", {})
            for param_name, conn_info in input_steps.items():
                source_step = str(conn_info.get("source_step"))
                input_connections_map.setdefault(source_step, []).append((tool_id, param_name))
                input_tools.add(tool_id)

        self.log.info(f"Tools that directly take input from workflow inputs are: {input_tools}")
        self.log.info("Collecting tool XML for workflow mapped input")

        # Pre-fetch XML definitions for all involved tools
        for tool_id in input_tools:
            try:
                xml = asyncio.run(self.tool_manager.get_tool_xml(tool_id))
                if xml:
                    tool_xml_cache[tool_id] = ET.fromstring(xml)
                else:
                    self.log.warning(f"Could not retrieve XML for tool_id: {tool_id}")
            except ET.ParseError as e:
                self.log.error(f"Failed to parse XML for tool {tool_id}: {e}")

        # Perform validation for each mapped input
        for input_id, input_details in mapped_input.items():
            if input_id == "name":
                continue

            connections = input_connections_map.get(str(input_id), [])
            for tool_id, param_name in connections:
                root = tool_xml_cache.get(tool_id)
                if root is None:
                    continue

                param_node = root.find(f".//inputs/param[@name='{param_name}']")
                if param_node is not None:
                    validation_info = {
                        "tool_id": tool_id,
                        "tool_param_name": param_name,
                        "type": param_node.get("type"),
                        "format": param_node.get("format", "data"),
                        "label": param_node.get("label"),
                        "help": (param_node.findtext("help") or "").strip(),
                    }

                    input_details.setdefault("validation", []).append(validation_info)
                    self.log.info(f"Validated input '{input_id}' against tool '{tool_id}' param '{param_name}'")
                else:
                    self.log.warning(f"Could not find param '{param_name}' in tool XML for tool_id '{tool_id}'")

        return mapped_input

    def build_input(self, workflow: Workflow, history: History)-> str:
        """builds workflow input html form from the workflow inputs and input steps"""
        # map inputs to input steps of the workflow to get information from both
        workflow= self.gi_object.gi.workflows.show_workflow(workflow_id=workflow.id)
        
        mapped_input = self._map_inputs_to_steps(name=workflow['name'], inputs=workflow['inputs'], steps=workflow['steps'])

        # builds a html form from the mapped input payload and validate it
        mapped_input = self.validate_input_to_tool(mapped_input, workflow)
        form_generator = WorkflowFormGenerator(mapped_workflow= mapped_input, data_manager= self.data_manager, workflow= workflow, history= history)
        html_form = form_generator._build_html()
        return html_form
    
    async def track_invocation(self, 
                            invocation: Invocation,
                            tracker_id: str = None,
                            ws_manager: SocketManager = None,
                            base_extension: int = 30,
                            initial_wait: int = 120,
                            invocation_check: bool = False
                        ) -> Tuple[
                            Dict[str,List],
                            Literal['Pending', 'Failed', 'Complete'],
                            Optional[str]
                        ]:
                            
        """Tracks invocation steps and waits for the invocation reaches a terminal state and returns with the invocation results""" 
        
        # Safety: Semaphore to limit concurrent API calls (prevent overwhelming Galaxy server)
        semaphore = asyncio.Semaphore(15)  # Adjust based on server capacity
        
        # Helper: Fetch with semaphore
        async def _fetch_with_semaphore(coro):
            async with semaphore:
                try:
                    return await coro
                except Exception as e:  # Catch BioBlend errors like GalaxyRequestError
                    self.log.error(f"API fetch error: {e}")
                    return None
        
        # # Helper: Wait for dataset safe
        # async def _wait_for_dataset_safe(dataset_id):
        #     try:
        #         coro = asyncio.to_thread(
        #             self.gi_object.datasets.get, dataset_id
        #         )
        #         return await _fetch_with_semaphore(coro)
        #     except Exception as e:
        #         self.log.error(f"Failed to wait for dataset {dataset_id}: {e}")
        #         return None
        # async def _wait_for_dataset_collection_safe(collection_id):
        #     try:
        #         coro = asyncio.to_thread(
        #         self.gi_object.dataset_collections.get, collection_id
        #         )
        #         return await _fetch_with_semaphore(coro)
        
        
        #     except Exception as e:
        #         self.log.error(f"Failed to wait for collection {collection_id}: {e}")
        #         return None
        
        async def cancel_invocation_background(invocation: Invocation):
            await asyncio.to_thread(invocation.cancel)
            self.log.info(f"Invocation {invocation.id} has been cancelled.")
            
        previous_states = {}
        invocation_outputs ={
            "output_datasets": [],
            "collection_datasets": []
        }

        start_time = time.time()
        inv = await _fetch_with_semaphore(asyncio.to_thread(
            self.gi_object.gi.invocations.show_invocation, invocation_id=invocation.id
        ))
        num_steps = len(inv["steps"])
        num_completed_steps = 0
        estimated_wait = 60 * num_steps  # assume ~60s per step
        effective_initial_wait = max(estimated_wait, initial_wait)  # Renamed for clarity
        deadline = start_time + effective_initial_wait
        max_extension = effective_initial_wait // 2
        poll_interval = max(3, min(5, num_steps // 10))  # Optimization: Start higher for large workflows

        invocation_state_result = "Failed"

        # Explicit initial check before loop for already-completed/failed workflows

        if inv is None:
            self.log.error("Failed to fetch initial invocation state.")
            if invocation_check:
                return invocation_outputs, invocation_state_result, None
            else:
                return invocation_outputs
        
        invocation_update_time = inv.get("update_time", None)
        invocation_state = inv.get("state")
        
        self.log.debug(json.dumps(inv, indent=4))  # Demoted to debug to reduce I/O

        if invocation_state in ("failed", "error"):
            invocation_state_result = "Failed"
            
            
            self.log.error("workflow invocation has failed.")
            
            if ws_manager:
                ws_data = {
                    "type": SocketMessageType.INVOCATION_FAILURE,
                    "payload": {"message": "Invocation failed or has error"}
                }
                await ws_manager.broadcast(
                    event=SocketMessageEvent.workflow_execute, 
                    data=ws_data, 
                    tracker_id=tracker_id
                )
            if invocation_check:
                return invocation_outputs, invocation_state_result, invocation_update_time
            else:
                return invocation_outputs

        # Assume 'succeeded' or similar for terminal success; adjust based on your API
        if invocation_state in ("succeeded", "completed", "ok"):  # Added 'ok' as common Galaxy state for success
            self.log.info("Invocation already completed.")
            step_jobs_coro = asyncio.to_thread(
                self.gi_object.gi.invocations.get_invocation_step_jobs_summary, invocation_id=invocation.id
            )
            step_jobs = await _fetch_with_semaphore(step_jobs_coro)
            if step_jobs is None:
                step_jobs = []  # Safety fallback
            self.log.debug(json.dumps(step_jobs, indent=4))
            # Proceed to process steps (will set all_ok and collect below)
        else:
            # Not terminal; fetch steps for initial poll
            step_jobs_coro = asyncio.to_thread(
                self.gi_object.gi.invocations.get_invocation_step_jobs_summary, invocation_id=invocation.id
            )
            step_jobs = await _fetch_with_semaphore(step_jobs_coro)
            if step_jobs is None:
                step_jobs = []  # Safety fallback
            self.log.debug(json.dumps(step_jobs, indent=4))

        # Initial processing of steps (handles already-completed case)
        all_ok = True
        progress_made = False
        # newly_completed = []
        has_error = False
        step_index = 1

        for step in step_jobs:
            step_id: str = step.get('id')
            states: dict = step.get('states', {})
            populated: str = step.get("populated_state", "Unknown")
            
            if populated in (JobState.FAILED) or populated not in (JobState.OK, JobState.WAITING, JobState.SKIPPED, JobState.RESUBMITTED, JobState.RUNNING, JobState.QUEUED):
                self.log.error(f"Step {step_index} with id {step_id} {populated}; cancelling invocation")
                current_state=JobState.ERROR
            else:
                # Pending/running/other
                if states.get(JobState.RUNNING, 0) > 0:
                    current_state = JobState.RUNNING
                elif states.get(JobState.ERROR, 0) > 0 or states.get(JobState.FAILED, 0) > 0:
                    current_state = JobState.ERROR
                elif states.get(JobState.SKIPPED, 0) > 0:
                    current_state = JobState.OK
                elif states.get(JobState.OK, 0) > 0:
                    current_state = JobState.OK
                else:
                    current_state = 'pending'

            # Log only on transitions (unchanged)
            prev = previous_states.get(step_id)
            
            if current_state != prev:   
                if current_state == JobState.RUNNING:
                    self.log.debug(f"Step {step_index} with Job id {step_id} started running.")
                    previous_states[step_id] = current_state
                    progress_made = True
                    
                        
                if current_state == JobState.OK:
                    self.log.info(f"Step {step_index} with Job id {step_id} has completed succesfully.")
                    previous_states[step_id] = current_state
                    progress_made = True
                    # Add to number of completed steps.
                    num_completed_steps +=1
                    
                    # Broadcast information
                    if ws_manager:
                        ws_data = {
                            "type": SocketMessageType.INVOCATION_STEP_UPDATE,
                            "payload": {
                                "Workflow_steps": num_steps,
                                "Completed step_index": step_index,
                                "Completed steps": num_completed_steps 
                            }
                        }
                        await ws_manager.broadcast(
                            event=SocketMessageEvent.workflow_execute,
                            data=ws_data,
                            tracker_id=tracker_id
                        )

            if current_state == JobState.ERROR:
                self.log.error(f"Step {step_index} with Job id {step_id} failed; cancelling invocation.")
                all_ok = False  # Ensure all_ok is False on error
                
                if ws_manager:
                    ws_data = {
                        "type": SocketMessageType.INVOCATION_FAILURE,
                        "payload": {"message": "Invocation failed or has error"}
                    }
                    await ws_manager.broadcast(
                        event=SocketMessageEvent.workflow_execute, 
                        data=ws_data, 
                        tracker_id=tracker_id
                    )
                    
                asyncio.create_task(cancel_invocation_background(invocation))
                error_occurred = True
                has_error = True
                break

            if current_state != JobState.OK:
                all_ok = False
            step_index += 1

        if has_error:
            # Early exit if error (collected prior steps)
            invocation_state_result = "Failed"
            if invocation_check:
                return invocation_outputs, invocation_state_result, invocation_update_time
            else:
                return invocation_outputs

        if all_ok:
            self.log.info("All steps completed successfully.")
            invocation_state_result = "Complete"
            
            if ws_manager:
                ws_data = {
                    "type": SocketMessageType.INVOCATION_COMPLETE,
                    "payload": {"message": "All steps completed successfully"}
                }
                await ws_manager.broadcast(
                    event=SocketMessageEvent.workflow_execute,
                    data=ws_data,
                    tracker_id=tracker_id
                )
            
            # Final full collection if completed (overrides partial for consistency - BioBlend opt)
            final_inv_coro = asyncio.to_thread(
                self.gi_object.gi.invocations.show_invocation, invocation_id=invocation.id
            )
            final_inv = await _fetch_with_semaphore(final_inv_coro)
            
            if final_inv and 'outputs' in final_inv:
                output_ids = []
                for label, output_info in final_inv.get('outputs', {}).items():
                    if 'id' in output_info:
                        output_ids.append(output_info['id'])
                    self.log.debug(f"Found output '{label}': {output_info}")

            if final_inv and 'output_collections' in final_inv:
                collection_output_ids = []
                for label, output_info in final_inv.get('output_collections', {}).items():
                    if 'id' in output_info:
                        collection_output_ids.append(output_info['id'])
                    self.log.debug(f"Found output '{label}': {output_info}")
                
                # Collect final output ids.
                invocation_outputs = {
                        "output_datasets": output_ids,
                        "collection_datasets": collection_output_ids
                    }
                self.log.info(f"Collected {len(output_ids)} dataset outputs and {len(collection_output_ids)} collection outputs")
                    
                #Update invocation update time
                invocation_update_time = final_inv.get("update_time", None)
            
            else:
                self.log.warning("No 'outputs' in final invocation; falling back to partial.")
            
            # For already-completed, we return here (no loop entered)
            if invocation_check:
                return invocation_outputs, invocation_state_result, invocation_update_time
            else:
                return invocation_outputs

        # If not completed/failed initially, enter polling loop
        while True:
            
            # Reset for this poll cycle
            num_completed_steps = 0
            progress_made = False
            # newly_completed = []
            has_error = False
            all_ok = True
            step_index = 1

            inv_coro = asyncio.to_thread(
                self.gi_object.gi.invocations.show_invocation, invocation_id=invocation.id
            )
            inv = await _fetch_with_semaphore(inv_coro)
            if inv is None:
                self.log.error("Failed to fetch invocation state during polling.")
                break
            invocation_state = inv.get("state")
            self.log.debug(json.dumps(inv, indent=4))
            if invocation_state in ("failed", "error"):
                self.log.error("workflow invocation has failed.")
                invocation_state_result = "Failed"
                if ws_manager:
                    ws_data = {
                        "type": SocketMessageType.INVOCATION_FAILURE,
                        "payload": {"message": "Invocation failed or has error"}
                    }
                    await ws_manager.broadcast(
                        event=SocketMessageEvent.workflow_execute, 
                        data=ws_data, 
                        tracker_id=tracker_id
                    )
                break
            
            invocation_state_result = "Pending"
            
            step_jobs_coro = asyncio.to_thread(
                self.gi_object.gi.invocations.get_invocation_step_jobs_summary, invocation_id=invocation.id
            )
            step_jobs = await _fetch_with_semaphore(step_jobs_coro)
            if step_jobs is None:
                step_jobs = []  # Safety fallback
            self.log.debug(json.dumps(step_jobs, indent=4))

            for step in step_jobs:
                step_id: str = step.get('id')
                states: dict = step.get('states', {})
                populated: str = step.get("populated_state", "Unknown")
                
                if populated in (JobState.FAILED) or populated not in (JobState.OK, JobState.WAITING, JobState.SKIPPED, JobState.RESUBMITTED, JobState.RUNNING, JobState.QUEUED):
                    self.log.error(f"Step {step_index} with id {step_id} {populated}; cancelling invocation")
                    current_state=JobState.ERROR
                else:
                    # Pending/running/other
                    if states.get(JobState.RUNNING, 0) > 0:
                        current_state = JobState.RUNNING
                    elif states.get(JobState.ERROR, 0) > 0 or states.get(JobState.FAILED, 0) > 0:
                        current_state = JobState.ERROR
                    elif states.get(JobState.SKIPPED, 0) > 0:
                        current_state = JobState.OK
                    elif states.get(JobState.OK, 0) > 0:
                        current_state = JobState.OK
                    else:
                        current_state = 'pending'

                # Log only on transitions (unchanged)
                prev = previous_states.get(step_id)
                if current_state != prev:   
                    if current_state == JobState.RUNNING:
                        self.log.debug(f"Step {step_index} with Job id {step_id} started running.")
                        previous_states[step_id] = current_state
                        progress_made = True
                        
                            
                    if current_state == JobState.OK:
                        self.log.info(f"Step {step_index} with Job id {step_id} has completed succesfully.")
                        previous_states[step_id] = current_state
                        progress_made = True
                        # Add to number of completed steps.
                        num_completed_steps +=1
                        
                        # Broadcast information
                        if ws_manager:
                            ws_data = {
                                "type": SocketMessageType.INVOCATION_STEP_UPDATE,
                                "payload": {
                                    "Workflow_steps": num_steps,
                                    "Completed step_index": step_index,
                                    "Completed steps": num_completed_steps 
                                }
                            }
                            await ws_manager.broadcast(
                                event=SocketMessageEvent.workflow_execute,
                                data=ws_data,
                                tracker_id=tracker_id
                            )

                if current_state == JobState.ERROR:
                    self.log.error(f"Step {step_index} with id {step_id} failed; cancelling invocation.")
                    all_ok = False  # Ensure all_ok is False on error
                    
                    if ws_manager:
                        ws_data = {
                            "type": SocketMessageType.INVOCATION_FAILURE,
                            "payload": {"message": "Invocation failed or has error"}
                        }
                        await ws_manager.broadcast(
                            event=SocketMessageEvent.workflow_execute, 
                            data=ws_data, 
                            tracker_id=tracker_id
                        )
                    asyncio.create_task(cancel_invocation_background(invocation))
                    has_error = True
                    break

                if current_state != JobState.OK:
                    all_ok = False
                step_index += 1
            
            if has_error:
                invocation_state_result = "Failed"
                break
            if all_ok:
                self.log.info("All steps completed successfully.")
                invocation_state_result = "Complete"
                
                if ws_manager:
                    ws_data = {
                        "type": SocketMessageType.INVOCATION_COMPLETE,
                        "payload": {"message": "All steps completed successfully"}
                    }
                    await ws_manager.broadcast(
                        event=SocketMessageEvent.workflow_execute,
                        data=ws_data,
                        tracker_id=tracker_id
                    )
                
                # Final full collection (as above)
                final_inv_coro = asyncio.to_thread(
                    self.gi_object.gi.invocations.show_invocation, invocation_id=invocation.id
                )
                final_inv = await _fetch_with_semaphore(final_inv_coro)
                if final_inv and 'outputs' in final_inv:
                    output_ids = []
                    for label, output_info in final_inv['outputs'].items():
                        if 'id' in output_info:
                            output_ids.append(output_info['id'])
                        self.log.debug(f"Found output '{label}': {output_info}")

                if final_inv and 'output_collections' in final_inv:
                    collection_output_ids = []
                    for label, output_info in final_inv['output_collections'].items():
                        if 'id' in output_info:
                            collection_output_ids.append(output_info['id'])
                        self.log.debug(f"Found output '{label}': {output_info}")

                    # Collect output ids.
                    invocation_outputs = {
                        "output_datasets": output_ids,
                        "collection_datasets": collection_output_ids
                    }
                    self.log.info(f"Collected {len(output_ids)} dataset outputs and {len(collection_output_ids)} collection outputs")
                    
                    # Update invocation update time.
                    invocation_update_time = final_inv.get("update_time", None)
                else:
                    self.log.warning("No 'outputs' in final invocation; falling back to partial.")
                
                break
            
            now = time.time()
            if progress_made:
                # Extend deadline if progress was made, capped by hard max_wait
                extension = min(base_extension, max_extension)
                deadline = max(deadline + extension, start_time + effective_initial_wait)

            if now > deadline:
                self.log.error(f"Invocation timed out after {int(now - start_time)} seconds.")
                
                if ws_manager:
                    ws_data = {
                        "type": SocketMessageType.INVOCATION_FAILURE,
                        "payload": {"message": "Invocation timed out"}
                    }
                    await ws_manager.broadcast(
                        event=SocketMessageEvent.workflow_execute, 
                        data=ws_data, 
                        tracker_id=tracker_id
                    )
                    
                asyncio.create_task(cancel_invocation_background(invocation))
                self.log.info(f"invocation: {invocation.id} cancelled")
                invocation_state_result = "Failed"
                break
            
            # Adaptive polling interval based on progress made (fixed assignment)
            if progress_made:
                poll_interval = 0.5
            else:
                poll_interval = min(5, poll_interval * 1.5)  # Optimization: Exponential backoff to 5s max for no-progress stalls
            
            await asyncio.sleep(poll_interval) 
        
        if invocation_check:
            return invocation_outputs, invocation_state_result, invocation_update_time
        else:
            return invocation_outputs

    def invoke_workflow(self, inputs: dict, workflow: Workflow, history: History ) -> Invocation:
        """
        Invoke a Galaxy workflow with specified inputs.
        
        :param workflow: workflow object
        :param inputs: Workflow inputs parameters mapping
        :param history: history object

        :return: The workflow invocation  
        """

        invoke = self.gi_object.gi.workflows.invoke_workflow(workflow_id=workflow.id, 
                                                                  inputs=inputs,
                                                                  history_id=history.id,
                                                                  parameters_normalized=True,
                                                                  require_exact_tool_versions=False)
        time.sleep(2)

        invocation=self.gi_object.invocations.get(invoke['id'])
        
        return invocation
    
    def _make_result(self, invocation: Invocation, outputs: list[Union[Dataset, DatasetCollection]]):
        """Result report of the workflow invocation"""
        invocation_id = invocation.id
        invocation_report = self.gi_object.gi.invocations.get_invocation_report(invocation.id)
        final_outputs=[]
        intermediate_outputs=[]

        # Classify the workflow invocation outputs as visible and hidden(i.e final outputs and intermediate step outputs)
        for output in outputs:
            if output.visible:
                final_outputs.append(output)
            else:
                intermediate_outputs.append(output)
        return invocation_id, invocation_report, intermediate_outputs, final_outputs
        
    async def run_track_workflow(self, inputs: dict, workflow: Workflow , history: History, ws_manager: SocketManager, tracker_id: str):
        """
        Start a workflow invocation, 
        track that invocation,
        get intermediate and final outputs, 
        and prepare a result
        """
        # start invocation
        invocation = await asyncio.to_thread(
            self.invoke_workflow, inputs = inputs, workflow = workflow, history = history
            )
        # track invocation steps and collect outputs
        outputs= await self.track_invocation(invocation, tracker_id, ws_manager) # can set base time extenstion and max_wait here
        # prepare workflow invocation results
        invocation_id, invocation_report, intermediate_outputs, final_outputs = self._make_result(invocation, outputs)
        
        # Return result of workflow invocation
        return invocation_id, invocation_report, intermediate_outputs, final_outputs