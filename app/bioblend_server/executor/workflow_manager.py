# tool_manager.py
from __future__ import annotations

import logging
import time
from typing import Dict, List, Any, Optional
import os
import json

from rapidfuzz import process, fuzz
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()

from sys import path
path.append('.')

from app.bioblend_server.galaxy import GalaxyClient
from bioblend.galaxy.objects.wrappers import Workflow, Invocation, History, Dataset
from bioblend.galaxy.toolshed import ToolShedClient


class WorkflowManager:

    def __init__(self):
        self.galaxy_client = GalaxyClient()

        # galaxy objects instance
        self.gi_object=self.galaxy_client.gi_object

        # Toolshed instance
        self.toolshed=ToolShedClient(self.gi_object.gi) 
        self.log = logging.getLogger(self.__class__.__name__)

    def upload_workflow(self, path: str)-> Workflow:
        with open(path, 'r') as f:
            workflow_json = json.loads(f.read())

           # Check if the tools are installed and install all missing tools
            workflow_steps=workflow_json.get('steps', None)
            if workflow_steps:
                for step in workflow_steps:
                    self.tool_check_install(step)
        
        workflow: Workflow= self.gi_object.workflows.import_new(src=workflow_json, publish=False)

        # Check if the workflow is considered runnable by the instance
        if workflow.is_runnable:
            return workflow
        else:
            return {'error': 'uploaded workflow is not runnable'}
    
    def get_worlflow_by_name(self, name: str, score_cutoff: int = 80) -> Workflow | None:
        workflow_list= self.gi_object.gi.workflows.get_workflows()

        match = process.extractOne(
            name,
            workflow_list,
            key=lambda w: w["name"],
            scorer=fuzz.partial_ratio,
            score_cutoff=score_cutoff
        )
        return self.gi_object.workflows.get(match[0]['id']) if match else None
    
    def get_workflow_by_id(self, id: str)-> Workflow | None:
        try:
            workflow= self.gi_object.workflows.get(id)
            return workflow 
        except:
            return None
        
    def tool_exists(self, step: dict) -> bool:
        tool_id = step.get('tool_id')
        if not tool_id:
            return True

        try:
            tool = self.gi_object.gi.tools.show_tool(tool_id)
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
    def tool_check_install(self, step: dict):
        # Recurse into subworkflow steps
        if step.get('type') == 'subworkflow':
            for sub_step in step['subworkflow']['steps'].values():
                self.tool_check_install(sub_step)
            return  # Skip install for subworkflow container itself

        # Skip steps without a tool_id
        if not step.get('tool_id'):
            return

        # Check if tool is already installed
        if not self.tool_exists(step):
            self.log.info(f"Installing tool for step {step['id']}")
            toolshed_info = step['tool_shed_repository']
            try:
                install_result = self.gi_object.gi.toolshed.install_repository_revision(
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

                self.gi_object.gi.tools.reload(step['tool_id'])

                for repo_info in install_result:
                    status = repo_info.get('status', 'unknown')
                    error_msg = repo_info.get('error_message', 'None') if status != 'installed' else 'None'

                    self.log.info(
                        f"Tool install result - Name: {repo_info.get('name', 'N/A')}, "
                        f"Owner: {repo_info.get('owner', 'N/A')}, "
                        f"Status: {status}, "
                        f"Error: {error_msg}"
                    )


            except Exception as e:
                self.log.error(f"Failed to install tool '{toolshed_info['name']}': {str(e)}")
        else:
            self.log.info(f"Tool found for step {step['id']}, skipping installation")


    def map_inputs_to_steps(self, inputs: dict, steps: dict) -> dict:
        """
        Map workflow inputs to their corresponding input steps.

        Args:
            inputs (dict): workflow["inputs"]
            steps (dict): workflow["steps"]

        Returns:
            dict: A mapping of input labels to detailed step info.
        """
        input_map = {}

        for input_id, input_info in inputs.items():
            step = steps.get(str(input_id))
            if not step:
                continue  # Skip if the step does not exist

            input_map[input_id] = {
                "Label": input_info["label"],
                "step_id": step["id"],
                "input_type": step["type"],
                "annotation": step.get("annotation", ""),
                "tool_inputs": step.get("tool_inputs", {}),
                "uuid": input_info.get("uuid", "")
            }

        return input_map

    def build_html(self, fields):
        # TODO: build html form from the the built and mapped strucutre of
        # a workflow from the inputs to input steps of the workflow.
        pass

        # TODO: Why is invocation not working
    def track_invocation(self, invocation: Invocation)-> List[Dataset]:
        """Tracks invocation steps and waits for the invocation reaches a terminal state""" 
        completed_steps=set()
        error_occurred=False
        previous_states={}
        invocation_outputs=[]

        # estimate maxwait based on the number of steps in the workflow?
        # How to dynamically set maxwait?? good enough to detect when the workflow has failed and when it has not
        while True:
            step_jobs = self.gi_object.gi.invocations.get_invocation_step_jobs_summary(invocation_id=invocation.id)
            all_ok = True
            step_index=0
            for step in step_jobs:
                step_id = step['id']
                states = step['states']

                # Simplified state
                if states.get('running') == 1:
                    current_state = 'running'
                elif states.get('ok') == 1:
                    current_state = 'ok'
                elif states.get('error') == 1:
                    current_state = 'error'
                else:
                    current_state = 'other'

                # Log only on transitions to running, ok, or error
                prev = previous_states.get(step_id)
                if current_state != prev and current_state in ('running', 'ok', 'error'):
                    self.log.info(f"Step {step_index} with id {step_id} transitioned to {current_state}")
                    previous_states[step_id] = current_state

                # Capture output exactly once when it first becomes ok
                if current_state == 'ok' and step_id not in completed_steps:
                    job = self.gi_client.jobs.show_job(step_id)
                    outputs = job.get('outputs', {})
                    if outputs:
                        first_output = next(iter(outputs.values()))
                        output_dataset=self.gi_object.datasets.get(first_output['id'])
                        invocation_outputs.append(output_dataset)
                    completed_steps.add(step_id)

                # Handle errors by logging, cancelling, and breaking out
                if current_state == 'error':
                    self.log.error(f"Step {step_index} with id {step_id} failed; cancelling invocation")
                    invocation.cancel()
                    error_occurred = True
                    break

                if current_state != 'ok':
                    all_ok = False
                step_index +=1

            if error_occurred:
                break
            if all_ok:
                self.log.info("All steps completed successfully.")
                break

            time.sleep(2)  # polling interval 

        return invocation_outputs

        # pass

    def invoke_workflow(self):
        pass