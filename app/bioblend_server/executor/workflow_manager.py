# tool_manager.py
from __future__ import annotations

import logging
import time
from typing import Dict, List, Any, Tuple
import os
import json

from rapidfuzz import process, fuzz
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()

from sys import path
path.append('.')

from app.bioblend_server.galaxy import GalaxyClient
from bioblend.galaxy.objects.wrappers import Workflow, Invocation, History, Dataset, DatasetCollection, Step
from bioblend.galaxy.toolshed import ToolShedClient
from app.bioblend_server.executor.data_manager import DataManager

class WorkflowManager:

    def __init__(self):
        self.galaxy_client = GalaxyClient()

        # galaxy objects instance
        self.gi_object=self.galaxy_client.gi_object
        self.data_manager=DataManager()
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
        """
        Checks if a specific version of a tool is installed within the galaxy instance 
        """
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


    def _map_inputs_to_steps(self, inputs: dict, steps: dict) -> dict:
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
            input_map[input_id] = {
                "Label": input_info.get("label"),
                "step_id": step.get('id'),
                "input_type": step.get('type'),
                "annotation": step.get('annotation', ''),
                "tool_inputs": step.get('tool_inputs', {}),
                "uuid": input_info.get("uuid", "")
            }

        return input_map

    def _build_html(self,mapped_workflows: list | dict, history: History) -> str:
        """
        Builds an HTML form from a list of mapped input dictionaries
        (each from map_inputs_to_steps).

        Args:
            mapped_workflows (list)
            history (History): history to select datasets and dataset collections from
        
        Returns:
            str: HTML string.
        """
        if isinstance(mapped_workflows, dict):
            mapped_workflows=[mapped_workflows]

        # Get dataset and dataset collection for option
        datasets, collections = self.data_manager.list_contents(history=history)

        html = ['<form method="POST" enctype="multipart/form-data">']

        for workflow_idx, workflow_inputs in enumerate(mapped_workflows):
            html.append(f"<fieldset><legend>Workflow</legend>")

            for input_id, input_info in workflow_inputs.items():
                label = input_info.get("Label", f"Input {input_id}")
                input_type = input_info.get("input_type", "")
                annotation = input_info.get("annotation", "")
                tool_inputs = input_info.get("tool_inputs", {})
                param_type = tool_inputs.get("parameter_type")
                field_name = f"wf{workflow_idx}_step{input_info['step_id']}"

                html.append(f'<div style="margin-bottom: 1em;">')
                html.append(f'<label for="{field_name}"><strong>{label}</strong></label><br>')
                if annotation:
                    html.append(f'<small>{annotation}</small><br>')

                # DATA INPUTS
                if input_type == "data_input":
                    html.append(f'<select name="{field_name}"  formatrequired>')
                    for option in datasets:
                        html.append(f'<option value="{option['id']}">{option['name']}</option>')
                    html.append('</select><br>')



                elif input_type == "data_collection_input":
                    html.append(f'<select name="{field_name}"  required>')
                    for option in collections:
                        html.append(f'<option value="{option['id']}">{option['name']}</option>')
                    html.append('</select><br>')

                # PARAMETER INPUTS
                elif input_type == "parameter_input":
                    required = "" if tool_inputs.get("optional", False) else "required"

                    #  Boolean
                    if param_type == "boolean":
                        html.append(f'''
                            <select name="{field_name}" {required}>
                                <option value="true">True</option>
                                <option value="false">False</option>
                            </select><br>
                        ''')

                    #  Restricted Choice / Enum
                    elif "restrictions" in tool_inputs:
                        html.append(f'<select name="{field_name}" {required}>')
                        for option in tool_inputs["restrictions"]:
                            html.append(f'<option value="{option}">{option}</option>')
                        html.append('</select><br>')

                    #  Default to Text
                    else:
                        html.append(f'<input type="text" name="{field_name}" {required}><br>')

                # Unknown Input Type
                else:
                    html.append(f'<input type="text" name="{field_name}" placeholder="Unsupported input type"><br>')

                html.append('</div>')  # end of field

            html.append("</fieldset>")

        html.append('<button type="submit">Run</button>')
        html.append("</form>")

        return "\n".join(html)

    def get_input(self, workflow: Workflow, history: History)-> str:
        """builds workflow input html form from the workflow inputs and input steps"""
        # map inputs to input steps of the workflow to get information from both
        workflow= self.gi_object.gi.workflows.show_workflow(workflow_id=workflow.id)
        mapped_input = self._map_inputs_to_steps(inputs=workflow['inputs'], steps=workflow['steps']) 
        # builds a html form from the mapped input payload
        html_form = self._build_html(mapped_workflows = mapped_input, history = history) 
        return html_form

    def track_invocation(self, invocation: Invocation)-> List[Dataset]:
        """Tracks invocation steps and waits for the invocation reaches a terminal state and returns with the invocation results""" 
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
                    job = self.gi_object.gi.jobs.show_job(step_id)
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
    
    def _make_result(self, invocation: Invocation, outputs: list[Dataset], pdf_report: bool = False):
        """Result report of the workflow invocation"""

        # TODO: output datasets should be handled as hidden and shown(i.e visible: True or False)
        if pdf_report:
            invocation_report = self.gi_object.gi.invocations.get_invocation_report_pdf(invocation.id) # Maybe get the report pdf here
        else:
            invocation_report = self.gi_object.gi.invocations.get_invocation_report(invocation.id) # Maybe get the report pdf here
        final_outputs=[]
        intermediate_outputs=[]
        # Classify the workflow invocation outputs as visible and hidden(i.e final outputs and intermediate step outputs)
        for output in outputs:
            if output.visible:
                final_outputs.append(output)
            else:
                intermediate_outputs.append(output)
        return invocation_report, intermediate_outputs, final_outputs
        



# if __name__ == "__main__":

#     wm= WorkflowManager()

#     workflow1= wm.get_workflow_by_id("f2db41e1fa331b3e")
#     history= wm.gi_object.histories.get("4b187121143038ff")
#     file_path = "app/bioblend_server/executor/form_demo.html"

#     html_build = wm.get_input(workflow=workflow1, history=history)
#     with open(file_path, 'wb') as f:
#         f.write(html_build.encode('utf-8'))