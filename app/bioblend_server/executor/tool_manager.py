from __future__ import annotations
import logging
import time
from typing import Dict, List, Any, Optional
import os
from dotenv import load_dotenv
from rapidfuzz import process, fuzz
import requests

load_dotenv()

from sys import path
path.append('.')

from app.bioblend_server.galaxy import GalaxyClient
from bioblend.galaxy.objects.wrappers import Job, History, Tool, Dataset

from app.bioblend_server.executor.form_generator import ToolFormGenerator
from app.bioblend_server.executor.data_manager import DataManager



class ToolManager:
    """
    Minimal, reusable tool-runner for Galaxy.
    Responsibilities
    ----------------
    - Discover tools
    - Build correct input payloads and generate html form for execution UI
    - Execute
    - Wait until done
    - Return outputs & diagnostics
    """

    def __init__(self):
        self.galaxy_client = GalaxyClient()
        self.gi_object=self.galaxy_client.gi_object
        self.log = logging.getLogger(self.__class__.__name__)
        self.data_manager = DataManager()
        self.poll_interval = 2

    def get_tool_by_name(self, name: str, score_cutoff: int = 80) -> Tool| None:
        """
        Return the tool whose *name* fuzzy-matches `name`.
        Only the tool's `name` field is considered.
        """
        tools = self.gi_object.gi.tools.get_tools()
        match = process.extractOne(
            name,
            tools,
            key=lambda t: t["name"],
            scorer=fuzz.partial_ratio,
            score_cutoff=score_cutoff
        )
        self.log.info(f'most similar tool name found: {match[0]['name']}')
        return self.gi_object.tools.get(match[0]['id']) if match else None
    
    def get_tool_by_id(self, id: str) -> Tool | None:
        """
        Return the Tool instance for the given Galaxy tool ID, or None if it does not exist.
        """
        try:
            return self.gi_object.tools.get(id)
        except Exception as e:
            self.log.info(f"Error occured: {e}")
            return None
        
    def get_tool_io (self, tool_id):
        """Get tool i/o details"""
        tool= self.gi_object.gi.tools.show_tool(tool_id=tool_id, io_details=True)
        return tool['inputs']
    
    def get_tool_xml(self, tool_id: str) -> str:
            """Retrieve tool XML to build dynamic HTML form for tool execution by making a direct api call."""
            try:
                api_key = os.getenv("GALAXY_API_KEY")
                base_url = os.getenv("GALAXY_URL")
                if not base_url or not api_key:
                    raise ValueError("Galaxy environment variables are not set")

                _url = f"{base_url.rstrip('/')}/api/tools/{tool_id}/raw_tool_source"
                _headers = {"x-api-key":api_key}

                response = requests.get(url= _url, headers= _headers)
                response.raise_for_status()
                return response.text  # Return raw XML, not JSON
            except Exception as e:
                self.log.error(f"Error getting tool XML: {e}")
                return None
            
    # TODO; May not be needed and maybe let build function handle it fully.
    def generate_state(self, inputs):
        """ Maps states from the tools io details"""
        def handle_input(param):
            mclass = param.get("model_class")
            ptype = param.get("type")

            if mclass == "DataToolParameter":
                # set a dummy dataset value to later be replaced
                return {"src": "hda", "id": "2a4bf9d66c01414a"}  # Id parameter needs to be added to be validated and accepted by the build function.

            elif mclass == "BooleanToolParameter": # A true or false option
                value = param.get('value', None)
                if value:
                    value=bool(value)
                return value

            elif mclass == "TextToolParameter": # A string input
                return param.get("value", "")

            elif mclass == "SelectToolParameter": # A switch case(select one from options)
                options = param.get("options", [])
                if options:
                    return options[0][1] 
                return ""
            
            elif mclass == "IntegerToolParameter": # Integer inputs
                value = param.get('value', None)
                if value:
                    value=int(value)
                return value
            
            elif mclass == "FloatToolParameter": # Float inputs
                value = param.get('value', None)
                if value:
                    value=float(value)
                return value
            
            elif mclass == "FileToolParameter": # File inputs(Uploaded files as input)
                return param.get('value', None)

            elif mclass == "Conditional": # Conditional inputs if else /when
                test_param = param["test_param"]
                selector_value = test_param.get("value", param["cases"][0]["value"])
                current_case_idx = next(
                    (i for i, c in enumerate(param["cases"]) if c["value"] == selector_value),
                    0
                )

                case_inputs = param["cases"][current_case_idx]["inputs"]
                conditional_state = {
                    test_param["name"]: selector_value,
                    "__current_case__": current_case_idx
                }

                for sub_param in case_inputs:
                    conditional_state[sub_param["name"]] = handle_input(sub_param)

                return conditional_state

            elif mclass == "Repeat":
                
                return [self.generate_state(param["inputs"])]

            elif mclass == "Section":
                return self.generate_state(param["inputs"])

            else:
                print(f"[!] Unhandled param type: {mclass}")
                return None

        state = {}
        for param in inputs:
            state[param["name"]] = handle_input(param)
            
        return state

    # Core executor

    def run( 
        self,
        tool_id: str,
        history: History,
        inputs: Optional[Dict[str, Any]] = None,
        ):
        """
        Run a tool and return a structured result.
        """
        inputs = inputs or {}
        # Build the tool payload 
        # (validation step to convert name: value pairs from the io details into state input)

        tool_payload = self._build_payload(tool_id, history, inputs)
        # pprint(tool_payload)
        # Run
        tool: Tool= self.gi_object.tools.get(tool_id, io_details= True)
        tool_execution= self.gi_object.gi.tools.run_tool(tool_id= tool.id, history_id=history.id, tool_inputs = tool_payload )
        # pprint(tool_execution)
        # The job id
        job_id = tool_execution['jobs'][0]['id']
        outputs = tool_execution['outputs']

        output_datasets: List[Dataset] = []

        for output in outputs:
            dataset =  self.gi_object.datasets.get(output['id'])
            output_datasets.append(dataset)

        self.log.info(f"Started job {job_id} for tool {tool_id!r}")

        job=self.wait(job_id) # Wait for job to complete before making result
        return self._make_result(job, output_datasets), output_datasets

    def wait(self, job_id, initial_wait: int = 60 , base_extension: int = 20):
        """Waits for a Galaxy job to finish, with dynamic timeout and progress tracking."""

        previous_state = None
        start_time = time.time()
        deadline = start_time + initial_wait
        max_extension = initial_wait // 2

        while True:
            job = self.gi_object.jobs.get(job_id)
            current_state = job.state

            if current_state != previous_state:
                self.log.info(f"Job {job_id} transitioned to {current_state}")
                previous_state = current_state
                # Extend deadline slightly when progress is detected
                extension = min(base_extension, max_extension)
                deadline = max(deadline + extension, start_time + initial_wait)

            if current_state in {'ok', 'error', 'cancelled'}:
                break

            if time.time() > deadline:
                self.log.error(f"Job {job_id} timed out after {int(time.time() - start_time)} seconds.")
                try:
                    self.gi_object.gi.jobs.cancel_job(job_id=job_id)
                    self.log.warning(f"Job {job_id} cancelled due to timeout.")
                except Exception as e:
                    self.log.warning(f"Failed to cancel job {job_id}: {e}")
                break

            time.sleep(3)  # polling interval

        return job

    # Internal helpers
    def _build_payload(
        self,
        tool_id: str,
        history: History,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Use Galaxy's `build` endpoint so we do not have to reverse-engineer
        the parameter tree.

        """
        # The low-level client is needed for the build endpoint
        payload = self.gi_object.gi.tools.build(
            tool_id=tool_id,
            inputs=inputs,
            history_id=history.id
        )
        return payload['state_inputs']

    def _make_result(self, job: Job, datasets: list[Dataset]) -> Dict[str, Any]:

        job_details= self.gi_object.gi.jobs.show_job(job_id= job.id, full_details=True)
        
        return {
            "dataset": [{d.name, d.id} for d in datasets],
            "state": job.state,
            "stdout": job_details.get("stdout", None),
            "stderr": job_details.get("stderr", None),
            "error": job_details.get("error_message") if job.state == "error" else None,
        }

    # Build tool html form from the tools xml.
    def build_html_form(self, tool: Tool, history: History) -> str | None:
        """
        Builds a dynamic HTML form from the tool's XML definition.
        """
        xml_str = self.get_tool_xml(tool_id= tool.id) # Get tool xml 
        if not xml_str:
            self.log.error(f"Could not retrieve XML for tool {tool.id}")
            raise

        # We only want active, visible datasets
        generator = ToolFormGenerator(xml_str, tool, history)
        html =  generator.build_html()
        return html