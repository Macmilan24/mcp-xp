# tool_manager.py
from __future__ import annotations

import logging
import time
from typing import Dict, List, Any, Optional
import os
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()

from sys import path
path.append('.')

from app.bioblend_server.galaxy import GalaxyClient
from bioblend.galaxy.objects.wrappers import Job, History, Tool, Dataset




class ToolManager:
    """
    Minimal, reusable tool-runner for Galaxy.
    Responsibilities
    ----------------
    - Discover tools
    - Build correct input payloads
    - Execute
    - Wait until done
    - Return outputs & diagnostics
    """

    def __init__(self):
        self.galaxy_client = GalaxyClient()
        self.gi_object=self.galaxy_client.gi_object
        self.log = logging.getLogger(self.__class__.__name__)
        self.poll_interval = 2

    # Discovery helpers (optional but handy)

    def list_tools(self, name_filter: str = "") -> List[Dict[str, Any]]:
        """Return all tools whose *name* contains `name_filter`."""
        tools = self.gi_object.gi.tools.get_tools()
        if name_filter:
            tools = [t for t in tools if name_filter.lower() in t["name"].lower()]
        return tools
    
    def get_tool_io (self, tool_id):
        """Get tool i/o details"""
        tool= self.gi_object.gi.tools.show_tool(tool_id=tool_id, io_details=True)
        return tool['inputs']
    
    def generate_state(self, inputs):
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
        ) -> Dict[str, Any]:
        """
        Run a tool and return a structured result.

        """
        inputs = inputs or {}
        # Build the tool payload 
        # (validation step to convert name: value pairs from the io details into state input)

        tool_payload = self._build_payload(tool_id, history, inputs)
        pprint(tool_payload)
        # Run
        tool: Tool= self.gi_object.tools.get(tool_id, io_details= True)
        tool_execution= self.gi_object.gi.tools.run_tool(tool_id= tool.id, history_id=history.id, tool_inputs = tool_payload )
        pprint(tool_execution)
        # The job id
        job_id = tool_execution['jobs'][0]['id']
        outputs = tool_execution['outputs']

        output_datasets: list[Dataset] = []

        for output in outputs:
            dataset =  self.gi_object.datasets.get(output['id'])
            output_datasets.append(dataset)

        self.log.info(f"Started job {job_id} for tool {tool_id!r}")

        job=self.wait(job_id) # Wait for job to complete before making result
        return self._make_result(job, output_datasets)

    def wait(self, job_id):
        counter = 0 
        while True:
            job = self.gi_object.jobs.get(job_id)
            counter +=1
            if job.state in {'ok', 'error', 'cancelled'}:
                print(f'checked {counter} times and resulted {job.state}')
                break
            time.sleep(3) # set polling interval
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


# if __name__ == '__main__':
#     # Existing connection
#     gi = GalaxyClient().gi_object
#     history = gi.histories.get(id_="52e496b945151ee8")
#     dataset=  gi.datasets.get('2a4bf9d66c01414a')
#     tool_id="fastqc"

#     tm = ToolManager()
#     inputs={"input_file": {'id': dataset.id , 'src': 'hda'}}
#     # Upload first if you need input

#     # Run FastQC
#     result = tm.run(
#         tool_id=tool_id,
#         history=history,
#         inputs=inputs
#     )

#     pprint(result)

#     # inputs = gi.gi.tools.show_tool(tool_id, io_details=True)['inputs']
#     # pprint(inputs)
#     # io_state = tm.generate_state(inputs)
#     # pprint(io_state)
#     # # build = tm._build_payload(tool_id, history, inputs=io_state)
#     # # pprint(build)