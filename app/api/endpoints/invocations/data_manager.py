
import asyncio
import logging
from datetime import datetime
from typing import Dict

from sys import path
path.append(".")

from app.bioblend_server.galaxy import GalaxyClient
from bioblend.galaxy.objects.wrappers import Invocation
from app.bioblend_server.executor.workflow_manager import WorkflowManager
from app.orchestration.invocation_cache import InvocationCache
from app.orchestration.invocation_tasks import InvocationBackgroundTasks
from app.orchestration.utils import NumericLimits


from app.exceptions import InternalServerErrorException


class InvocationDataManager:
    """Handles data fetching, formatting, and structuring operations"""
    
    def __init__(self, cache: InvocationCache, background_tasks: InvocationBackgroundTasks):
        self.cache = cache
        self.log = logging.getLogger(__class__.__name__)
        self.background_tasks = background_tasks
    
    async def fetch_core_data(
        self,
        username: str,
        workflow_manager: WorkflowManager,
        workflow_id: str | None,
        history_id: str | None
    ) -> tuple[list, list]:
        """Fetch invocations and workflows with parallel processing"""
            
        # Prepare filter dictionary for caching
        filters = {"workflow_id": workflow_id, "history_id": history_id}
        
        async def fetch_invocations():
            """Fetch invocations with caching"""
            cached_invocations = await self.cache.get_invocations_cache(username, filters)
            if not cached_invocations:
                # Fetch all from Galaxy API based on filters
                if history_id and workflow_id:
                    invocations = await asyncio.to_thread(
                        workflow_manager.gi_object.gi.invocations.get_invocations,
                        workflow_id=workflow_id,
                        history_id=history_id,
                        limit = NumericLimits.INVOCATION_LIMIT.value
                    )
                elif history_id:
                    invocations = await asyncio.to_thread(
                        workflow_manager.gi_object.gi.invocations.get_invocations,
                        history_id=history_id,
                        limit = NumericLimits.INVOCATION_LIMIT.value
                    )
                elif workflow_id:
                    invocations = await asyncio.to_thread(
                        workflow_manager.gi_object.gi.invocations.get_invocations,
                        workflow_id=workflow_id,
                        limit = NumericLimits.INVOCATION_LIMIT.value
                    )
                else:
                    invocations = await asyncio.to_thread(
                        workflow_manager.gi_object.gi.invocations.get_invocations,
                        limit = NumericLimits.INVOCATION_LIMIT.value
                    )

                # Cache the results
                await self.cache.set_invocations_cache(username, invocations, filters)
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
            recent = await asyncio.to_thread(
                workflow_manager.gi_object.gi.invocations.get_invocations,
                **filter_params,
                limit = NumericLimits.INVOCATION_LIMIT.value
            )
            
            has_new_or_updated = False
            for inv in recent:
                inv_id = inv.get('id')
                inv_create = datetime.fromisoformat(inv['create_time']) if inv.get('create_time') else datetime.min
                if inv_id not in cached_ids or inv_create > max_create:
                    has_new_or_updated = True
                    break
            
            if not has_new_or_updated:
                self.log.info("No new invocations, using cache")
                return cached_invocations
            
            # Has new or updated, fetch all
            invocations = await asyncio.to_thread(
                workflow_manager.gi_object.gi.invocations.get_invocations,
                **filter_params
            )
            
            # Cache the results
            await self.cache.set_invocations_cache(username, invocations, filters)
            return invocations
        
        async def fetch_workflows():
            """Fetch workflows with caching"""
            cached_workflows = await self.cache.get_workflows_cache(username)
        
            if cached_workflows:
                self.log.info("Using cached workflows")
                return cached_workflows
            
            workflows = await self.background_tasks.fetch_workflows_safely(workflow_manager=workflow_manager, fetch_details = True)
            # Cache the results
            await self.cache.set_workflows_cache(username, workflows)
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
                    self.log.error(f"Failed to fetch {operation}: {result}")
            
            return invocations_data, workflows_data
            
        except Exception as e:
            self.log.error(f"Error in parallel data fetching: {e}")
            return [], []

    async def structure_inputs(
        self,
        inv: Dict,
        workflow_manager: WorkflowManager
    ) -> dict:
        """Structure invocation inputs"""
    
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
            
            self.log.info("Structuring invocation inputs for result.")
        except Exception as e:
            self.log.error
            raise InternalServerErrorException("structuring input structure failed")
        
        return inputs_formatted
    
    async def structure_outputs(
        self,
        _invocation: Invocation,
        outputs: Dict[str, list],
        workflow_manager: WorkflowManager,
        failure: bool = False
    ) -> tuple[list, str | None]:
        """Structure invocation result outputs, datasets, and collections"""

        # Prepare workflow invocation results
        invocation_report = workflow_manager.gi_object.gi.invocations.get_invocation_report(_invocation.id)
        
        collection_outputs = outputs.get( "collection_datasets")
        dataset_outputs = outputs.get("output_datasets")
        
        self.log.info(f"Structuring invocation outputs for result.")

        try:
            # Format the outputs (Pydantic schemas could be added here for validation)
            final_output_dataset = []
            final_collection_dataset = []
            semaphore = asyncio.Semaphore(NumericLimits.SEMAPHORE_LIMIT.value)
            
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
                            
                            # Create tasks for all element data_type fetches
                            element_data_type_tasks = [
                                asyncio.to_thread(
                                    workflow_manager.gi_object.gi.datasets.show_dataset,
                                    dataset_id=e.get("object", {}).get("id", "")
                                ) if e.get("object", {}).get("id", "") else None
                                for e in output.elements
                            ]
                            
                            async def get_empty_result():
                                return {'extension': 'unknown'}

                            # Gather all data_type results at once
                            element_data_types = await asyncio.gather(*[
                                task if task is not None else get_empty_result()
                                for task in element_data_type_tasks
                            ])
                            
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
                                        "data_type": element_data_types[i].get('extension', element_data_types[i].get('file_ext', 'unknown'))
                                    }
                                    for i, e in enumerate(output.elements)
                                ],
                                "is_intermediate": not output.visible
                            })
                            
                    except Exception as e:
                        self.log.error(f"Error when structuring outputs: {e}")

            # Create coroutines for formatting
            dataset_tasks = [structure_and_append(output, final_output_dataset, False) for output in dataset_outputs]
            collection_tasks = [structure_and_append(output, final_collection_dataset, True) for output in collection_outputs]
            
            # Run them concurrently
            await asyncio.gather(
                        *dataset_tasks,
                        *collection_tasks
                    )
            
            self.log.info(f"output datasets formatted: {len(final_output_dataset)}, collection datasets formatted: {len(final_collection_dataset)}")
            
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
            self.log.error(f"Error formatting outputs: {e}")
            raise

    
    async def report_invocation_failure(
        self,
        galaxy_client: GalaxyClient,
        invocation_id: str
    ) -> str:
        """Generate failure report for an invocation"""
    
        explanation = ""
        failed_job_descriptions = []
        
        # Get invocation jobs
        try:
            invocation_jobs = galaxy_client.gi_client.jobs.get_jobs(invocation_id=invocation_id)
        except Exception as e:
            self.log.error(f"Error retrieving invocation jobs: {str(e)}")
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
                self.log.info(f"Number of failed jobs in invocation {len(job_details_list)}")
            except Exception as e:
                self.log.error(f"Error retrieving job details concurrently: {str(e)}")
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
                    self.log.error(f"Error processing job details: {str(e)}")
                    explanation += f"Error processing failed job details: {str(e)}\n"
        
        if failed_job_descriptions:
            explanation += "\n\nFailed Jobs Detected:\n" + "\n".join(failed_job_descriptions) + "\n"
        
        else:
            explanation += "\nNo failed jobs detected in this invocation.\n"
            
        return explanation
