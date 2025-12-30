import redis
import asyncio
from typing import List
import logging

from sys import path
path.append(".")

from fastapi.concurrency import run_in_threadpool
from bioblend.galaxy.objects.wrappers import HistoryDatasetAssociation, HistoryDatasetCollectionAssociation

from app.galaxy import GalaxyClient
from app.GX_integration.workflows.workflow_manager import WorkflowManager
from app.GX_integration.invocations.output_indexer import OutputIndexer
from app.api.schemas import invocation
from app.api.socket_manager import SocketManager
from app.api.enums import SocketMessageEvent, SocketMessageType
from app.orchestration.invocation_cache import InvocationCache
from app.enumerations import NumericLimits, JobState

from app.GX_integration.invocations.data_manager import InvocationDataManager

# invocation_tracker.py
class InvocationTracker:
    """Handles state tracking, background monitoring, and cleanup operations"""
    
    def __init__(self, cache: InvocationCache, redis_client: redis.Redis, inv_data_manager:InvocationDataManager):
        self.cache = cache
        self.redis_client = redis_client
        self.inv_data_manager = inv_data_manager
        self.log = logging.getLogger(__class__.__name__)
        

    async def format_invocations(
        self,
        username: str,
        invocations_data: list[dict],
        workflow_mapping: dict[str, dict],
        workflow_manager: WorkflowManager
    ) -> list[invocation.InvocationListItem]:
        """Format invocations with optimized state mapping"""
    
        self.log.info(f"length og mappings: {len(workflow_mapping)}  and invocations: {len(invocations_data)}")
        length = 0
        filtered_invocations = []
        for inv in invocations_data:
            inv_id = inv.get('id')
            
            # Get workflow info from mapping
            workflow_info = workflow_mapping.get(inv_id, {})
            workflow_name = workflow_info.get('workflow_name', None)
            workflow_id = workflow_info.get("workflow_id", None)
            
            # Skip if no workflow name : Likely due to the invocation being a subworkflow invocation or workflow being deleted.
            # NOTE: Maybe in the future if invocation are still needed even after the workflow has been deleted, we can refactor implemenation.
            if not workflow_name:
                length += 1
                continue
            
            filtered_invocations.append((inv, workflow_name, workflow_id))

        # Gather all state mappings concurrently
        state_tasks = [
            self.map_invocation_state(
                username=username,
                raw_state=inv.get('state'),
                invocation_id=inv.get("id"),
                workflow_manager=workflow_manager
            )
            for inv, _, _ in filtered_invocations
        ]
        states = await asyncio.gather(*state_tasks)

        # Build the final list
        invocation_list = [
            invocation.InvocationListItem(
                id=inv.get('id'),
                workflow_name=workflow_name,
                workflow_id=workflow_id,
                history_id=inv.get('history_id'),
                state=state,
                create_time=inv.get('create_time'),
                update_time=inv.get('update_time')
            )
            for (inv, workflow_name, workflow_id), state in zip(filtered_invocations, states)
        ]
                
        self.log.debug(f"skipped {length} invocation becasue workflowname was missing.")
        
        return invocation_list

    
    async def map_invocation_state(
        self,
        username: str,
        raw_state: str,
        invocation_id: str,
        workflow_manager: WorkflowManager
    ) -> str:
        """Map Galaxy invocation states to user-friendly states"""
        
        # Get invocation state if saved for accurate state.
        cached_state = await self.cache.get_invocation_state(username, invocation_id)
        if cached_state:
            self.log.debug(f"Getting cached invocation state: {cached_state}")
            return cached_state

        if not raw_state:
            self.log.warning("Invocation state is None or empty")
            return "Failed"
        
        # Define state mappings
        failed_states = {"cancelled", "failed", "cancelling"}
        pending_states = {"requires_materialization", "ready", "new", "scheduled"}
        
        if raw_state in failed_states:
            return "Failed"

        elif raw_state in pending_states:
            return await self.compute_deep_invocation_state(
                workflow_manager = workflow_manager,
                invocation_id = invocation_id,
                username = username
            )

        else:
            self.log.warning(f"Unknown invocation state: {raw_state}")
            return "Failed"
        
    async def compute_deep_invocation_state(
        self,
        workflow_manager: WorkflowManager,
        invocation_id: str,
        username: str
    ) -> str:
        """Compute detailed invocation state from step jobs"""
        try:
            # Fetch step jobs summary
            step_jobs = await asyncio.to_thread(
                workflow_manager.gi_object.gi.invocations.get_invocation_step_jobs_summary,
                invocation_id=invocation_id
            )

            all_ok = True
            has_failed = False

            for step in step_jobs:
                states = step.get('states', {})

                ok_count = states.get(JobState.OK.value, 0)
                skipped_count = states.get(JobState.SKIPPED.value, 0)
                failed_count = states.get(JobState.FAILED.value, 0)
                error_count = states.get(JobState.ERROR.value, 0)
                running_count = states.get(JobState.RUNNING.value, 0)

                total_jobs = sum(states.values())
                completed_jobs = ok_count + skipped_count
                failed_jobs = failed_count + error_count
                all_jobs_failed = (failed_jobs == total_jobs and total_jobs > 0)

                # Set flags
                if all_jobs_failed:
                    has_failed = True

                # Determine step state (similar to _analyze_step_jobs)
                if all_jobs_failed:
                    step_state = JobState.ERROR.value
                elif completed_jobs == total_jobs:
                    step_state = JobState.OK.value
                elif running_count > 0:
                    step_state = JobState.RUNNING.value
                elif failed_jobs > 0 and failed_jobs < total_jobs:
                    step_state = JobState.RUNNING.value
                else:
                    step_state = 'Pending'

                if step_state != JobState.OK.value:
                    all_ok = False

            if has_failed:
                inv_state = "Failed"
            elif all_ok:
                inv_state = "Complete"
            else:
                inv_state = "Pending"
            
            await self.cache.set_invocation_state(username = username,
                                                        invocation_id = invocation_id,
                                                        state = inv_state
                                                        )
            return inv_state
            
        except Exception as e:
            self.log.error(f"Error computing deep state for {invocation_id}: {e}")

    
    async def background_track_and_cache(
        self,
        invocation_id: str,
        galaxy_client: GalaxyClient,
        workflow_manager: WorkflowManager,
        history_id: str,
        create_time: str,
        last_update_time: str,
        inputs_formatted: dict,
        workflow_description,
        ws_manager: SocketManager,
        tracking_key: str
    ):
        """Background task to track invocation and cache results"""
        username = galaxy_client.whoami
        output_indexer = OutputIndexer(username = username, galaxy_client = galaxy_client, cache = self.cache, ws_manager = ws_manager)
        
        try:
    
            _invocation = await run_in_threadpool(
                galaxy_client.gi_object.invocations.get,
                id_=invocation_id
            )
            if not _invocation:
                self.log.error(f"Invocation {invocation_id} not found in background")
                return

            outputs, inv_state, update_time = await workflow_manager.track_invocation(
                invocation=_invocation,
                tracker_id=invocation_id,
                ws_manager=ws_manager,
                invocation_check=True
            )
            
            # Set invocation to cache
            await self.cache.set_invocation_state(username, invocation_id, inv_state)

            if inv_state == "Failed":
                
                invocation_report = await self.inv_data_manager.report_invocation_failure(galaxy_client, invocation_id)
                workflow_result, _ = await self.inv_data_manager.structure_outputs(
                _invocation=_invocation, outputs=outputs, workflow_manager=workflow_manager, failure = True
                )
            else:
                workflow_result, invocation_report = await self.inv_data_manager.structure_outputs(
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
            await self.cache.set_invocation_result(username, invocation_id, result_dict)
            
            self.log.info("Invocation results are complete and ready.")
            if ws_manager:
                ws_data = {
                    "type": SocketMessageType.INVOCATION_COMPLETE.value,
                    "payload": {"message": "Invocation results are complete and ready."}
                }
                await ws_manager.broadcast(
                    event=SocketMessageEvent.workflow_execute.value,
                    data=ws_data,
                    tracker_id=invocation_id
                )
            
            # Index invocation outputs
            await output_indexer.index_datasets(invocation_result= result_dict)


        except Exception as e:
            self.log.error(f"Error in background tracking for {invocation_id}: {e}")
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
            await self.cache.set_invocation_result(username, invocation_id, result_dict)

        finally:
            await run_in_threadpool(self.redis_client.delete, tracking_key)

    
    async def cancel_invocation_and_delete_data(
        self,
        invocation_ids: List[str],
        workflow_manager: WorkflowManager,
        username: str
    ):
        """Cancel invocations and delete associated data"""
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
                    await asyncio.sleep(NumericLimits.SHORT_SLEEP.value)

                # TODO: Check cache first before doing a full track for output id extraction 
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
                await asyncio.gather(*[self.cache.delete_invocation_state(username, inv_id) for inv_id in invocation_ids])
                
            
            self.log.info(f"Invocation {invocation_ids} deletion complete.")
        except Exception as e:
                self.log.error(f"Error while deleting invocation: {e}")