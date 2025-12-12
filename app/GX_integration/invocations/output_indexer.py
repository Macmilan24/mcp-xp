import asyncio
import logging

from app.galaxy import GalaxyClient
from app.api.socket_manager import SocketManager
from app.GX_integration.tool_manager import ToolManager
from app.orchestration.invocation_cache import InvocationCache

class OutputIndexer:
    """support dataset indexing for galaxy workflow invocation output datasets for visualization purposes."""
    
    def __init__(self, username: str, galaxy_client: GalaxyClient, cache: InvocationCache, ws_manager: SocketManager):

        self.username = username
        self.cache = cache
        self.ws_manager = ws_manager
        self.tool_manager = ToolManager(galaxy_client = galaxy_client)
        
        self.log = logging.getLogger(__class__.__name__)
        
        # Datasets indexing counter variables.
        self.index_count = 0
        self.total_index = 0
        
    def _structure_tool_input(self, history_id, dataset_id):
        """structure tool input for execution"""
        pass
        
    
    async def index_fasta(self, history_id: str, dataset: tuple[str, str]):
        """ Index fasta files to generate a fai index.

        Args:
            dataset (tuple): fasta dataset name and id.
        """
        name, dataset_id = dataset

        # TODO: Implement galaxy tool execution here
        self.index_count += 1
        self.log.info(f"fasta indexing complete for dataset id: {dataset_id} - {self.index_count}/{self.total_index}")
        pass
    
    async def index_vcf(self, history_id: str, dataset: tuple[str, str]):
        """ Index vcf files to generate a tabix index.

        Args:
            dataset_id (tuple): vcf dataset name and id.
        """
        name, dataset_id = dataset

        # TODO: Implement galaxy tool execution here
        self.index_count += 1
        self.log.info(f"vcf indexing complete for dataset id: {dataset_id} - {self.index_count}/{self.total_index}")
        pass
    
    async def index_bam(self, history_id: str, dataset: tuple[str, str]):
        """ Index vcf files to return a bai index.

        Args:
            dataset_id (tuple): bam dataset name and id.
        """
        name, dataset_id = dataset

        # TODO: Implement galaxy tool execution here
        self.index_count += 1
        self.log.info(f"bam indexing complete for dataset id: {dataset_id} - {self.index_count}/{self.total_index}")
        pass
    
    #TODO: Add more dataset indexing functions that JBrowse supports.
    
    async def index_datasets(self, invocation_result: dict):
        
        # create dict to collect datasets to be indexed. NOTE: Add as needed, for now we support fasta, vcf and bam files. 
        index_datasets = {
                "fasta" : [],
                "vcf": [],
                "bam": []
                }
        
        # extract invocation and history id of the invocation.
        invocation_id = invocation_result.get("invocation_id", None)
        history_id = invocation_result.get("history_id")
        
        if invocation_id:
           
            invocation_outputs: list[dict]= invocation_result.get("result", [])
            
            # collect each dataset within the invocation output to sort out which indexing the datasets need.
            for output in invocation_outputs:
                output_type = output.get("type")
                
                if output_type == "collection":
                    collection_elements: list[dict] = output.get("elements", [])
                    for element in collection_elements:
                        element_type = element.get("data_type", "unknown")
                        element_id = element.get("id")
                        element_name = element.get("name")
                        if element_type in index_datasets.keys() and element_id and element_name:
                            index_datasets[element_type].append((element_name, element_id))
                            
                elif output_type == "dataset":
                    element_type = output.get("data_type")
                    element_id = output.get("id")
                    element_name = output.get("name")
                    if element_type in index_datasets.keys() and element_id and element_name:
                            index_datasets[element_type].append((element_name, element_id))
            
            # start of with fasta indexing: NOTE: Add indexer function and task as necessary here.
            indexing_task = [self.index_fasta(history_id, dataset) for dataset in index_datasets.get("fasta", [])]
            indexing_task.extend([self.index_vcf(history_id, dataset) for dataset in index_datasets.get("vcf", [])])
            indexing_task.extend([self.index_bam(history_id, dataset) for dataset in index_datasets.get("bam", [])])

            # count total indxes that are to be used.
            self.total_index = len(indexing_task)
            
            # gather index results;
            index_results = await asyncio.gather(*indexing_task)
            invocation_outputs.extend(index_results)
            
            invocation_result["result"] = invocation_outputs
            await self.cache.set_invocation_result(
                username = self.username,
                invocation_id = invocation_id,
                result = invocation_result
                )
        
        
            
        
        
    