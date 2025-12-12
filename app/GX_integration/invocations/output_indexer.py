import asyncio
import logging

from app.galaxy import GalaxyClient
from app.api.socket_manager import SocketManager
from app.GX_integration.tool_manager import ToolManager
from app.orchestration.invocation_cache import InvocationCache

class OutputIndexer:
    """support dataset indexing for galaxy workflow invocation output datasets for visualization purposes."""
    
    def __init__(self, username: str, galaxy_client: GalaxyClient, cache: InvocationCache, ws_manager: SocketManager):

        self.gi = galaxy_client.gi_client
        self.username = username
        self.cache = cache
        self.ws_manager = ws_manager
        self.tool_manager = ToolManager(galaxy_client = galaxy_client)
        
        self.log = logging.getLogger(__class__.__name__)
        
        # Datasets indexing counter variables.
        self.index_count = 0
        self.total_index = 0      
    
    async def _structure_indexed_data(self, history_id: str, dataset_name: str, index_outputs: list[dict]):
        """structure indexed result"""
        
        self.log.debug(f"Structuring index data for history_id={history_id}, "
                   f"target_name={dataset_name}, outputs={len(index_outputs)}")
        structured_index = []
        
        for index in index_outputs:
            # Update to approapriate name for index file.
            try:
                self.log.debug(f"Renaming index dataset id={index.get('id')} -> {dataset_name}")
                await asyncio.to_thread(
                    self.gi.histories.update_dataset,
                    history_id=history_id,
                    dataset_id=index.get("id"),
                    name=dataset_name
                )
            except Exception as e:
                self.log.error(f"Failed to rename index dataset id={index.get('id')}: {e}")
                raise
            
            # even though it is most likely a single file, structure it into a list so as to not break code.
            structured_index.append(
            {
                "type": "dataset",
                "id": index.get("id"),
                "name": index.get("name"),
                "visible": index.get("visible"),
                "file_path": index.get('file_name'),
                "peek": index.get('peek'),
                "data_type": index.get('extension', index.get('file_ext', 'unknown')),
                "is_intermediate": not index.get("visible")
                }
            )
            
        self.log.info(f"Index structuring complete for {dataset_name} ({len(structured_index)} files).")
        return structured_index
       
       
    async def index_fasta(self, history_id: str, dataset: tuple[str, str]):
        """ Index fasta files to generate a fai index.

        Args:
            dataset (tuple): fasta dataset name and id.
        """
        name, dataset_id = dataset
        tool_id = "CONVERTER_fasta_to_fai"
        tool_input = {"input" : {"src": "hda", "id": dataset_id}}
        
        self.log.debug(f"Starting FASTA indexing: history={history_id}, dataset={dataset_id}, tool={tool_id}")
        try:
            result = await self.tool_manager.run(
                tool_id= tool_id,
                history_id = history_id,
                inputs = tool_input
                )
        except Exception as e:
            self.log.error(f"FASTA indexing failed for dataset id={dataset_id}: {e}")
            return[]
        
        self.log.debug(f"FASTA indexing tool completed: produced={len(result['dataset'])} output(s)")
        
        self.index_count += 1
        self.log.info(f"fasta indexing complete for dataset id: {dataset_id} - {self.index_count}/{self.total_index}")
        
        return await self._structure_indexed_data(
            history_id = history_id,
            dataset_name = f"{name}_fai_index", 
            index_outputs = result["dataset"]
            )
    
    async def index_vcf(self, history_id: str, dataset: tuple[str, str]):
        """ Index vcf files to generate a tabix index.

        Args:
            dataset (tuple): vcf dataset name and id.
        """
        name, dataset_id = dataset

        # Execute 2 tools consecutively as kind of like workflows.
        tool_id1 = "CONVERTER_uncompressed_to_gz"
        tool_id2 = "CONVERTER_vcf_bgzip_to_tabix_0"
        self.log.debug(f"Starting VCF compression: history={history_id}, dataset={dataset_id}, tool={tool_id1}")
        tool1_input = {"input1": {"src": "hda", "id": dataset_id}}
        
        try:   
            result_1 = await self.tool_manager.run(
                tool_id = tool_id1, 
                history_id = history_id, 
                inputs = tool1_input
                )
        except Exception as e:
            self.log.error(f"VCF indexing failed for dataset id={dataset_id}: {e}")
            return []
        
        self.log.debug(f"VCF compressed: new_dataset_id={result_1_id}")
        
        # extract dataset id of the compressed dataset.
        result_1_id = result_1.get("dataset")[0].get("id")
        tool2_input = {"input1": {"src": "hda", "id" : result_1_id }}
        
        self.log.debug(f"Starting tabix indexing: tool={tool_id2}, input={result_1_id}")
        try:
            result = await self.tool_manager.run(
                tool_id = tool_id2,
                history_id = history_id,
                inputs = tool2_input
            )
        except Exception as e:
            self.log.error(f"VCF indexing failed for dataset id={dataset_id}: {e}")
            return []
        
        self.index_count += 1
        self.log.info(f"vcf indexing complete for dataset id: {dataset_id} - {self.index_count}/{self.total_index}")
        return await self._structure_indexed_data(
            history_id = history_id,
            dataset_name = f"{name}_tabix_index", 
            index_outputs = result["dataset"]
            )
    
    
    async def index_bam(self, history_id: str, dataset: tuple[str, str]):
        """ Index vcf files to return a bai index.

        Args:
            dataset (tuple): bam dataset name and id.
        """
        name, dataset_id = dataset

        tool_id = "CONVERTER_Bam_Bai_0"
        tool_input = {"input1" : {"src": "hda", "id": dataset_id}}
        self.log.debug(f"Starting BAM indexing: history={history_id}, dataset={dataset_id}, tool={tool_id}")
        try:
            result = await self.tool_manager.run(
                tool_id= tool_id,
                history_id = history_id,
                inputs = tool_input
                )
        except Exception as e:
            self.log.error(f"BAM indexing failed for dataset id={dataset_id}: {e}")
            return []       
         
        self.index_count += 1
        self.log.info(f"bam indexing complete for dataset id: {dataset_id} - {self.index_count}/{self.total_index}")
        return await self._structure_indexed_data(
            history_id = history_id,
            dataset_name = f"{name}_bai_index", 
            index_outputs = result["dataset"]
            )
    
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
            
            self.log.debug(f"Collected datasets for indexing: {index_datasets}")
            
            # start of with fasta indexing: NOTE: Add indexer function and task as necessary here.
            indexing_task = [self.index_fasta(history_id, dataset) for dataset in index_datasets.get("fasta", [])]
            indexing_task.extend([self.index_vcf(history_id, dataset) for dataset in index_datasets.get("vcf", [])])
            indexing_task.extend([self.index_bam(history_id, dataset) for dataset in index_datasets.get("bam", [])])

            # count total indxes that are to be used.
            self.total_index = len(indexing_task)

            if self.total_index > 0:
                
                self.log.info(f"Total datasets to index: {self.total_index}")
                self.log.info(f"Beginning dataset indexing for invocation_id={invocation_id} in history_id={history_id}")
                # gather index results;
                index_results = await asyncio.gather(*indexing_task)
                flat_results = [item for sublist in index_results for item in sublist]
                invocation_outputs.extend(flat_results)
                
                invocation_result["result"] = invocation_outputs
                await self.cache.set_invocation_result(
                    username = self.username,
                    invocation_id = invocation_id,
                    result = invocation_result
                    )