# data_manager.py
from __future__ import annotations

import logging
import os
import time
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Union, Tuple
from pandas import DataFrame
from rapidfuzz import process, fuzz


from bioblend.galaxy.objects.wrappers import  History, Dataset, DatasetCollection
from app.galaxy import GalaxyClient


class CollectionType(str, Enum):
    """Enumeration of Galaxy collection kinds that this manager supports."""
    LIST = "list"
    PAIRED = "paired"
    LIST_PAIRED = "list:paired"


@dataclass
class UploadResult:
    """Tiny DTO returned by every upload function."""
    wrapper: Union[Dataset, DatasetCollection]
    id: str
    name: str


class DataManager:
    """
    Pure data I/O layer for Galaxy:
    * upload single files / URLs
    * build & upload collections
    * download any number of outputs
    * list / inspect history contents
    """

    def __init__(self, galaxy_client: GalaxyClient):
        self.galaxy_client  = galaxy_client
        self.gi = self.galaxy_client.gi_object

        self.log = logging.getLogger(self.__class__.__name__)

    # SINGLE DATASET UPLOAD
    def upload_file(
                self,
                history: History,
                path: Union[str, os.PathLike],
                file_type: str = "auto",
                dbkey: str = "?",
                wait: bool = True,
            ) -> UploadResult:
        
        """Upload a local file."""
        path = Path(path)
        ds = history.upload_file(
            str(path),
            file_type=file_type,
            dbkey=dbkey,
            wait=wait,
        )
        ds.wait() # wait for dataset to finish uploading to the history
        return UploadResult(ds, ds.id, ds.name)
    
    
     # COLLECTION UPLOAD

    def upload_collection(
                    self,
                    history: History,
                    collection_type: CollectionType,
                    inputs: List[str], 
                    collection_name = None
                ) -> UploadResult:
        """
        Build and upload a collection.

        Parameters
        ----------
        element_mappings
            * LIST:  [{'name': 'sample1', 'src': 'hda', 'id': <dataset-id>}, ...]
            * PAIRED: [{'name': 'forward', 'src': 'hda', 'id': <id>},
                       {'name': 'reverse', 'src': 'hda', 'id': <id>}]
            * LIST_PAIRED: each element must be another dict with nested
              'forward' / 'reverse' in the same format.
        """
        try:
            if collection_name is None:
                collection_name= f"{collection_type.value}_{int(time.time())}"
            # Maybe use LLMs here to congigure the element identifiers configurations
            element_mappings=[]
            # check what type of collection type it is
            # for list collection
            if collection_type.value == collection_type.LIST:
                for _input in inputs:
                    dataset_new=self.upload_file(history = history, path = _input)
                    element_mappings.append({'name': dataset_new.name, 'src': 'hda', 'id': dataset_new.id})
            
            # for paired collection
            elif collection_type.value == collection_type.PAIRED:
                if len(inputs) == 2:
                    input_1 = self.upload_file(history=history, path=inputs[0])
                    input_2 = self.upload_file(history=history, path=inputs[1])
                    element_mappings=[
                        {'name': 'forward', 'src': 'hda', 'id': input_1.id},
                        {'name': 'reverse', 'src': 'hda', 'id': input_2.id}
                        ]
                else:
                    self.log.error('Invalid input !!')
                    return 
            # for list:paired collections
            elif collection_type.value == collection_type.LIST_PAIRED:
                if isinstance(inputs, list) and all(isinstance(i, list) for i in inputs):
                    for _input in inputs:
                        input_1 = self.upload_file(history=history, path=_input[0])
                        input_2 = self.upload_file(history=history, path=_input[1])
                        element_mappings.append([
                            {'name': 'forward', 'src': 'hda', 'id': input_1.id},
                            {'name': 'reverse', 'src': 'hda', 'id': input_2.id}
                        ]) 
                else:
                    self.log.error("invalid inputs !!")
                    return
                
            payload = {
                "collection_type": collection_type.value,
                "element_identifiers": element_mappings,
                "name": collection_name,
            }
            coll = self.gi.gi.histories.create_dataset_collection(history_id=history.id, collection_description=payload)
            coll_obj = self.gi.dataset_collections.get(coll['id'])
            return UploadResult(coll_obj, coll_obj.id, coll_obj.name)
        except Exception as e:
             raise Exception(f"Error during creating collection: {e}")

    
    # Output handling/ downloading datasets/collections
    def download_outputs(
        self,
        outputs: List[Union[Dataset, DatasetCollection]],
        dest_dir: Union[str, os.PathLike],
        overwrite: bool = True,
    ) -> List[Path]:
        """
        Download every dataset / collection to `dest_dir`.

        Collections are saved as .tar.gz archives named after the collection.
        Single datasets keep their original names.
        Returns list of local paths.
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        downloaded: List[Path] = []

        for item in outputs:
            if isinstance(item, Dataset):
                # to get the extenstion of the dataset to be downloaded
                item_details= self.gi.gi.datasets.show_dataset(dataset_id=item.id)
                name = re.sub(r'[\\/*?:"<>|]', '', item_details['name']).replace(' ', '_')
                ext = re.sub(r'[\\/*?:"<>|]', '', item_details['extension']).replace(' ', '_')

                item_name = f"{name}.{ext}"
                target = dest_dir / item_name
                if target.exists() and not overwrite:
                    raise FileExistsError(target)
                
                # Prefer display/download URL from Galaxy
                download_url = item_details.get('download_url', None)
                if download_url:
                    full_url = f"{self.gi.gi.base_url}{download_url}"
                    response = self.gi.gi.make_get_request(url = full_url, stream =True )
                    with target.open("wb") as fh:
                        for chunk in response.iter_content(chunk_size=8192):
                            fh.write(chunk)
                else:
                    # Fallback to direct download
                    with target.open("wb") as fh:
                        item.download(fh)

            elif isinstance(item, DatasetCollection):
                name = re.sub(r'[\\/*?:"<>|]', '', item.name).replace(' ', '_')
                archive = dest_dir / f"{name}.tar.gz"
                if archive.exists() and not overwrite:
                    raise FileExistsError(archive)
                
                # Use the procedural style bioblend to download dataset collection since downloading collections is unavailable on the OOP bioblend
                self.gi.gi.dataset_collections.download_dataset_collection(dataset_collection_id=item.id, file_path=archive )
                downloaded.append(archive)

            else:
                self.log.warning(f"Skipping unknown output type: {type(item)}")

        return downloaded

    def list_contents(
        self, history: History, include_deleted: bool = False
    ) -> Tuple[List, List]:
        """Return every top-level item in the history."""
        items=self.gi.gi.datasets.get_datasets(history_id=history.id)
        # Map info objects to actual wrappers
        datasets = [
            item
            for item in items 
            if item['history_content_type'] == "dataset" and (include_deleted or not item['deleted'])
        ]
        
        collections = [
            item
            for item in items 
            if item['history_content_type'] == "dataset_collection" and (include_deleted or not item['deleted'])
        ]

        return datasets, collections
    
    def list_data_tables(self):
        """List of data tables names of reference data in the galaxy instance"""
        data_tables = self.gi.gi.tool_data.get_data_tables()
        return [data['name'] for data in data_tables]
    
    def get_data_table_elements(self, data_table_name: str) -> DataFrame:
        """Get a list of elements of a data table"""
        data_table_list = self.list_data_tables()
        # Finding the best match for the data table in the galaxy instance
        find_match = process.extractOne(
            query= data_table_name,
            choices= data_table_list,
            scorer= fuzz.partial_ratio
        )[0]
        data_table = self.gi.gi.tool_data.show_data_table(find_match) 
        
        return DataFrame(data_table["fields"], columns = data_table["columns"])