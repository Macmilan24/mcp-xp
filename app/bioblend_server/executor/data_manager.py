# data_manager.py
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Union, Optional


from bioblend.galaxy.objects.wrappers import  History, Dataset, DatasetCollection
from app.bioblend_server.galaxy import GalaxyClient


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

    def __init__(self):
        self.gi = GalaxyClient().gi_object 
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
                    element_mappings: List[Dict[str, Any]],
                    wait: bool = True,
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

        # Maybe use LLMs here to congigure the element identifiers configurations
        payload = {
            "collection_type": collection_type.value,
            "element_identifiers": element_mappings,
            "name": f"{collection_type.value}_{int(time.time())}",
        }
        coll = history.create_dataset_collection(payload, wait=wait)
        return UploadResult(coll, coll.id, coll.name)

    
    # Output handling/ downloading datasets/collections
    def download_outputs(
        self,
        outputs: List[Union[Dataset, DatasetCollection]],
        dest_dir: Union[str, os.PathLike],
        overwrite: bool = False,
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
                target = dest_dir / f'{item_details['name']}.{item_details['extension']}'
                if target.exists() and not overwrite:
                    raise FileExistsError(target)
                with target.open("wb") as fh:
                    item.download(fh)
                downloaded.append(target)

            elif isinstance(item, DatasetCollection):

                archive = dest_dir / f"{item.name}.tar.gz"
                if archive.exists() and not overwrite:
                    raise FileExistsError(archive)
                # Use the procedural style bioblend to download dataset collection
                self.gi.gi.dataset_collections.download_dataset_collection(dataset_collection_id=item.id, file_path=archive )
                downloaded.append(archive)

            else:
                self.log.warning(f"Skipping unknown output type: {type(item)}")

        return downloaded

    # HISTORY INTROSPECTION

    def list_contents(
        self, history: History, include_deleted: bool = False
    ) -> List[Union[Dataset, DatasetCollection]]:
        """Return every top-level item in the history."""
        items = history.content_infos(include_deleted=include_deleted)
        # Map info objects to actual wrappers
        datasets = [self.gi.datasets.get(item["id"]) for item in items if item["history_content_type"] == "dataset"]
        collections = [
            self.gi.dataset_collections.get(item["id"])
            for item in items
            if item["history_content_type"] == "dataset_collection"
        ]
        return datasets + collections


## TESTING
# if __name__== "__main__" : 
#     dm = DataManager()
#     hist = dm.gi.histories.create(name="DataManager-Demo")

#     # 1. Upload two FASTQ files
#     fwd = dm.upload_file(hist, "/data/sample_R1.fastq.gz")
#     rev = dm.upload_file(hist, "/data/sample_R2.fastq.gz")

#     # 2. Build a paired collection
#     paired = dm.upload_collection(
#         history=hist,
#         collection_type=CollectionType.PAIRED,
#         element_mappings=[
#             {"name": "forward", "src": "hda", "id": fwd.id},
#             {"name": "reverse", "src": "hda", "id": rev.id},
#         ],
#     )

#     # 3. Later, download everything produced by a tool or workflow
#     outputs = dm.list_contents(hist)
#     local_paths = dm.download_outputs(outputs, dest_dir="downloads")
#     print(local_paths)