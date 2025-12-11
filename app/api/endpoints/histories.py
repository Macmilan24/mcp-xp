from sys import path
path.append('.')

import tempfile
import os
from typing import List
import re
import json
import functools
from anyio import to_thread

from fastapi import APIRouter, UploadFile, responses, HTTPException, File, Form, Path, Body, Query, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
import pathlib
import shutil

from app.context import current_api_key
from app.galaxy import GalaxyClient
from app.GX_integration.data_manager import DataManager, CollectionType
from app.GX_integration.history_manager import HistoryManager
from app.api.schemas import dataset, history

from app.exceptions import InternalServerErrorException, BadRequestException

router = APIRouter()

# @router.get(
#     "/",
#     response_model = List[history.HistoryResponse],
#     summary="List all Galaxy histories",
#     tags=["Histories & Data"]
# )
async def list_histories():
    """
    Retrieve a list of all Galaxy histories for the current API key.
    """
    galaxy_client = GalaxyClient(current_api_key.get())
    history_manager = HistoryManager(galaxy_client)

    try:
        histories = await run_in_threadpool(history_manager.list_history)
        return [
                history.HistoryResponse(
                    id= h.id,
                    name= h.name,
                    )
            for h in histories
        ]
    except Exception as e:
        raise InternalServerErrorException("Failed to list histories")


# @router.post(
#     "/create",
#     response_model = history.HistoryResponse,
#     summary="Create a new Galaxy history",
#     tags=["Histories & Data"]
# )
async def create_history(
    name: str | None = Query(None, description="Name of the new history")
):
    """
    Create a new Galaxy history.
    """
    galaxy_client = GalaxyClient(current_api_key.get())
    history_manager = HistoryManager(galaxy_client)

    try:
        new_history = await run_in_threadpool(
            history_manager.create, name
        )
        return history.HistoryResponse(
            id= new_history.id,
            name = new_history.name
        )
    except Exception as e:
        raise InternalServerErrorException("Failed to create history")

# @router.post(
#     "/{history_id}/upload-file",
#     response_model=dataset.FileUploadResponse,
#     summary="Upload a Single File",
#     tags=["Histories & Data"]
# )
async def upload_file_to_history(
    history_id: str = Path(..., description="The ID of the Galaxy history."),
    file: UploadFile = File(..., description="The file to upload.")
):
    """
    Uploads a single file to the specified Galaxy history.
    The file is saved temporarily on the server before being sent to Galaxy.
    """

    galaxy_client = GalaxyClient(current_api_key.get())
    data_manager = DataManager(galaxy_client)

    try:
        # Use a temporary file to handle the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Get the history object
        galaxy_history = await run_in_threadpool(data_manager.gi.histories.get, history_id)

        # Run the synchronous upload in a thread pool
        result = await run_in_threadpool(
            data_manager.upload_file,
            history=galaxy_history,
            path=tmp_path
        )
        os.remove(tmp_path) # Clean up the temporary file
        return {"id": result.id, "name": result.name}
    except Exception as e:
        # Clean up in case of error
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise InternalServerErrorException("An error occurred")


# @router.post(
#     "/{history_id}/upload-collection",
#     response_model=dataset.CollectionResponse,
#     summary="Create a Dataset Collection from new files",
#     tags=["Histories & Data"]
# )
async def upload_dataset_collection(
    history_id: str = Path(..., description="The ID of the Galaxy history."),
    files: List[UploadFile] = File(..., description="The file(s) to upload for the collection."),
    collection_type: CollectionType = Form(..., description="Type of collection to create."),
    collection_name: str | None = Form(None, description="(Optional) custom name for the new collection."),
    structure: str | None = Form(
        None,
        description="A JSON string describing the structure for a 'list:paired' collection. "
                    "Example: '[[\"sample1_R1.fq\", \"sample1_R2.fq\"], [\"sample2_R1.fq\", \"sample2_R2.fq\"]]'.",
    ),
):
    """
    Uploads files and creates a Galaxy dataset collection.

    - For **list** collections, simply upload all files.
    - For **paired** collections, upload exactly two files.
    - For **list:paired** collections, you must provide the `structure` form field
      to define the pairs.

    Temporary files are cleaned up automatically after the request finishes.
    """

    galaxy_client = GalaxyClient(current_api_key.get())
    data_manager = DataManager(galaxy_client)

    tmpdir = tempfile.mkdtemp(prefix="galaxy_coll_")
    try:
        # Save all uploaded files to a temp directory and map filename to its path
        file_path_map = {}
        for uf in files:
            # Basic security: prevent path traversal attacks
            if ".." in uf.filename or "/" in uf.filename:
                raise BadRequestException("Invalid filename: {uf.filename}")
            
            tmp_path = pathlib.Path(tmpdir) / uf.filename
            with tmp_path.open("wb") as fout:
                # Use a thread for the blocking file write operation
                await to_thread.run_sync(fout.write, await uf.read())
            file_path_map[uf.filename] = str(tmp_path)

        # Prepare the `inputs` list based on the collection type
        inputs = []
        if collection_type == CollectionType.LIST:
            inputs = list(file_path_map.values())
        
        elif collection_type == CollectionType.PAIRED:
            if len(files) != 2:
                raise BadRequestException("A 'paired' collection requires exactly two files.")
            inputs = list(file_path_map.values())

        elif collection_type == CollectionType.LIST_PAIRED:
            if not structure:
                raise BadRequestException("The 'structure' field is required for 'list:paired' collections.")
            try:
                paired_filenames = json.loads(structure)
                for fwd_name, rev_name in paired_filenames:
                    fwd_path = file_path_map.get(fwd_name)
                    rev_path = file_path_map.get(rev_name)
                    if not fwd_path or not rev_path:
                        raise BadRequestException("A file in the provided structure was not uploaded: {fwd_name} or {rev_name}")
                    inputs.append([fwd_path, rev_path])
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                raise BadRequestException("Invalid 'structure' format")

        # Get the history object
        galaxy_history = await to_thread.run_sync(data_manager.gi.histories.get, history_id)

        upload_task = functools.partial(
            data_manager.upload_collection,
            history=galaxy_history,
            collection_type=collection_type,
            inputs=inputs,
            collection_name=collection_name,
        )
        result = await to_thread.run_sync(upload_task)
        return {"id": result.id, "name": result.name}

    except Exception as e:
        # Re-raise HTTPExceptions, otherwise wrap general exceptions
        if isinstance(e, HTTPException):
            raise
        raise InternalServerErrorException("An unexpected error occurred during collection upload")

    finally:
        # Always clean up the temporary directory in a thread
        cleanup_task = functools.partial(shutil.rmtree, tmpdir, ignore_errors=True)
        await to_thread.run_sync(cleanup_task)

# @router.get(
#     "/{history_id}/contents",
#     response_model=dataset.HistoryContentsResponse,
#     summary="List History Contents",
#     tags=["Histories & Data"]
# )
async def list_history_contents(
    history_id: str = Path(..., description="The ID of the Galaxy history.")
):
    """
    Retrieves a list of all visible datasets and dataset collections in a history.
    """

    galaxy_client = GalaxyClient(current_api_key.get())
    data_manager = DataManager(galaxy_client)

    try:
        galaxy_history = await run_in_threadpool(data_manager.gi.histories.get, history_id)
        datasets, collections = await run_in_threadpool(data_manager.list_contents, history=galaxy_history)

        # Format the response using the Pydantic schemas
        datasets_info = [{"id": d['id'], "name": d['name'], "type": "dataset"} for d in datasets]
        collections_info = [{"id": c['id'], "name": c['name'], "type": "dataset_collection"} for c in collections]

        return {"datasets": datasets_info, "collections": collections_info}
    except Exception as e:
        raise InternalServerErrorException("Failed to list history contents")

# @router.post(
#     "/{history_id}/create_collection",
#     response_model=dataset.CollectionResponse,
#     summary="Create a Dataset Collection from pre-existing history",
#     tags=["Histories & Data"]
# )
async def create_dataset_collection(
    history_id: str = Path(..., description="The ID of the Galaxy history."),
    collection_details: dataset.CollectionCreate = Body(...)
):
    """
    Creates a new dataset collection in a history from **pre-existing datasets**.

    **Note:** This endpoint assumes datasets have already been uploaded.
    The `element_identifiers` should contain the IDs of datasets already in Galaxy.
    """

    galaxy_client = GalaxyClient(current_api_key.get())
    data_manager = DataManager(galaxy_client)

    try:
        galaxy_history = await run_in_threadpool(data_manager.gi.histories.get, history_id)
        collection_type_enum = CollectionType(collection_details.collection_type)

        # use the `create_dataset_collection` method directly.
        payload = {
            "collection_type": collection_type_enum.value,
            "element_identifiers": collection_details.element_identifiers,
            "name": collection_details.name,
        }

        # Run the synchronous creation call in a thread pool
        created_collection = await run_in_threadpool(
            galaxy_history.create_dataset_collection,
            payload,
            wait=True
        )

        return {"id": created_collection.id, "name": created_collection.name}
    except ValueError:
        raise BadRequestException("Invalid collection_type: '{collection_details.collection_type}'")
    except Exception as e:
        raise InternalServerErrorException("Failed to create collection")


def _rmtree_sync(path):
    """Synchronous wrapper so it can run in a thread."""
    shutil.rmtree(path, ignore_errors=True)


@router.get(
        "/download",
        response_class = responses.FileResponse,
        summary= "Download datasets or dataset collection from a given list of ids",
        tags=["Histories & Data", "Download"]
        )
async def download_files(
    dataset_ids: List[str] = Query(default=[]),
    collection_ids: List[str] = Query(default=[]),
):

    galaxy_client = GalaxyClient(current_api_key.get())
    data_manager = DataManager(galaxy_client)
   
    if not dataset_ids and not collection_ids:
        raise BadRequestException("No dataset or collection IDs provided")
    
    # 1. Create a temp dir that we will clean up ourselves
    tmpdir = tempfile.mkdtemp(prefix="galaxy_dl_")
    tmpdir_path = pathlib.Path(tmpdir)

    try:
        outputs = [
            *(data_manager.gi.datasets.get(i) for i in dataset_ids),
            *(data_manager.gi.dataset_collections.get(i) for i in collection_ids),
        ]
        downloaded_files = await run_in_threadpool(
            data_manager.download_outputs, outputs=outputs, dest_dir=tmpdir_path
        )

        if len(downloaded_files) == 1:
            file_to_serve = downloaded_files[0]
            filename = re.sub(r'[\\/*?:"<>|]', '', file_to_serve.name).replace(' ', '_')
        else:
            # zip everything
            zip_path = tmpdir_path / "downloaded_outputs.zip"
            shutil.make_archive(str(zip_path.with_suffix("")), "zip", tmpdir_path)
            file_to_serve = zip_path
            filename = "downloaded_outputs.zip"
        
        # 2. Schedule cleanup after the response is sent
        background = BackgroundTasks()
        background.add_task(to_thread.run_sync, _rmtree_sync, tmpdir)

        return responses.FileResponse(
            path=file_to_serve,
            filename=filename,
            media_type="application/octet-stream",
            background=background,
        )

    except Exception as exc:
        # Clean up immediately on error
        to_thread.run_sync(_rmtree_sync, tmpdir)
        raise InternalServerErrorException("Download failed")
