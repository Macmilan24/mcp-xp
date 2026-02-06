import httpx
from typing import Optional, Literal

from fastapi import (
    APIRouter,
    Query,
    HTTPException
    )

from app.context import current_api_key
from app.config import GALAXY_URL

from app.api.schemas.dataset import LocalDatasetImportResponse

router = APIRouter()

@router.post(
"/local_import",
response_model = LocalDatasetImportResponse,
summary="Import/Adopt a local dataset into galaxy, return the existing history_id and the dataset_id of the adopted file.",
tags = ["Histories & Data"]

)
async def proxy_adopt_local_file(
    file_path: str = Query(..., description="Absolute path to the local file"),
    file_name: Optional[str] = Query(None),
    extension: Optional[str] = Query(None),
    file_origin: Optional[Literal["annotation", "hypothesis"]] = Query(None),
    hist_id: Optional[str] = Query(None),
    ):
    
    params = {
    "file_path": file_path,
    "file_name": file_name,
    "extension": extension,
    "file_origin": file_origin,
    "hist_id": hist_id,
    }


    # remove None values to avoid polluting query string
    params = {k: v for k, v in params.items() if v is not None}
    api_key = current_api_key.get()
    headers = {"x-api-key": api_key}
    

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            
            resp = await client.post(
                url = f"{GALAXY_URL}/api/datasets/adopt_file",
                params=params,
                headers=headers,
            )
            
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Upstream service unreachable: {exc}",
                )

    if resp.status_code >= 400:
        raise HTTPException(
            status_code=resp.status_code,
            detail=resp.text,
            )

    response = resp.json()
    return LocalDatasetImportResponse(**response)

@router.post(
    "/object_import",
    response_model=LocalDatasetImportResponse,
    summary=(
        "Import/Adopt a MinIO/S3 object into Galaxy, return the existing history_id and the dataset_id of the adopted object."
        "Note: For this to work properly, the .dat object naming should follow the format 'dataset_{uuid}.dat' as Galaxy's internal code expects this format."
        ),
    tags=["Histories & Data"]
)
async def proxy_adopt_minio_object(
    bucket: str = Query(..., description="Bucket name (must match the bucket and the object store id in the config.)"),
    object_key: str = Query(..., description="Full object key (path for object store .../file_path/object_name.dat)."),
    file_name: Optional[str] = Query(None, description="Display name in history"),
    extension: Optional[str] = Query(None, description="Galaxy extension (fastq.gz, vcf, etc.)"),
    file_origin: Optional[Literal["annotation", "hypothesis"]] = Query(None),
    hist_id: Optional[str] = Query(None, description="Encoded history_id"),
):
    
    params = {
        "bucket": bucket,
        "object_key": object_key,
        "file_name": file_name,
        "extension": extension,
        "file_origin": file_origin,
        "hist_id": hist_id,
    }

    # Remove None values to avoid polluting query string
    params = {k: v for k, v in params.items() if v is not None}
    api_key = current_api_key.get()
    headers = {"x-api-key": api_key}
    

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.post(
                url=f"{GALAXY_URL}/api/datasets/adopt_object",
                params=params,
                headers=headers,
            )
            
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Upstream service unreachable: {exc}",
            )

    if resp.status_code >= 400:
        raise HTTPException(
            status_code=resp.status_code,
            detail=resp.text,
        )

    response = resp.json()
    return LocalDatasetImportResponse(**response)