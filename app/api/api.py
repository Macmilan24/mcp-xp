import os
import httpx
from dotenv import load_dotenv
from typing import Optional, Literal

from fastapi import (
    APIRouter,
    Query,
    HTTPException
    )

from app.api.schemas.dataset import DatasetInfo, LocalDatasetImportResponse
from app.api.endpoints import (
    histories,
    tools,
    workflows,
    invocation
    )

from app.context import current_api_key
from app.config import GALAXY_URL

load_dotenv()

api_router = APIRouter()

# Include the histories router with a prefix and tags
api_router.include_router(
    histories.router,
    prefix="/histories",
    tags=["Histories & Data"]
)

# Include the workflows router with a prefix and tags
api_router.include_router(
    workflows.router,
    prefix="/workflows",
    tags=["Workflows"]
)

# Include the tools router with a prefix and tags
api_router.include_router(
    tools.router,
    prefix="/tools",
    tags=["Tools"]
)
# Include the invocation router with a prefix and tags
api_router.include_router(
    invocation.router,
    prefix="/invocation",
    tags=["Invocation"]
)


@api_router.post(
"/datasets/local_import",
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