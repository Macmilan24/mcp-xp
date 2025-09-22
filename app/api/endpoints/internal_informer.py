from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.dependencies import get_galaxy_client
from app.bioblend_server.galaxy import GalaxyClient
from app.bioblend_server.informer.informer import GalaxyInformer

router = APIRouter()


class InformerRequest(BaseModel):
    query: str
    query_type: str
    entity_id: str | None = None


@router.post(
    "/get-entity-info",
    summary="Internal: Get detailed information about a Galaxy entity",
)
async def get_galaxy_info(
    request: InformerRequest, client: GalaxyClient = Depends(get_galaxy_client)
):
    informer = await GalaxyInformer.create(
        galaxy_client=client, entity_type=request.query_type
    )
    result = await informer.get_entity_info(
        search_query=request.query, entity_id=request.entity_id
    )
    return result
