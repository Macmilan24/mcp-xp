from fastapi import Header, HTTPException, status, Depends
from app.bioblend_server.galaxy import GalaxyClient


async def get_user_api_key(
    user_api_key: str | None = Header(default=None, alias="USER-API-KEY"),
):
    key = user_api_key
    if not key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="USER-API-KEY header is required for internal API calls",
        )
    return key


async def get_galaxy_client(api_key: str = Depends(get_user_api_key)):
    return GalaxyClient(user_api_key=api_key)
