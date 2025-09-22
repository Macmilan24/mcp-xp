from fastapi import APIRouter
from app.api.endpoints import (
    histories,
    tools,
    workflows,
    internal_executor,
    internal_informer,
)

api_router = APIRouter()

# Include the histories router with a prefix and tags
api_router.include_router(
    histories.router, prefix="/histories", tags=["Histories & Data"]
)

# Include the workflows router with a prefix and tags
api_router.include_router(workflows.router, prefix="/workflows", tags=["Workflows"])

# Include the tools router with a prefix and tags
api_router.include_router(tools.router, prefix="/tools", tags=["Tools"])

api_router.include_router(
    internal_executor.router, prefix="/internal/executor", tags=["Internal :: Executor"]
)
api_router.include_router(
    internal_informer.router, prefix="/internal/informer", tags=["Internal :: Informer"]
)
