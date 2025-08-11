from pydantic import BaseModel, Field
from typing import Literal, Optional

from fastmcp.server.middleware import Middleware, MiddlewareContext, CallNext
from mcp.types import CallToolRequestParams, CallToolResult

from contextvars import ContextVar

current_api_key_server: ContextVar[str] = ContextVar("current_api_key_server", default=None)

# Structure for the executor tool to respond with.
class ExecutorToolResponse(BaseModel):
    entity: Literal["tool", "workflow"] = Field(..., title="Entity")
    name: str = Field(..., title="Name")
    id: str = Field(..., title="Id")
    description: Optional[str] = Field(default=None, title="Description")
    action_link: str = Field(..., title="Action Link")

# Middleware to extract, save to context variable and remove the api_key from tool call arguments 
class ApiKeyMiddleware(Middleware):
    async def on_call_tool(
        self,
        context: MiddlewareContext[CallToolRequestParams],
        call_next: CallNext[CallToolRequestParams, CallToolResult]
    ) -> CallToolResult:
        params = context.message
        arguments = dict(params.arguments)
        api_key = arguments.pop('api_key', None)
        if api_key is None:
            raise ValueError("No API key provided")
        current_api_key_server.set(api_key)
        try:
            new_params = params.model_copy(update={'arguments': arguments})
            new_context = context.copy(message=new_params)
            result = await call_next(new_context)
        except Exception as e:
            raise e
        return result