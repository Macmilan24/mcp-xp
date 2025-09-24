from pydantic import BaseModel, Field
from typing import Literal, Optional
from contextvars import ContextVar

from fastmcp.server.middleware import Middleware, MiddlewareContext, CallNext
from mcp.types import CallToolRequestParams, CallToolResult

from app.api.security import decrypt_api_key_from_token

current_api_key_server: ContextVar[str] = ContextVar(
    "current_api_key_server", default=None
)


class ExecutorToolResponse(BaseModel):
    entity: Literal["tool", "workflow"] = Field(..., title="Entity")
    name: str = Field(..., title="Name")
    id: str = Field(..., title="Id")
    description: Optional[str] = Field(default=None, title="Description")
    action_link: str = Field(..., title="Action Link")


class JWTAuthMiddleware(Middleware):
    """
    A FastMCP middleware that enforces JWT-based authentication for tool calls.
    """

    async def on_call_tool(
        self,
        context: MiddlewareContext[CallToolRequestParams],
        call_next: CallNext[CallToolRequestParams, CallToolResult],
    ) -> CallToolResult:
        params = context.message
        arguments = dict(params.arguments)

        token = arguments.pop("token", None)
        if not token:
            raise PermissionError(
                "Authentication failed: 'token' argument is missing from the tool call."
            )

        try:
            api_key = decrypt_api_key_from_token(token)
            if not api_key:
                raise PermissionError(
                    "Authentication failed: The provided token is invalid or expired."
                )

            current_api_key_server.set(api_key)

            new_params = params.model_copy(update={"arguments": arguments})
            new_context = context.copy(message=new_params)

            return await call_next(new_context)

        except PermissionError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred in the authentication middleware: {e}"
            )
