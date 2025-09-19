import asyncio
import logging
from typing import Any
from contextlib import AsyncExitStack
from fastmcp import Client 
from fastmcp.client.transports import StreamableHttpTransport


class Server:
    """Manages MCP server connections and tool execution. Supports STDIO or HTTP."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.client: Client | None = None  # Use fastmcp.Client
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self.logger = logging.getLogger(__class__.__name__)
        self.is_http = "url" in config  # Detect transport type

    async def initialize(self, custom_header:str = None) -> None:
        """Initialize the connection based on config (STDIO or HTTP)."""
        if self.is_http:
            
            url = self.config["url"]
            headers = self.config.get("headers", {})
            self.logger.info(f"Connecting to HTTP MCP server at {url}")
            # fastmcp.Client auto-detects HTTP/SSE based on URL
            mcp_client = Client(
                transport = StreamableHttpTransport(
                    url,
                    headers=headers
                )
            )
            self.client = await self.exit_stack.enter_async_context(mcp_client)
            
    async def list_tools(self) -> list[Any]:
        if not self.client:
            raise RuntimeError(f"Server {self.name} not initialized")
        tools_response = await self.client.list_tools()
        # self.logger.info(f"{type(tools_response)}: {tools_response}")
        self.logger.info("tools fetched")
        return tools_response  # Adjust if response differs

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] = {},
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        if not self.client:
            raise RuntimeError(f"Server {self.name} not initialized")
        attempt = 0
        while attempt < retries:
            try:
                self.logger.info(f"attempting tool execution: {tool_name}")
                result = await self.client.call_tool(tool_name, arguments)
                return result
            except Exception as e:
                attempt += 1
                if attempt < retries:
                    await asyncio.sleep(delay)
                else:
                    raise

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            await self.exit_stack.aclose()
            self.client = None
        except Exception as e:
            pass