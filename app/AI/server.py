import os
import asyncio
from typing import Any
from contextlib import AsyncExitStack
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
import logging

from app.context import current_api_key


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self.logger = logging.getLogger(__class__.__name__)

        # add Mcp url from config
        self.mcp_server_url = self.config.get("url")

    async def initialize(self) -> None:
        """Initialize the server connection."""
        if not self.mcp_server_url:
            raise ValueError("MCP Server Url is not configured.")
        try:
            self.logger.info("Attempting HTTP connection to MCP")

            http_transport = await self.exit_stack.enter_async_context(
                streamablehttp_client(url=self.mcp_server_url)
            )
            read, write, _ = http_transport

            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )

            await session.initialize()
            self.session = session
            self.logger.info(f"Server {self.name} connected to MCP server")

        except Exception as e:
            # logging.error(f"Error initializing server {self.name}: {e}")
            self.logger.error(
                f"Error initializing HTTP connection to MCP server {self.name}: {e}"
            )
            await self.cleanup()
            raise

    async def list_tools(self) -> list[Any]:
        if not self.session:
            # logging.error(f"Server {self.name} not initialized")
            raise RuntimeError(f"Server {self.name} not initialized")

        # logging.debug("getting tools")
        tools_response = await self.session.list_tools()
        self.logger.info("tools fetched")
        return tools_response

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] = {},
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                self.logger.info(f"attempting tool execution: {tool_name}")
                # logging.info(f"Executing {tool_name}...")
                exec_arguments = {**arguments, "api_key": current_api_key.get()}
                result = await self.session.call_tool(tool_name, exec_arguments)

                return result

            except Exception as e:
                attempt += 1
                if attempt < retries:
                    # logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    # logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                # logging.error(f"Error during cleanup of server {self.name}: {e}")
                pass
