import logging
from typing import List
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    ClientCapabilities,
    TextContent,
    Tool,
    ListRootsResult,
    RootsCapability,
)
from galaxy_tools import get_tools

logger = logging.getLogger("bioblend_server")
logging.basicConfig(level=logging.INFO)

async def serve():
    logger.info("Server is starting...")
    server = Server("galaxyTools")

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        logger.info("Listing tools for galaxyTools")
        galaxy_tools = get_tools()
        tools = []
        for tool in galaxy_tools:
            tools.append(
                Tool(
                    name=tool.name,
                    description=tool.description,
                    inputSchema=tool.input_schema,  # Changed from 'inputSchema' to match MCP types
                )
            )
        
        logger.info(f"Returning tools: {[tool.name for tool in tools]}")
        return tools
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        logger.info(f"Calling tool: {name} with args: {arguments}")
        try:
            if name == "get_galaxy_tools":
                return [
                    TextContent(
                        type="text",
                        text="here are the galaxy tools",
                    )
                ]
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
        except Exception as e:
            logger.error(f"Tool error: {str(e)}")
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options, raise_exceptions=True)