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
from bioblend_server import get_tools
from pydantic import BaseModel



logger = logging.getLogger("bioblend_server")
logging.basicConfig(level=logging.INFO)

async def serve():
    # logger.info("Server is starting...")
    server = Server("galaxyTools")

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        logger.info("Listing tools for galaxyTools")
        # galaxy_tools = get_tools()
        
        tools = []

        tools.append(
                Tool(
                    name="galaxy_tools",
                    description="get galaxy tools",
                    inputSchema={
                        "type" : "object",
                        "properties": {
                            "number_of_tools": {
                                "type": "integer",
                                "description": "The number of tools to fetch",
                                "default": 10,
                            },
                        },
                    }, 
                )
            )

        return tools
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            logger.info(f"Calling tool: {name} with args: {arguments}")
            # logger.info(f"Tools: {galaxy_tools}")
            if name == "galaxy_tools":
                try:
                    galaxy_tools = get_tools(arguments["number_of_tools"])
                    logger.info(f"galaxy_tools: {galaxy_tools}")
                except Exception as e:
                    logger.error(f"error: {str(e)}")
                    return [TextContent(type="text", text=f"Error in executing get tools: {str(e)}")]
                return [
                    TextContent(
                        type="text",
                        text=galaxy_tools,
                    )
                ]
        except Exception as e:
            # logger.error(f"Tool error: {str(e)}")
            raise ValueError(f"Tool error: {str(e)}")
    
    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options, raise_exceptions=True)