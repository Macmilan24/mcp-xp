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


async def serve():
    print("Server is starting...")

    server = Server("galaxyTools")

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        return [
            Tool(
                name="get_galaxy_tools",
                description="Get Galaxy Tools",
                inputschema={},
            )
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            if name == "get_galaxy_tools":
                # Here you would call the function to get the tools
                # For example:
                # tools = get_galaxy_tools(arguments)
                # return tools
                return [
                    TextContent(
                        type="text",
                        text="here are the galaxy tools",
                    )
                ]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options, raise_exceptions=True)