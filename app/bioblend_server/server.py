import logging
import json # Import the json library
import os   # Import os to construct file paths
from typing import Dict, Any, Callable, Awaitable # Added Callable for type hinting
import sys
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool, LoggingMessageNotification
sys.path.append('.')

# The information retreiver tool
from app.bioblend_server.galaxy import get_galaxy_information

# --- Server Implementation ---
async def serve():
    logger = logging.getLogger("bioblend_server")
    # Ensure logging is configured
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("Server is starting...")
    server = Server("galaxyTools")

    @server.list_tools()
    async def list_tools():
        return [
                Tool(
                        name="get_galaxy_information",
                        description="Fetch detailed information about Galaxy entities "
                                    "(tools, datasets, workflows and workflow invocation details) and answer questions based on entities from galaxy entities.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "The user's query, accompanied by full and detailed contextual information"},
                                "query_type": {
                                    "type": "string",
                                    "enum": ["tool", "dataset", "workflow"],
                                    "description": "Entity type, select workflow for workflow details and workflow invocation details as well"
                                    },
                                "entity_id": { 
                                    "type":"string" ,
                                    "description": "Optional parameter used only when the user's query explicitly includes an ID, allowing retrieval of information by that ID."
                                    }
                            }, 
                            "required": ["query", "query_type"],
                        },
                    )
                ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        # print(f"Attempting to call tool: '{name}' with args: {arguments}", file=sys.stderr)
        try:
            result_text = None # Initialize result text

            # --- Handle REAL BioBlend Tools ---
            if name == "get_galaxy_information":
                try:
                    query = arguments.get("query")
                    query_type=arguments.get("query_type")
                    entity_id= arguments.get("entity_id", None)
                    result_text = await get_galaxy_information(query=query, query_type=query_type, entity_id=entity_id)
                    logger.info(f"Successfully executed REAL tool function: get_tools")
                except Exception as e:
                    logger.error(f"Error executing REAL tool function 'get_tools': {str(e)}", exc_info=True)
                    return [TextContent(type="text", text=f"Error executing tool '{name}': {str(e)}")]

            else:
                logger.warning(f"Attempted to call unknown tool: {name}")
                return [TextContent(type="text", text=f"Error: Unknown tool name '{name}' provided.")]

            # --- Wrap result in TextContent if successful ---
            if result_text is not None:
                 return [TextContent(type="text", text=str(result_text))]
            else:
                 # Should only happen if a real tool returns None unexpectedly or error handling fails
                 logger.error(f"Tool '{name}' matched but did not produce a result or handle error appropriately.")
                 return [TextContent(type="text", text=f"Internal server error processing tool '{name}'.")]

        except Exception as e:
            # Catch unexpected errors during tool dispatch or mock function execution
            logger.error(f"Unexpected error in call_tool dispatch for tool '{name}': {str(e)}", exc_info=True)
            return [TextContent(type="text", text=f"An unexpected server error occurred while trying to execute tool '{name}'.")]


    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
                        read_stream,
                        write_stream, 
                        options, 
                        raise_exceptions=True
                          )

# # Optional guard for direct execution (less useful now as it depends on the JSON)
# if __name__ == "__main__":
#     # import asyncio
#     # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     # asyncio.run(serve())

#     x=TextContent(type="text", text=f"Internal server error processing tool.")
#     print(type(x.text))
#     print(x)