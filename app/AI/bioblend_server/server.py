import logging
import json # Import the json library
import os   # Import os to construct file paths
from typing import List, Dict, Any, Callable # Added Callable for type hinting

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool


# Keep existing imports - DO NOT CHANGE
from app.AI.bioblend_server.galaxy_tools import get_tools, get_tool

logger = logging.getLogger("bioblend_server")
# Ensure logging is configured
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Constants ---
MOCK_TOOLS_JSON_PATH = os.path.join(os.path.dirname(__file__), "mock_tools.json")

# --- Generic Mock Function Executor ---
# This single function will handle the execution logic for ALL mock tools.
# It's identified by the tool_name passed to it.
def execute_generic_mock_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Logs execution and returns a standard mock response string."""
    logger.info(f"Executing MOCK function for: {tool_name} with args: {arguments}")
    result_text = f"Executed mock tool function: {tool_name}"
    logger.info(f"Mock function {tool_name} returning: '{result_text}'")
    return result_text

# --- Tool Loading and Dispatcher Setup ---
# Load mock tool definitions from JSON
try:
    with open(MOCK_TOOLS_JSON_PATH, 'r') as f:
        MOCK_TOOL_DEFINITIONS = json.load(f)
    logger.info(f"Successfully loaded {len(MOCK_TOOL_DEFINITIONS)} mock tool definitions from {MOCK_TOOLS_JSON_PATH}")
except FileNotFoundError:
    logger.error(f"Mock tools JSON file not found at {MOCK_TOOLS_JSON_PATH}. No mock tools will be available.")
    MOCK_TOOL_DEFINITIONS = []
except json.JSONDecodeError as e:
    logger.error(f"Error decoding JSON from {MOCK_TOOLS_JSON_PATH}: {e}. No mock tools will be available.")
    MOCK_TOOL_DEFINITIONS = []

# Create a dispatcher dictionary mapping mock tool names to the generic executor
# We use a lambda here to capture the tool_name for the generic function
MOCK_TOOL_DISPATCHER: Dict[str, Callable[[Dict[str, Any]], str]] = {
    tool_def["name"]: (lambda name=tool_def["name"]: lambda args: execute_generic_mock_tool(name, args))()
    for tool_def in MOCK_TOOL_DEFINITIONS
}
logger.info(f"Created dispatcher for {len(MOCK_TOOL_DISPATCHER)} mock tools.")

# --- Server Implementation ---
async def serve():
    logger.info("Server is starting...")
    server = Server("galaxyTools")

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        logger.info("Listing tools for galaxyTools server")
        tools = []

        # 1. Add the REAL BioBlend tools manually
        # tools.append(
        #     Tool(
        #         name="galaxy_tools",
        #         description="Fetch a list of tools from the connected Galaxy instance using BioBlend.",
        #         inputSchema={
        #             "type" : "object",
        #             "properties": {
        #                 "number_of_tools": {
        #                     "type": "integer",
        #                     "description": "The maximum number of tools to fetch from Galaxy.",
        #                     "default": 10,
        #                 },
        #             },
        #         },
        #     )
        # )
        # tools.append(
        #     Tool(
        #         name="galaxy_tool_by_id",
        #         description="Fetch detailed information about a specific tool by its ID from the Galaxy instance using BioBlend.",
        #         inputSchema={
        #             "type" : "object",
        #             "properties": {
        #                 "tool_id": {
        #                     "type": "string",
        #                     "description": "The exact ID of the tool to fetch from Galaxy (e.g., 'upload1').",
        #                 },
        #             },
        #             "required" : ["tool_id"]
        #         },
        #     )
        # )

        # 2. Add MOCK tools loaded from JSON
        for tool_def in MOCK_TOOL_DEFINITIONS:
            try:
                tools.append(
                    Tool(
                        name=tool_def["name"],
                        # Add "Mock Tool:" prefix to the description
                        description=f"Mock Tool: {tool_def.get('description_suffix', 'No description provided.')}",
                        inputSchema=tool_def.get("inputSchema", {"type": "object", "properties": {}}) # Use provided schema or default
                    )
                )
            except KeyError as e:
                logger.warning(f"Skipping mock tool definition due to missing key {e} in JSON: {tool_def}")
            except Exception as e:
                 logger.warning(f"Skipping mock tool definition due to unexpected error: {e} in JSON: {tool_def}")


        logger.info(f"Total tools listed: {len(tools)}")
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        logger.info(f"Attempting to call tool: '{name}' with args: {arguments}")
        try:
            result_text = None # Initialize result text

            # --- Handle REAL BioBlend Tools ---
            # if name == "galaxy_tools":
            #     try:
            #         logger.info(f"Calling REAL tool handler function: get_tools")
            #         num_tools = arguments.get("number_of_tools", 10)
            #         result_text = get_tools(num_tools)
            #         logger.info(f"Successfully executed REAL tool function: get_tools")
            #     except Exception as e:
            #         logger.error(f"Error executing REAL tool function 'get_tools': {str(e)}", exc_info=True)
            #         return [TextContent(type="text", text=f"Error executing tool '{name}': {str(e)}")]

            # elif name == "galaxy_tool_by_id":
            #     try:
            #         logger.info(f"Calling REAL tool handler function: get_tool")
            #         tool_id = arguments.get("tool_id")
            #         if not tool_id:
            #             logger.error(f"Missing 'tool_id' argument for tool: {name}")
            #             return [TextContent(type="text", text=f"Error: Missing required argument 'tool_id' for tool '{name}'.")]
            #         result_text = get_tool(tool_id)
            #         logger.info(f"Successfully executed REAL tool function: get_tool for id: {tool_id}")
            #     except Exception as e:
            #         logger.error(f"Error executing REAL tool function 'get_tool' with id '{arguments.get('tool_id')}': {str(e)}", exc_info=True)
            #         return [TextContent(type="text", text=f"Error executing tool '{name}': {str(e)}")]

            # --- Handle MOCK Tools using the Dispatcher ---
            if name in MOCK_TOOL_DISPATCHER:
                 mock_function = MOCK_TOOL_DISPATCHER[name]
                 # The mock_function already includes the tool name via the lambda capture
                 result_text = mock_function(arguments)
                 # Logging is now done inside execute_generic_mock_tool

            # --- Handle Unknown Tool ---
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
        await server.run(read_stream, write_stream, options, raise_exceptions=False)

# # Optional guard for direct execution (less useful now as it depends on the JSON)
# if __name__ == "__main__":
#      import asyncio
#      logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#      # Check if JSON loading worked before trying to run
#      if not MOCK_TOOL_DEFINITIONS:
#          logger.error("Cannot run server directly, mock tools failed to load.")
#      else:
#          asyncio.run(serve())