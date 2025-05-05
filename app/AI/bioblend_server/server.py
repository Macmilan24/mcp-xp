# app/AI/bioblend_server/server.py
import logging
from typing import List, Dict, Any # Added Dict, Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    ClientCapabilities,
    TextContent,
    Tool,
    ListRootsResult,
    RootsCapability,
)

print("bioblend_server server.py")

# Import the new functions from galaxy_tools
from app.AI.bioblend_server.galaxy_tools import (
    get_tools,
    get_tool,
    list_histories,
    get_history_details,
    create_history,
    delete_history,
    list_workflows,
    get_workflow_details,
    list_datasets_in_history,
    get_dataset_details,
    list_users,
    get_user_details,
    list_libraries,
    get_library_details,
)
from pydantic import BaseModel # Keep if needed for other parts, not strictly necessary for schema here

logger = logging.getLogger("bioblend_server")
# Ensure basicConfig is called only once, potentially in your main entry point
# logging.basicConfig(level=logging.INFO) # Comment out if configured elsewhere

async def serve():
    logger.info("Starting galaxyTools MCP Server...")
    server = Server("galaxyTools")

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        logger.info("Listing tools for galaxyTools server")

        tools = []

        # --- Existing Tools ---
        tools.append(
            Tool(
                name="galaxy_get_tools_list", # Renamed slightly for clarity
                description="Get a list of available tools (analysis tools like aligners, etc.) from the Galaxy instance.",
                inputSchema={
                    "type" : "object",
                    "properties": {
                        "number_of_tools": {
                            "type": "integer",
                            "description": "The maximum number of tools to fetch from the list.",
                            "default": 10,
                        },
                    },
                },
            )
        )

        tools.append(
            Tool(
                name="galaxy_get_tool_details", # Renamed slightly for clarity
                description="Get detailed information about a specific Galaxy analysis tool by its ID.",
                inputSchema={
                    "type" : "object",
                    "properties": {
                        "tool_id": {
                            "type": "string",
                            "description": "The ID of the analysis tool to fetch (e.g., 'upload1', 'cat1').",
                        },
                    },
                    "required" : ["tool_id"]
                },
            )
        )

        # --- New Tools ---

        # History Tools
        tools.append(
            Tool(
                name="galaxy_list_histories",
                description="List all accessible histories for the current user.",
                inputSchema={"type": "object", "properties": {}}, # No input needed
            )
        )
        tools.append(
            Tool(
                name="galaxy_get_history_details",
                description="Get detailed information about a specific history by its ID.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "history_id": {
                            "type": "string",
                            "description": "The ID of the history to fetch details for.",
                        },
                    },
                    "required": ["history_id"],
                },
            )
        )
        tools.append(
            Tool(
                name="galaxy_create_history",
                description="Create a new, empty history with a specified name.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The desired name for the new history.",
                        },
                    },
                    "required": ["name"],
                },
            )
        )
        tools.append(
            Tool(
                name="galaxy_delete_history",
                description="Delete a history by its ID. Can optionally purge (permanently delete) it.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "history_id": {
                            "type": "string",
                            "description": "The ID of the history to delete.",
                        },
                        "purge": {
                            "type": "boolean",
                            "description": "Set to true to permanently delete the history, false to just mark as deleted.",
                            "default": False,
                        }
                    },
                    "required": ["history_id"],
                },
            )
        )

        # Workflow Tools
        tools.append(
            Tool(
                name="galaxy_list_workflows",
                description="List all accessible workflows for the current user.",
                inputSchema={"type": "object", "properties": {}}, # No input needed
            )
        )
        tools.append(
            Tool(
                name="galaxy_get_workflow_details",
                description="Get detailed information about a specific workflow by its ID.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workflow_id": {
                            "type": "string",
                            "description": "The ID of the workflow to fetch details for.",
                        },
                    },
                    "required": ["workflow_id"],
                },
            )
        )

        # Dataset Tools
        tools.append(
            Tool(
                name="galaxy_list_datasets_in_history",
                description="List all datasets within a specific history.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "history_id": {
                            "type": "string",
                            "description": "The ID of the history whose datasets should be listed.",
                        },
                    },
                    "required": ["history_id"],
                },
            )
        )
        tools.append(
            Tool(
                name="galaxy_get_dataset_details",
                description="Get detailed information about a specific dataset by its ID.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string",
                            "description": "The ID of the dataset to fetch details for.",
                        },
                    },
                    "required": ["dataset_id"],
                },
            )
        )

        # User Tools (Note: Often require Admin privileges)
        tools.append(
            Tool(
                name="galaxy_list_users",
                description="List all users registered in the Galaxy instance. Requires admin privileges.",
                inputSchema={"type": "object", "properties": {}}, # No input needed
            )
        )
        tools.append(
            Tool(
                name="galaxy_get_user_details",
                description="Get detailed information about a specific user by their ID. Requires admin privileges.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "The ID of the user to fetch details for.",
                        },
                    },
                    "required": ["user_id"],
                },
            )
        )

        # Library Tools
        tools.append(
            Tool(
                name="galaxy_list_libraries",
                description="List all accessible data libraries in the Galaxy instance.",
                inputSchema={"type": "object", "properties": {}}, # No input needed
            )
        )
        tools.append(
            Tool(
                name="galaxy_get_library_details",
                description="Get detailed information about a specific data library by its ID.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "library_id": {
                            "type": "string",
                            "description": "The ID of the data library to fetch details for.",
                        },
                    },
                    "required": ["library_id"],
                },
            )
        )

        logger.info(f"Defined {len(tools)} tools for the LLM.")
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> list[TextContent]: # Use Dict[str, Any] for arguments
        logger.info(f"Received tool call request for: '{name}' with args: {arguments}")
        result_text = "" # Initialize result text

        try:
            # --- Map tool name to function call ---
            if name == "galaxy_get_tools_list":
                num_tools = arguments.get("number_of_tools", 10) # Use default if not provided
                result_text = get_tools(num_tools)
            elif name == "galaxy_get_tool_details":
                tool_id = arguments.get("tool_id")
                if not tool_id: raise ValueError("Missing required argument: tool_id")
                result_text = get_tool(tool_id)
            elif name == "galaxy_list_histories":
                result_text = list_histories()
            elif name == "galaxy_get_history_details":
                history_id = arguments.get("history_id")
                if not history_id: raise ValueError("Missing required argument: history_id")
                result_text = get_history_details(history_id)
            elif name == "galaxy_create_history":
                hist_name = arguments.get("name")
                if not hist_name: raise ValueError("Missing required argument: name")
                result_text = create_history(hist_name)
            elif name == "galaxy_delete_history":
                history_id = arguments.get("history_id")
                purge = arguments.get("purge", False) # Use default if not provided
                if not history_id: raise ValueError("Missing required argument: history_id")
                result_text = delete_history(history_id, purge)
            elif name == "galaxy_list_workflows":
                result_text = list_workflows()
            elif name == "galaxy_get_workflow_details":
                workflow_id = arguments.get("workflow_id")
                if not workflow_id: raise ValueError("Missing required argument: workflow_id")
                result_text = get_workflow_details(workflow_id)
            elif name == "galaxy_list_datasets_in_history":
                history_id = arguments.get("history_id")
                if not history_id: raise ValueError("Missing required argument: history_id")
                result_text = list_datasets_in_history(history_id)
            elif name == "galaxy_get_dataset_details":
                dataset_id = arguments.get("dataset_id")
                if not dataset_id: raise ValueError("Missing required argument: dataset_id")
                result_text = get_dataset_details(dataset_id)
            elif name == "galaxy_list_users":
                result_text = list_users()
            elif name == "galaxy_get_user_details":
                user_id = arguments.get("user_id")
                if not user_id: raise ValueError("Missing required argument: user_id")
                result_text = get_user_details(user_id)
            elif name == "galaxy_list_libraries":
                result_text = list_libraries()
            elif name == "galaxy_get_library_details":
                library_id = arguments.get("library_id")
                if not library_id: raise ValueError("Missing required argument: library_id")
                result_text = get_library_details(library_id)
            else:
                logger.error(f"Unknown tool name received: {name}")
                raise ValueError(f"Unknown tool name: {name}")

            # Log the successful result before returning
            # Limit log length to avoid flooding logs with huge outputs
            log_result = result_text[:500] + "..." if len(result_text) > 500 else result_text
            logger.info(f"Tool '{name}' executed successfully. Result preview: {log_result}")
            return [TextContent(type="text", text=result_text)]

        except Exception as e:
            logger.error(f"Error calling tool '{name}' with args {arguments}: {str(e)}", exc_info=True)
            # Return a user-friendly error message within the expected structure
            return [TextContent(type="text", text=f"Error executing tool '{name}': {str(e)}")]


    options = server.create_initialization_options()
    logger.info(f"Server initialization options created. Running server via stdio...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options, raise_exceptions=False) # Set raise_exceptions=False for production

# Make sure serve() is called if this script is run directly
# (though usually it's run via the command in servers_config.json)
if __name__ == "__main__":
     import asyncio
     # Ensure logging is configured if running standalone
     if not logger.handlers:
         logging.basicConfig(level=logging.INFO)
         logger.addHandler(logging.StreamHandler())
         logger.setLevel(logging.INFO)
     try:
         asyncio.run(serve())
     except KeyboardInterrupt:
         logger.info("Server stopped by user.")