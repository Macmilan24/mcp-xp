import os
import sys
import asyncio
import logging
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.openapi.utils import get_openapi

load_dotenv()

from app.AI.chatbot import ChatSession, initialize_session

from app.log_setup import configure_logging
from app.api.middleware import GalaxyAPIKeyMiddleware
from app.api.api import api_router
from app.api.socket_manager import ws_manager, SocketMessageEvent

configure_logging()

logger = logging.getLogger("main")
sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)  # Optional: Add project root to Python path
sessions = {}


class MessageRequest(BaseModel):
    message: str


APP_DESCRIPTION = """
This FastAPI application provides a RESTful API for dynamically interacting with the Galaxy platform and the Galaxy Agent.
It supports operations such as executing Galaxy tools and workflows, managing history and datasets, and coordinating tasks through the agent.

All requests to this API require authentication via the USER-API-KEY header.
This key uniquely identifies the user and ensures secure access to Galaxy resources.
Requests without a valid key will be rejected.
"""


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    # Generate default schema
    openapi_schema = get_openapi(
        title="Galaxy API",
        version="1.0.0",
        description=APP_DESCRIPTION,
        routes=app.routes,
    )

    # Add global security scheme for API key in header
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {
            "type": "apiKey",
            "in": "header",
            "name": "USER-API-KEY",
        }
    }

    # Apply this scheme globally, every endpoint in docs will require it
    openapi_schema["security"] = [{"APIKeyHeader": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app = FastAPI()


# Add middleware
app.add_middleware(GalaxyAPIKeyMiddleware)

# Include the API router
app.include_router(api_router, prefix="/api")

# Override FastAPI’s OpenAPI generator
app.openapi = custom_openapi


@app.get("/", tags=["Root"])
async def read_root():
    """Add a simple root endpoint for a health check"""

    return {"message": "Welcome to the Galaxy API!"}


@app.get("/chat_history")
async def get_chat_history(request: Request):
    """Conversation history in a session"""

    user_ip = request.client.host
    if user_ip not in sessions:
        chat_session = await initialize_session(user_ip)
        sessions[user_ip] = chat_session
        # return {"error": "Chat session not initiated for this IP"}

    return {"memory": sessions[user_ip].memory}


@app.post("/send_message")
async def send_message(request: Request, message: MessageRequest):
    """Conversate with the Galaxy Agent"""
    from app.context import current_api_key

    logger.info(f"Current user api: ******{current_api_key.get()[-4:]}")

    user_ip = request.client.host
    if user_ip not in sessions:
        chat_session: ChatSession = await initialize_session(user_ip)
        sessions[user_ip] = chat_session
        # return {"error": "Chat session not initiated for this IP"}

    chat_session = sessions[user_ip]
    response = await chat_session.respond(model_id="gemini", user_input=message.message)
    return {"response": response}


@app.get("/list_tools")
async def list_tools(request: Request):
    """List MCP server tools available for the LLM"""

    logger.info("listing tools")

    user_ip = request.client.host
    if user_ip not in sessions:
        chat_session = await initialize_session(user_ip)
        sessions[user_ip] = chat_session

    chat_session = sessions[user_ip]
    all_tools = []
    for server in chat_session.servers:
        logger.info(f"server {server.name}")
        tools = await server.list_tools()
        logger.info(f"found tools: {[tool.name for tool in tools.tools]}")
        all_tools.extend([tool for tool in tools.tools])
    return {"tools": all_tools}


@app.websocket("/ws/{tracker_id}")
async def websocket_endpoint(websocket: WebSocket, tracker_id: str):
    # Accept and register socket in the tracker room
    await ws_manager.connect(websocket, tracker_id)

    # Background keepalive pinger
    async def ping_loop():
        try:
            while True:
                await asyncio.sleep(30)
                # Keeping shape consisitent
                await ws_manager.broadcast(
                    event=SocketMessageEvent.ping, data={}, tracker_id=tracker_id
                )
        except WebSocketDisconnect:
            # Exit silently
            pass
        except Exception as e:
            logger.error(f"error in websocket pinging: {e}")
            pass

    ping_task = asyncio.create_task(ping_loop())

    try:
        # Block until the client disconnects (close frame)
        while True:
            try:
                # keeps the connection open
                await websocket.receive_json()
            except WebSocketDisconnect:
                break
            except RuntimeError as e:
                logger.info(f"RuntimeError error caused: {e} ")
                break
    finally:
        ping_task.cancel()
        try:
            await ping_task
        except asyncio.CancelledError as e:
            logger.error(f"asyncio CancelledError error: {e}")
            pass
        await ws_manager.disconnect(websocket, tracker_id)
