import os
import sys
from pydantic import BaseModel
from fastapi import FastAPI, Request
from app.AI.chatbot import ChatSession, initialize_session
import logging

from app.log_setup import configure_logging
from app.api.middleware import GalaxyAPIKeyMiddleware
from app.api.api import api_router 

configure_logging()

logger= logging.getLogger('main')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Optional: Add project root to Python path
sessions = {}

class MessageRequest(BaseModel):
    message: str

app = FastAPI(
    title="Galaxy API",
    description="An API to dynamically interact with Galaxy and Galaxy agent",
)

# Add middleware
app.add_middleware(GalaxyAPIKeyMiddleware)

# Include the API router
app.include_router(api_router, prefix="/api")



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
    response = await chat_session.respond(model_id="openai", user_input=message.message)
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
        logger.info(f'found tools: {[tool.name for tool in tools.tools]}')
        all_tools.extend([tool for tool in tools.tools])
    return {"tools": all_tools}