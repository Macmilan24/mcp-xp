from fastapi import FastAPI
import uvicorn
import os
import sys
from app.AI.chatbot import ChatSession, initialize_session
from pydantic import BaseModel
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Optional: Add project root to Python path

class MessageRequest(BaseModel):
    message: str

_chatSession = None

app = FastAPI()

@app.post("/initiate_chat")
async def initiate_chat():
    """
    Initiate a chat session
    """
    global _chatSession
   
    _chatSession = await initialize_session()
    return {"message": "Chat session initiated"}


@app.post("/send_message")
async def send_message(model_id: str, request: MessageRequest):
    """
    Send a message to the chat session with a specific model ID
    """
    global _chatSession
    if _chatSession is None:
        return {"error": "Chat session not initiated"}
    response = await _chatSession.respond(
        model_id=model_id,
        user_input=request.message,
    )
    if response is None:
        return {"error": "No response from model"}
    return {"response": response}