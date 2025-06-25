import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, Request
from app.AI.chatbot import ChatSession, initialize_session

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Optional: Add project root to Python path
# def session_file_path(user_ip: str) -> str:
#     safe_ip = user_ip.replace(":", "_")  # Windows-safe file naming
#     return os.path.join(SESSION_DIR, f"{safe_ip}.pkl")

# async def save_session(user_ip: str, session: ChatSession):
#     with open(session_file_path(user_ip), "wb") as f:
#         pickle.dump(session, f)

# def load_session(user_ip: str) -> ChatSession:
#     path = session_file_path(user_ip)
#     if os.path.exists(path):
#         with open(path, "rb") as f:
#             return pickle.load(f)
#     return None

# SESSION_DIR = "per_session"
# os.makedirs(SESSION_DIR, exist_ok=True)

# Store sessions per user IP
sessions = {}

class MessageRequest(BaseModel):
    message: str

app = FastAPI()


@app.get("/chat_history")
async def get_chat_history(request: Request):
    user_ip = request.client.host
    if user_ip not in sessions:
        chat_session = await initialize_session(user_ip)
        sessions[user_ip] = chat_session
        # return {"error": "Chat session not initiated for this IP"}

    return {"memory": sessions[user_ip].memory}


@app.post("/send_message")
async def send_message(request: Request, message: MessageRequest):
    user_ip = request.client.host
    if user_ip not in sessions:
            chat_session = await initialize_session(user_ip)
            sessions[user_ip] = chat_session
            # return {"error": "Chat session not initiated for this IP"}

    chat_session = sessions[user_ip]
    response = await chat_session.respond(model_id="gemini", user_input=message.message)
    return {"response": response}

@app.post("/list_tools")
async def list_tools(request: Request):
    print("ksdnflksfd")
    
    user_ip = request.client.host
    if user_ip not in sessions:
            chat_session = await initialize_session(user_ip)
            sessions[user_ip] = chat_session

    chat_session = sessions[user_ip]
    all_tools = []
    for server in chat_session.servers:
        print("server 1")
        tools = await server.list_tools()
        print("toosls uuuu ", tools)
        all_tools.extend([tool for tool in tools])
    return {"tools": all_tools}


@app.post("/upload_file")
async def uploaf_file():
        return 1
