import os
import sys
import json
import httpx
import asyncio
import logging
import redis

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from pydantic import BaseModel
from cryptography.fernet import Fernet
from contextlib import asynccontextmanager

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.openapi.utils import get_openapi
from app.AI.chatbot import ChatSession, initialize_session

from app.utils import import_published_workflows
from app.log_setup import configure_logging
from app.api.middleware import JWTGalaxyKeyMiddleware, RateLimiterMiddleware
from app.api.api import api_router 
from app.api.socket_manager import ws_manager, SocketMessageEvent
from app.orchestration.invocation_cache import InvocationCache
from app.orchestration.invocation_tasks import InvocationBackgroundTasks

load_dotenv()

# Global variables for cache and background tasks
# Initialize Redis client
redis_client = redis.Redis(host='localhost', port=os.getenv("REDIS_PORT"), db=0, decode_responses=True)
invocation_cache = None
background_tasks = None
sessions = {}

logger= logging.getLogger('main')

GALAXY_URL = os.getenv("GALAXY_URL")
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY is missing from environment")
fernet = Fernet(SECRET_KEY.encode() if isinstance(SECRET_KEY, str) else SECRET_KEY)


configure_logging()

# Base Schemas

class MessageRequest(BaseModel):
    message: str

class GalaxyUserAccount(BaseModel):
    username: str
    api_token: str


APP_DESCRIPTION = """
- This **FastAPI application** provides a RESTful API for dynamically interacting with the **Galaxy platform** and the **Galaxy Agent**.\n
- It supports operations such as **executing Galaxy tools and workflows**, **managing history and datasets**, and **coordinating tasks** through the agent.\n\n  
- All API requests require **authentication** via an encrypted **USER-API-KEY token** in the request header.\n
- This token is obtained by **registering and validating your Galaxy user API key**.\n
- Requests **without a valid registered key will be rejected**.

"""

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    # Generate default schema
    openapi_schema = get_openapi(
        title="Galaxy Interaction API",
        version="1.0.0",
        description= APP_DESCRIPTION,
        routes=app.routes,
    )

    # Bearer JWT security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }

    # Apply globally
    openapi_schema["security"] = [{"BearerAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown tasks"""
    global redis_client, invocation_cache, background_tasks
    
    # Startup
    try:        
        # Test Redis connection
        await asyncio.get_event_loop().run_in_executor(None, redis_client.ping)
        logger.info("Redis connection established")
        
        # Initialize cache
        invocation_cache = InvocationCache(redis_client)
        
        # Initialize background tasks
        background_tasks = InvocationBackgroundTasks(invocation_cache, redis_client)
        
        # Start background tasks
        asyncio.create_task(background_tasks.start_cache_warming_task())
        asyncio.create_task(background_tasks.start_cleanup_task())
        
        logger.info("Background tasks started")
        logger.info("Application startup complete")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield  # Application runs here
    
    # Shutdown
    try:
        logger.info("Shutting down application...")
        
        # Cancel background tasks
        for task in asyncio.all_tasks():
            if not task.done():
                task.cancel()
        
        # Close Redis connection
        if redis_client:
            redis_client.close()
        
        logger.info("Shutdown complete")
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")



# Initialize the FastAPI application with a lifespan context
app = FastAPI(lifespan=lifespan)



# Add middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(JWTGalaxyKeyMiddleware)
app.add_middleware(RateLimiterMiddleware, redis_client=redis_client)

# Include the API router
app.include_router(api_router, prefix="/api")

# Override FastAPI’s OpenAPI generator
app.openapi = custom_openapi


@app.get("/", tags=["Root"])
async def read_root():
    """Add a simple root endpoint for a health check"""

    return {"message": "Welcome to the Galaxy API!"}


@app.get("/chat_history", tags=["Agent"])
async def get_chat_history(request: Request):
    """Conversation history in a session"""

    user_ip = request.client.host
    if user_ip not in sessions:
        chat_session = await initialize_session(user_ip)
        sessions[user_ip] = chat_session
        # return {"error": "Chat session not initiated for this IP"}

    return {"memory": sessions[user_ip].memory}


@app.post("/send_message", tags=["Agent"])
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


@app.get("/list_tools", tags=["Agent"])
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


@app.post("/register-user",
          response_model=GalaxyUserAccount,
          tags =["Signup Auth"]
           )
async def get_create_galaxy_user_and_key(
    email: str = Query(description="email of the user to be registered to our galaxy instance."),
    password: str = Query(description="passcode to use for our galaxy isntance.")
    ) -> tuple[str, str]:

    """ Register Galaxy user from platform using the same credentials   """

    # 1. Create user via admin API
    username=None
    api_token=None

    try:
        logger.info("creating galaxy user account from galaxy service.")
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                url = f"{GALAXY_URL}/api/users",
                headers={"x-api-key": os.getenv("GALAXY_API_KEY")},
                json={
                    "username": email.split("@")[0],  # or whatever scheme you like
                    "email":    email,
                    "password": password
                }
            )
                
        resp.raise_for_status()
        galaxy_user_id = resp.json()["id"]
        username= resp.json()["username"]
        logger.info(f"Galaxy account created with username {username}")

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 400:

            # Else Fetch user by email if already exists
            try:
                logger.info("account already exists, getting api.")
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(
                        f"{GALAXY_URL}/api/users",
                        headers={"x-api-key": os.getenv("GALAXY_API_KEY")},
                        params={"f_email": email}
                    )
                resp.raise_for_status()
                users = resp.json()
                galaxy_user_id = users[0]["id"]
                username= users[0]["username"]
                logger.info(f"Galaxy User fetched with username {username}")

            except HTTPException as e: 
                raise HTTPException(status_code=400, detail= f"error getting/creating galaxy user: {e}")
            except Exception as e:
                logger.error(f"Error: {e}")
                raise
        elif e.response.status_code == 401:
            logger.error(f"Unauthorized admin id: {e}")
            raise HTTPException(status_code=401, detail= f"Unauthorized admin id: {e}")
        else:
            raise Exception(f"error caused during getting api_key for the user: {e}")

    except HTTPException as e:
        logger.error(f"errror creating user acount: {e}")
        raise HTTPException(status_code= 500, detail=f"error creating user account: {e}")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
 
    # 2. Return generated galaxy api-key
    try:
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            key_resp = await client.post(
                url = f"{GALAXY_URL}/api/users/{galaxy_user_id}/api_key",
                headers={"x-api-key": os.getenv("GALAXY_API_KEY")},
                json={
                    "name": "auto-generated from platform"
                    }
            )

        key_resp.raise_for_status()
        api_key = key_resp.json()

        payload = json.dumps({"apikey": api_key}).encode("utf-8")
        api_token = fernet.encrypt(payload).decode("utf-8")
        logger.info(f"Galaxy api-key extracted and encrypted for user with galaxy id {galaxy_user_id}")

    except HTTPException as e:
        logger.error(f"error creating galaxy user api key: {e}")
        raise HTTPException(status=500, detail= f"error getting galaxy user api-key: {e}")
    except Exception as e:
        logger.error(f"Error: {e}")

    # import missing public workflows into the users account.    
    import_published_workflows(galaxy_url=GALAXY_URL, api_key=api_key)

    return  GalaxyUserAccount(
        username = username,
        api_token = api_token
    )

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
                    event=SocketMessageEvent.ping,
                    data={},
                    tracker_id=tracker_id
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