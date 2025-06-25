import asyncio
from app.bioblend_server.server import serve


if __name__ == "__main__":
    asyncio.run(serve())