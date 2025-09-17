import uvicorn
import logging
from app.bioblend_server.server import bioblend_app
from fastmcp.server.http import create_streamable_http_app

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("bioblend_server_main")
http_app = create_streamable_http_app(server=bioblend_app, streamable_http_path="/")

if __name__ == "__main__":
    port = 8001
    logger.info(f"Firing up bioblend mcp server as an HTTP service on port {port}")
    uvicorn.run(http_app, host="0.0.0.0", port=port)
