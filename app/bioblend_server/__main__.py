# app/bioblend_server/__main__.py
import logging
import argparse
from app.bioblend_server.server import bioblend_app

logger = logging.getLogger("bioblend_server_main")

def run_server():
    parser = argparse.ArgumentParser(description="Run BioBlend MCP server")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host interface to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8897, help="Port to bind to (default: 8897)"
    )
    args = parser.parse_args()

    logger.info(f"Firing up bioblend MCP server on {args.host}:{args.port}")
    bioblend_app.run(
        transport="http",
        host=args.host,
        port=args.port,
    )

if __name__ == "__main__":
    run_server()