import logging
from app.bioblend_server.server import bioblend_app

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("bioblend_server_main")

if __name__ == "__main__":
    logger.info('Firing up bioblend mcp server')
    bioblend_app.run()