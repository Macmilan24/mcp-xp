import logging
import os

LOG_FILE_PATH = './MCP_logger.log'


def configure_logging():
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    root_logger = logging.getLogger()

    # Clear any existing handlers to avoid duplicate or conflicting logs
    root_logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(LOG_FILE_PATH, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # Stream handler (CLI)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)