import hashlib
import shutil
import asyncio
import pathlib
import logging


logger = logging.getLogger("Invocation")


def generate_request_hash(*args) -> str:
    """Generate hash for request deduplication"""
    request_string = "|".join(str(arg) if arg is not None else "None" for arg in args)
    return hashlib.md5(request_string.encode()).hexdigest()

def _rmtree_sync(path: pathlib.Path):
    shutil.rmtree(path, ignore_errors=True)
    
def log_task_error(task: asyncio.Task, *, task_name: str) -> None:
    """Log errors from completed background tasks."""
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        logger.info(f"Background task '{task_name}' was cancelled.")
        return
    
    if exc is not None:
        logger.error(
            f"Background task '{task_name}' failed",
            exc_info=exc,
        )