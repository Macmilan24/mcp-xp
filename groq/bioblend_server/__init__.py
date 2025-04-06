# import server

# __all__ = ["server"]
from .galaxy_tools import get_tools, setup_instance

import asyncio
from .server import serve


if __name__ == "__main__":
    asyncio.run(serve())


__all__ = ["get_tools", "setup_instance"]