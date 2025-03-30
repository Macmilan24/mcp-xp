from . import server
import asyncio


def main():
    """Main entry point for the package."""
    asyncio.run(server.serve())


# Optionally expose other important items at package level
if "__name__" == "__main__":
    main()