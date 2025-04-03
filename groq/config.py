# filepath: /home/biniam/Desktop/Projects/mcp-xp/groq/config.py

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Example: Access an environment variable
GALAXY_URL = os.getenv("GALAXY_URL")
GALAXY_API_KEY = os.getenv("GALAXY_API_KEY")
LLM_API_KEY = os.getenv("LLM_API_KEY")