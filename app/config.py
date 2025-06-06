
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Example: Access an environment variable
GALAXY_URL = os.getenv("GALAXY_URL")
GALAXY_API_KEY = os.getenv("GALAXY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 