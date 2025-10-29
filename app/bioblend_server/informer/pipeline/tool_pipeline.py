# app/bioblend_server/informer/pipeline/tool_pipeline.py

import os
import json
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import logging

import requests
import yaml
from bioblend.galaxy import GalaxyInstance
from dotenv import load_dotenv
from jsonschema import validate, ValidationError
from tqdm import tqdm

# Module logger
logger = logging.getLogger("tool_pipeline")

# --- 1. Configuration ---
logger.info("Loading configuration...")
# Load environment variables from a .env file in this directory
load_dotenv()

GALAXY_URL = os.getenv("GALAXY_URL", "https://usegalaxy.org")
GALAXY_API_KEY = os.getenv("GALAXY_API_KEY")

if not GALAXY_API_KEY:
    raise ValueError("GALAXY_API_KEY must be set in the .env file")

# The final, clean output file
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "data", "processed_tools.json")

# Schema for validation is now embedded here to avoid extra files
TOOL_SCHEMA = {
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Galaxy Tool Metadata Schema",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "tool_id": {"type": "string"},
      "name": {"type": "string", "minLength": 1},
      "description": {"type": "string"},
      "categories": {
        "type": "array",
        "items": {"type": ["string", "null"]},
      },
      "version": {"type": "string"},
      "help": {"type": ["string", "null"]},
    },
    "required": ["tool_id", "name", "description", "categories", "version", "help"],
  },
}

# --- 2. Helper Functions (from preprocessor and downloader) ---

def clean_help_text(text: str) -> str:
    """Removes HTML, markdown, and other artifacts from help text."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[=_]{2,}", " ", text)
    text = re.sub(r"[│║╔╗╚╝╠╣═╦╩╬┼─┐└┘┌┴┬├┤┬┴┼]", " ", text)
    text = re.sub(r"[\+\|\-]+", " ", text)
    text = re.sub(r"[\*_`]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def fetch_and_process_tool(tool: dict, gi: GalaxyInstance) -> dict | None:
    """Fetches detailed tool info, extracts help text, and cleans it."""
    tool_id = tool.get("id", "")
    try:
        # Get raw XML to extract the help section
        raw_tool_url = f"{GALAXY_URL}/api/tools/{tool_id}/raw_tool_source"
        response = requests.get(raw_tool_url, params={'key': GALAXY_API_KEY})
        response.raise_for_status()
        tool_xml = response.text

        help_text = ""
        root = ET.fromstring(tool_xml)
        help_elem = root.find("help")
        if help_elem is not None and help_elem.text:
            help_text = help_elem.text.strip()

        # The final, processed record for this tool
        processed_tool = {
            "tool_id": tool_id,
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "categories": [tool.get("panel_section_name", "Uncategorized")],
            "version": tool.get("version", ""),
            "help": clean_help_text(help_text)
        }
        return processed_tool

    except Exception as e:
        logger.info(f"Error processing tool {tool_id}: {e}")
        return None

def validate_data(data: list[dict]) -> bool:
    """Validates the final data structure against the embedded schema."""
    logger.info("\n🧪 Validating final data against schema...")
    try:
        validate(instance=data, schema=TOOL_SCHEMA)
        logger.info("✅ Validation successful!")
        return True
    except ValidationError as e:
        logger.info(f"❌ Validation FAILED: {e.message}")
        return False

# --- 3. Main Execution Logic ---

def main():
    """Main function to run the entire tool processing pipeline."""
    start_time = datetime.now()
    logger.info(f"🛠️ Starting tool pipeline at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    logger.info(f"Connecting to Galaxy instance at {GALAXY_URL}...")
    gi = GalaxyInstance(url=GALAXY_URL, key=GALAXY_API_KEY)

    logger.info("Fetching base tool list from Galaxy...")
    try:
        # We fetch all tools, no sampling needed for the master list
        all_tools = gi.tools.get_tools()
        logger.info(f"Found {len(all_tools)} tools.")
    except Exception as e:
        logger.info(f"❌ Failed to fetch tools from Galaxy: {e}")
        return

    processed_tools = []
    # Using ThreadPoolExecutor for faster IO-bound operations
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_tool = {executor.submit(fetch_and_process_tool, tool, gi): tool for tool in all_tools}
        
        for future in tqdm(as_completed(future_to_tool), total=len(all_tools), desc="Processing tools"):
            result = future.result()
            if result:
                processed_tools.append(result)

    if not processed_tools:
        logger.info("❌ No tools were successfully processed. Exiting.")
        return

    logger.info(f"\nSuccessfully processed {len(processed_tools)} tools.")

    # Validate the final list of dictionaries
    if not validate_data(processed_tools):
        logger.info("Aborting due to validation failure.")
        return

    # Save the final, clean data
    logger.info(f"💾 Saving processed data to '{OUTPUT_FILE}'...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(processed_tools, f, indent=2, ensure_ascii=False)

    duration = datetime.now() - start_time
    logger.info(f"\n✅ Pipeline completed successfully in {duration}.")
    logger.info(f"Final output is located at: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()