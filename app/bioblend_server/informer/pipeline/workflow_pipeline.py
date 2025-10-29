# app/bioblend_server/informer/pipeline/workflow_pipeline.py

import os
import json
import re
from datetime import datetime
import logging
import requests
from dotenv import load_dotenv
from jsonschema import validate, ValidationError
from tqdm import tqdm

# Module logger
logger = logging.getLogger("workflow_pipeline")

# --- 1. Configuration ---
logger.info("Loading configuration...")
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
if not GITHUB_TOKEN:
    logger.info("⚠️ No GITHUB_TOKEN found. GitHub API requests may be rate-limited.")

GITHUB_API_URL = "https://api.github.com/repos/galaxyproject/iwc/contents/workflows"
RAW_BASE_URL = "https://raw.githubusercontent.com/galaxyproject/iwc/main"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "data", "processed_workflows.json")

WORKFLOW_SCHEMA = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "required": ["category", "workflow_repository", "tool_names", "readme_cleaned", "raw_download_url"],
    "properties": {
      "category": {"type": "string"},
      "workflow_repository": {"type": "string"},
      "tool_names": {"type": "array", "items": {"type": "string"}},
      "readme_cleaned": {"type": "string"},
      "raw_download_url": {"type": "string"},
    },
  },
}

# --- 2. Helper Functions ---

def github_api_get(url):
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()

def fetch_file_content(path_or_url):
    """
    Fetch text content from either a full raw GitHub URL or a repo-relative path.
    """
    url = path_or_url
    if not isinstance(path_or_url, str):
        raise ValueError("fetch_file_content requires a string URL or path")
    if not path_or_url.startswith("http"):
        url = f"{RAW_BASE_URL}/{path_or_url.lstrip('/')}"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.text

def clean_text(text: str) -> str:
    """Cleans README or other text content."""
    if not isinstance(text, str): return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\*_`#]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_ga_content(ga_text: str) -> list:
    """Parses a .ga file string to extract tool names."""
    tools_used = []
    try:
        data = json.loads(ga_text)
        for step in data.get("steps", {}).values():
            if step.get("type") == "tool":
                # Prefer tool_id if available, fallback to name
                tool_id = step.get("tool_id")
                if tool_id and tool_id not in tools_used:
                    tools_used.append(tool_id)
    except json.JSONDecodeError:
        logger.info("Warning: Could not parse .ga file content.")
    return tools_used

def process_workflow_repo(category: str, repo: dict) -> dict | None:
    """Scans a single workflow repository and returns a processed dictionary."""
    repo_name = repo["name"]
    base_path = f"{category}/{repo_name}"
    url = f"{GITHUB_API_URL}/{category}/{repo_name}"
    
    try:
        repo_contents = github_api_get(url)
        readme_url = None
        main_ga_file = None

        for item in repo_contents:
            if item["type"] == "file":
                if item["name"].lower() == "readme.md":
                    # Use direct download_url when available
                    readme_url = item.get("download_url") or item.get("path")
                if item["name"].endswith(".ga") and not main_ga_file:
                    main_ga_file = item

        if not main_ga_file:
            return None  # Skip if no workflow file found

        # Fetch GA content (prefer download_url)
        ga_source = main_ga_file.get("download_url") or main_ga_file["path"]
        ga_text = fetch_file_content(ga_source)

        # Fetch README content if present; don't fail the whole repo if it 404s
        readme_content = ""
        if readme_url:
            try:
                readme_content = fetch_file_content(readme_url)
            except Exception as e:
                logger.info(f"Note: README fetch failed for {repo_name}: {e}")
                readme_content = ""

        return {
            "workflow_id": f"iwc_{repo_name.lower()}",
            "category": category.lower(),
            "workflow_repository": repo_name.lower(),
            "tool_names": parse_ga_content(ga_text),
            "readme_cleaned": clean_text(readme_content),
            "raw_download_url": main_ga_file.get("download_url", ""),
        }
    except Exception as e:
        logger.info(f"Could not process repo {repo_name}: {e}")
        return None

def validate_data(data: list[dict]) -> bool:
    """Validates the final data structure against the embedded schema."""
    logger.info("\n🧪 Validating final data against schema...")
    try:
        validate(instance=data, schema=WORKFLOW_SCHEMA)
        logger.info("✅ Validation successful!")
        return True
    except ValidationError as e:
        logger.info(f"❌ Validation FAILED: {e.message}")
        return False

# --- 3. Main Execution Logic ---
def main():
    """Main function to run the entire workflow processing pipeline."""
    start_time = datetime.now()
    logger.info(f"🛠️ Starting workflow pipeline at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    all_workflows = []
    try:
        categories = [cat for cat in github_api_get(GITHUB_API_URL) if cat["type"] == "dir"]
        for cat in tqdm(categories, desc="Scanning Categories"):
            repos = [repo for repo in github_api_get(cat["url"]) if repo["type"] == "dir"]
            for repo in repos:
                processed = process_workflow_repo(cat["name"], repo)
                if processed:
                    all_workflows.append(processed)
    except Exception as e:
        logger.info(f"❌ An error occurred during GitHub scraping: {e}")
        return

    if not all_workflows:
        logger.info("❌ No workflows were successfully processed. Exiting.")
        return

    logger.info(f"\nSuccessfully processed {len(all_workflows)} workflows.")

    if not validate_data(all_workflows):
        logger.info("Aborting due to validation failure.")
        return

    logger.info(f"💾 Saving processed data to '{OUTPUT_FILE}'...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_workflows, f, indent=2, ensure_ascii=False)

    duration = datetime.now() - start_time
    logger.info(f"\n✅ Pipeline completed successfully in {duration}.")
    logger.info(f"Final output is located at: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()