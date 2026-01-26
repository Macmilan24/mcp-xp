import re
import logging
from bioblend.galaxy import GalaxyInstance

log = logging.getLogger("Import workflow")

def import_published_workflows(galaxy_url: str, api_key: str):
    """Imports published workflows from a Galaxy instance that aren't already in the user's account."""
    try:
        imported = []
        skipped = []
        failed = []

        try:
            # Connect to Galaxy
            gi = GalaxyInstance(url=galaxy_url, key=api_key)
        except Exception as e:
            log.error(f"Failed to connect to Galaxy. Please check URL and API key. Error: {e}")

        # Get the NAMES of existing workflows for deduplication
        my_workflows = gi.workflows.get_workflows()
        my_wf_names = {wf.get("name") for wf in my_workflows}
        
        # Build a normalized set: remove "imported: " prefix if it exists
        normalized_wf_names = {name.lower().removeprefix("imported: ").strip() for name in my_wf_names if name}

        # Get all published workflows
        published_wfs = gi.workflows.get_workflows(published=True)

        # Import missing workflows by checking their NAME
        for wf in published_wfs:
            wf_id = wf["id"]
            wf_name = wf["name"]
            wf_name_norm = wf_name.lower().removeprefix("imported: ").strip()

            # Skip if a workflow with the same name is already present
            if wf_name_norm in normalized_wf_names:
                skipped.append(wf_name)
                continue

            # Try to import the workflow
            try:
                gi.workflows.import_shared_workflow(wf_id)
                imported.append(wf_name)
                log.debug(f"Successfully imported workflow: {wf_name} to the user account.")
            except Exception as e:
                failed.append((wf_name, str(e)))
                log.error(f"Failed to import {wf_name}: {e}")
        if imported:
            log.info(f"{len(imported)} Workflows have been imported.")
        if failed:
            log.error(f"{len(failed)} workflows have failed to be imported.")
    except Exception as e:
        log.error(f"Error trying to import workflows: {e}")
        
    
def _extract_json_from_llm_response(content: str) -> str:
    """
    Extracts a JSON string from a given content string, handling various formatting scenarios.
    The method attempts to extract JSON data from the input string by:
        1. Looking for a JSON code block delimited by triple backticks (```json ... ```).
        2. Unquoting the string if the JSON is wrapped in single or double quotes.
        3. Searching for the first JSON object or array using a regular expression.
        4. Returning the raw string if none of the above methods succeed.
    Args:
        content (str): The input string potentially containing JSON data.
    Returns:
        str: The extracted JSON string or the original content if extraction fails.
    """
    try:
        
        # 1. Try to extract JSON inside ```json ... ```
        start = content.find("```json")
        end = content.rfind("```")
        if start != -1 and end != -1 and end > start:
            json_content = content[start + 7:end].strip()
            return json_content
        else:
            json_content = content.strip()

        # 2. If it's quoted JSON, unquote and retry
        if (json_content.startswith('"') and json_content.endswith('"')) or \
        (json_content.startswith("'") and json_content.endswith("'")):
            return json_content.strip('"').strip("'")

        # 3. Try to extract first {...} or [...] block with regex
        match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
        if match:
            candidate = match.group(1)
            return candidate
        
        # 4. If nothing works, just return the raw string
        return content
    
    except Exception as e:
        log.error(f"error parsing json content from LLM. {e}")