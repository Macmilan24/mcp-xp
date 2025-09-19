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