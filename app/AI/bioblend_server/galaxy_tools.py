# app/AI/bioblend_server/galaxy_tools.py
print("bioblend_server galaxytools.py")
from app.config import GALAXY_URL, GALAXY_API_KEY
from bioblend import galaxy
from bioblend.galaxy.client import Client # Import base client for error handling
from bioblend.galaxy import histories, workflows, datasets, users, libraries # Import specific clients
import os
import pprint
import logging # Import logging

# --- Configure Logging ---
# Use the same logger name as in server.py or a specific one for tools
tool_logger = logging.getLogger("bioblend_server.galaxy_tools")
# Set a basic handler if not configured elsewhere (useful for standalone testing)
if not tool_logger.handlers:
    logging.basicConfig(level=logging.INFO)
    tool_logger.addHandler(logging.StreamHandler())
    tool_logger.setLevel(logging.INFO)
# --- End Logging Configuration ---

def setup_instance():
    """Sets up and returns a BioBlend GalaxyInstance."""
    tool_logger.debug(f"Setting up GalaxyInstance for URL: {GALAXY_URL}")
    try:
        gi = galaxy.GalaxyInstance(url=GALAXY_URL, key=GALAXY_API_KEY)
        # Test connection by getting config or whoami
        gi.config.get_config()
        tool_logger.debug(f"GalaxyInstance setup successful. User: {gi.users.get_current_user()['email']}")
        return gi
    except Exception as e:
        tool_logger.error(f"Failed to setup GalaxyInstance: {e}", exc_info=True)
        raise # Re-raise the exception after logging

# --- Existing Tools (Modified for consistent logging) ---

def get_tools(number_of_tools=10):
    """
    Get a list of tools from the galaxy instance.
    """
    tool_logger.info(f"Executing get_tools with number_of_tools={number_of_tools}")
    try:
        gi = setup_instance()
        tool_client = galaxy.tools.ToolClient(gi)
        # Using get_tools which returns a list of dicts directly
        all_tools = tool_client.get_tools()
        # Ensure number_of_tools is an int
        num = int(number_of_tools)
        selected_tools = all_tools[:num]
        tool_logger.info(f"Found {len(all_tools)} tools, returning details for {len(selected_tools)}.")
        # Format output
        formatted_tools = [pprint.pformat(tool) for tool in selected_tools]
        return "\n---\n".join(formatted_tools)
    except Exception as e:
        tool_logger.error(f"Error in get_tools: {e}", exc_info=True)
        return f"Error fetching tools: {e}"


def get_tool(tool_id):
    """
    Get a specific tool by its ID from the galaxy instance.
    """
    tool_logger.info(f"Executing get_tool with tool_id={tool_id}")
    try:
        gi = setup_instance()
        tool_client = galaxy.tools.ToolClient(gi)
        tool_details = tool_client.show_tool(tool_id=tool_id)
        tool_logger.info(f"Successfully retrieved tool details for ID: {tool_id}")
        return pprint.pformat(tool_details)
    except Client.NotFound:
         tool_logger.warning(f"Tool not found with ID: {tool_id}")
         return f"Error: Tool with ID '{tool_id}' not found."
    except Exception as e:
        tool_logger.error(f"Error in get_tool for ID {tool_id}: {e}", exc_info=True)
        return f"Error fetching tool {tool_id}: {e}"

# --- New Tools ---

# -- History Tools --
def list_histories():
    """Lists all accessible histories for the user."""
    tool_logger.info("Executing list_histories")
    try:
        gi = setup_instance()
        history_client = histories.HistoryClient(gi)
        history_list = history_client.get_histories()
        tool_logger.info(f"Found {len(history_list)} histories.")
        return pprint.pformat(history_list)
    except Exception as e:
        tool_logger.error(f"Error in list_histories: {e}", exc_info=True)
        return f"Error listing histories: {e}"

def get_history_details(history_id):
    """Gets detailed information about a specific history."""
    tool_logger.info(f"Executing get_history_details for history_id={history_id}")
    try:
        gi = setup_instance()
        history_client = histories.HistoryClient(gi)
        details = history_client.show_history(history_id=history_id, contents=False) # Set contents=False for brevity, or True for datasets
        tool_logger.info(f"Successfully retrieved details for history ID: {history_id}")
        return pprint.pformat(details)
    except Client.NotFound:
         tool_logger.warning(f"History not found with ID: {history_id}")
         return f"Error: History with ID '{history_id}' not found."
    except Exception as e:
        tool_logger.error(f"Error in get_history_details for ID {history_id}: {e}", exc_info=True)
        return f"Error fetching history details {history_id}: {e}"

def create_history(name):
    """Creates a new history with the given name."""
    tool_logger.info(f"Executing create_history with name='{name}'")
    try:
        gi = setup_instance()
        history_client = histories.HistoryClient(gi)
        new_history = history_client.create_history(name=name)
        tool_logger.info(f"Successfully created history '{name}' with ID: {new_history.get('id')}")
        return pprint.pformat(new_history)
    except Exception as e:
        tool_logger.error(f"Error in create_history for name '{name}': {e}", exc_info=True)
        return f"Error creating history '{name}': {e}"

def delete_history(history_id, purge=False):
    """Deletes a history by its ID. Set purge=True to permanently delete."""
    tool_logger.info(f"Executing delete_history for history_id={history_id}, purge={purge}")
    try:
        gi = setup_instance()
        history_client = histories.HistoryClient(gi)
        # Note: delete_history returns the history dict *before* deletion
        result = history_client.delete_history(history_id=history_id, purge=bool(purge))
        tool_logger.info(f"Successfully initiated deletion for history ID: {history_id}. Purge={purge}. Result: {result}")
        # Provide a clearer success message as the returned dict might be confusing
        return f"History deletion initiated for ID {history_id}. Purge={purge}. Result: {pprint.pformat(result)}"
    except Client.NotFound:
         tool_logger.warning(f"History not found for deletion with ID: {history_id}")
         return f"Error: History with ID '{history_id}' not found for deletion."
    except Exception as e:
        tool_logger.error(f"Error in delete_history for ID {history_id}: {e}", exc_info=True)
        return f"Error deleting history {history_id}: {e}"

# -- Workflow Tools --
def list_workflows():
    """Lists all accessible workflows for the user."""
    tool_logger.info("Executing list_workflows")
    try:
        gi = setup_instance()
        workflow_client = workflows.WorkflowClient(gi)
        workflow_list = workflow_client.get_workflows()
        tool_logger.info(f"Found {len(workflow_list)} workflows.")
        return pprint.pformat(workflow_list)
    except Exception as e:
        tool_logger.error(f"Error in list_workflows: {e}", exc_info=True)
        return f"Error listing workflows: {e}"

def get_workflow_details(workflow_id):
    """Gets detailed information about a specific workflow."""
    tool_logger.info(f"Executing get_workflow_details for workflow_id={workflow_id}")
    try:
        gi = setup_instance()
        workflow_client = workflows.WorkflowClient(gi)
        details = workflow_client.show_workflow(workflow_id=workflow_id)
        tool_logger.info(f"Successfully retrieved details for workflow ID: {workflow_id}")
        return pprint.pformat(details)
    except Client.NotFound:
         tool_logger.warning(f"Workflow not found with ID: {workflow_id}")
         return f"Error: Workflow with ID '{workflow_id}' not found."
    except Exception as e:
        tool_logger.error(f"Error in get_workflow_details for ID {workflow_id}: {e}", exc_info=True)
        return f"Error fetching workflow details {workflow_id}: {e}"

# -- Dataset Tools --
def list_datasets_in_history(history_id):
    """Lists all datasets within a specific history."""
    tool_logger.info(f"Executing list_datasets_in_history for history_id={history_id}")
    try:
        gi = setup_instance()
        dataset_client = datasets.DatasetClient(gi)
        # We use show_history(contents=True) as there isn't a direct list_datasets call
        history_client = histories.HistoryClient(gi)
        dataset_list = history_client.show_history(history_id=history_id, contents=True)
        tool_logger.info(f"Found {len(dataset_list)} datasets in history ID: {history_id}.")
        return pprint.pformat(dataset_list)
    except Client.NotFound:
         tool_logger.warning(f"History not found when listing datasets with ID: {history_id}")
         return f"Error: History with ID '{history_id}' not found when trying to list datasets."
    except Exception as e:
        tool_logger.error(f"Error in list_datasets_in_history for history {history_id}: {e}", exc_info=True)
        return f"Error listing datasets for history {history_id}: {e}"

def get_dataset_details(dataset_id):
    """Gets detailed information about a specific dataset."""
    tool_logger.info(f"Executing get_dataset_details for dataset_id={dataset_id}")
    try:
        gi = setup_instance()
        dataset_client = datasets.DatasetClient(gi)
        details = dataset_client.show_dataset(dataset_id=dataset_id)
        tool_logger.info(f"Successfully retrieved details for dataset ID: {dataset_id}")
        return pprint.pformat(details)
    except Client.NotFound:
         tool_logger.warning(f"Dataset not found with ID: {dataset_id}")
         return f"Error: Dataset with ID '{dataset_id}' not found."
    except Exception as e:
        tool_logger.error(f"Error in get_dataset_details for ID {dataset_id}: {e}", exc_info=True)
        return f"Error fetching dataset details {dataset_id}: {e}"

# -- User Tools --
def list_users():
    """Lists all users (requires Galaxy admin privileges)."""
    tool_logger.info("Executing list_users")
    try:
        gi = setup_instance()
        user_client = users.UserClient(gi)
        user_list = user_client.get_users()
        tool_logger.info(f"Found {len(user_list)} users.")
        return pprint.pformat(user_list)
    except Exception as e:
        # Catch potential permission errors
        tool_logger.error(f"Error in list_users: {e}", exc_info=True)
        if "403" in str(e):
             return "Error listing users: This action likely requires Galaxy administrator privileges."
        return f"Error listing users: {e}"

def get_user_details(user_id):
    """Gets detailed information about a specific user (requires Galaxy admin privileges)."""
    tool_logger.info(f"Executing get_user_details for user_id={user_id}")
    try:
        gi = setup_instance()
        user_client = users.UserClient(gi)
        details = user_client.show_user(user_id=user_id)
        tool_logger.info(f"Successfully retrieved details for user ID: {user_id}")
        return pprint.pformat(details)
    except Client.NotFound:
         tool_logger.warning(f"User not found with ID: {user_id}")
         return f"Error: User with ID '{user_id}' not found."
    except Exception as e:
        tool_logger.error(f"Error in get_user_details for ID {user_id}: {e}", exc_info=True)
        if "403" in str(e):
             return f"Error fetching user details for {user_id}: This action likely requires Galaxy administrator privileges."
        return f"Error fetching user details {user_id}: {e}"

# -- Library Tools --
def list_libraries():
    """Lists all accessible data libraries."""
    tool_logger.info("Executing list_libraries")
    try:
        gi = setup_instance()
        library_client = libraries.LibraryClient(gi)
        library_list = library_client.get_libraries()
        tool_logger.info(f"Found {len(library_list)} libraries.")
        return pprint.pformat(library_list)
    except Exception as e:
        tool_logger.error(f"Error in list_libraries: {e}", exc_info=True)
        return f"Error listing libraries: {e}"

def get_library_details(library_id):
    """Gets detailed information about a specific data library."""
    tool_logger.info(f"Executing get_library_details for library_id={library_id}")
    try:
        gi = setup_instance()
        library_client = libraries.LibraryClient(gi)
        details = library_client.show_library(library_id=library_id, contents=False) # Set contents=True to see folders/datasets
        tool_logger.info(f"Successfully retrieved details for library ID: {library_id}")
        return pprint.pformat(details)
    except Client.NotFound:
         tool_logger.warning(f"Library not found with ID: {library_id}")
         return f"Error: Library with ID '{library_id}' not found."
    except Exception as e:
        tool_logger.error(f"Error in get_library_details for ID {library_id}: {e}", exc_info=True)
        return f"Error fetching library details {library_id}: {e}"


# Example of running one function directly for testing (optional)
if __name__ == "__main__":
    print("Running standalone test...")
    try:
        # test_tools = get_tools(3)
        # print("--- Tools ---")
        # print(test_tools)

        test_histories = list_histories()
        print("\n--- Histories ---")
        print(test_histories)

        # Add more test calls here if needed
        # print("\n--- Creating History ---")
        # new_hist = create_history("Test History via Script")
        # print(new_hist)
        # history_id_to_delete = new_hist.get('id') # Get ID from the created history
        # if history_id_to_delete:
        #     print(f"\n--- Deleting History {history_id_to_delete} ---")
        #     delete_result = delete_history(history_id_to_delete)
        #     print(delete_result)

    except Exception as e:
        print(f"Error during standalone test: {e}")