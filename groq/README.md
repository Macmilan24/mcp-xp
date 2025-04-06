```markdown
# Grok Chatbot with Galaxy Integration

This project integrates a chatbot (`grok_chatbot.py`) with a custom MCP server (`bioblend_server`) to fetch tools from a Galaxy instance (e.g., https://usegalaxy.eu/). The chatbot uses the Grok LLM to process user requests and interacts with Galaxy via the bioblend library.

## Project Overview

**Purpose:** Fetch Galaxy tools and their metadata using a chatbot interface.

**Components:**

-   `grok_chatbot.py`: The main script that runs the chatbot, interfacing with the LLM and MCP servers.
-   `bioblend_server/`: An MCP server that connects to Galaxy and fetches tools.
-   `utils/`: Helper functions for fetching and processing Galaxy tool data.

**Dependencies:** Python 3.8+, bioblend, mcp, httpx, python-dotenv, and other packages listed in `requirements.txt`.

## File Structure

The project is structured as follows:

```
/home/biniam/Desktop/Projects/mcp-xp/groq/
├── bioblend_server/
│   ├── __init__.py
│   ├── __main__.py
│   ├── galaxy_tools.py
│   ├── server.py
├── docs/
│   └── (documentation files)
├── grok_chatbot.py
├── .env
├── .env.example
├── config.py
├── requirements.txt
├── servers_config.json
├── README.md
├── test.db
├── .venv/
├── .gitignore
├── python-version
├── pyproject.toml
└── uv.lock
```

## Prerequisites

### Python Environment:

-   Ensure Python 3.8 or higher is installed.
-   Set up a virtual environment:

    ```bash
    cd /home/biniam/Desktop/Projects/mcp-xp/groq/
    python3 -m venv .venv
    source .venv/bin/activate
    ```

### Install Dependencies:

-   Install required packages:

    ```bash
    pip install bioblend mcp httpx python-dotenv
    ```

-   If using `requirements.txt`, run:

    ```bash
    pip install -r requirements.txt
    ```

## Setup

### 1. Configure Environment Variables

-   Create or edit the `.env` file in `/home/biniam/Desktop/Projects/mcp-xp/groq/bioblend_server/` with the following content:

    ```
    LLM_API_KEY=<your-grok-api-key>
    GALAXY_API_KEY=<your-galaxy-api-key>
    GALAXY_URL=[https://usegalaxy.eu/](https://usegalaxy.eu/)
    ```

    -   `LLM_API_KEY`: Your API key for the Grok LLM (provided by xAI).
    -   `GALAXY_API_KEY`: Your API key for the Galaxy instance (e.g., from https://usegalaxy.eu/).
    -   `GALAXY_URL`: The URL of the Galaxy instance (default: https://usegalaxy.eu/).

### 2. Configure Servers

-   Ensure `servers_config.json` in `/home/biniam/Desktop/Projects/mcp-xp/groq/` contains:

    ```json
    {
      "mcpServers": {
        "galaxyTools": {
          "command": "python3",
          "args": ["-m", "bioblend_server"],
          "env": {}
        }
      }
    }
    ```

    This configures the `bioblend_server` to run as an MCP server named `galaxyTools`.

## Running the Chatbot

1.  Navigate to the Project Directory:

    ```bash
    cd /home/biniam/Desktop/Projects/mcp-xp/groq/
    ```

2.  Activate the Virtual Environment (if not already active):

    ```bash
    source .venv/bin/activate
    ```

3.  Run the Chatbot:

    ```bash
    python3 grok_chatbot.py
    ```

    The chatbot will initialize, start the `bioblend_server`, and prompt for input with `You:`.

## Using the Chatbot

**Input:** At the `You:` prompt, type:

```
get me 5 galaxy tools
```

**Expected Output:**

The chatbot will fetch 5 tools from the Galaxy instance and display their metadata. The output will look like:

```
galaxy_tools: {'description': 'from your computer', 'edam_operations': ['operation_0224'], 'edam_topics': [], 'form_style': 'regular', 'hidden': '', 'id': 'upload1', 'is_workflow_compatible': False, 'labels': [], 'link': '/tool_runner?tool_id=upload1', 'min_width': -1, 'model_class': 'Tool', 'name': 'Upload File', 'panel_section_id': 'get_data', 'panel_section_name': 'Get Data', 'target': 'galaxy_main', 'version': '1.1.7', 'xrefs': []}
{'description': 'Trying to get open files out of SEEK', 'edam_operations': [], 'edam_topics': [], 'form_style': 'special', 'hidden': '', 'id': 'ds_seek_test', 'is_workflow_compatible': False, 'labels': ['beta'], 'link': '/tool_runner/data_source_redirect?tool_id=ds_seek_test', 'min_width': '800', 'model_class': 'DataSourceTool', 'name': 'SEEK test', 'panel_section_id': 'get_data', 'panel_section_name': 'Get Data', 'target': '_top', 'version': '0.0.1', 'xrefs': []}
{'description': 'table browser', 'edam_operations': ['operation_0224'], 'edam_topics': [], 'form_style': 'special', 'hidden': '', 'id': 'ucsc_table_direct1', 'is_workflow_compatible': False, 'labels': [], 'link': '/tool_runner/data_source_redirect?tool_id=ucsc_table_direct1', 'min_width': '800', 'model_class': 'DataSourceTool', 'name': 'UCSC Main', 'panel_section_id': 'get_data', 'panel_section_name': 'Get Data', 'target': '_top', 'version': '1.0.0', 'xrefs': []}
{'description': 'table browser', 'edam_operations': ['operation_0224'], 'edam_topics': [], 'form_style': 'special', 'hidden': '', 'id': 'ucsc_table_direct_archaea1', 'is_workflow_compatible': False, 'labels': [], 'link': '/tool_runner/data_source_redirect?tool_id=ucsc_table_direct_archaea1', 'min_width': '800', 'model_class': 'DataSourceTool', 'name': 'UCSC Archaea', 'panel_section_id': 'get_data', 'panel_section_name': 'Get Data', 'target': '_top', 'version': '1.0.0', 'xrefs': []}
{'description': 'import data from the NCBI Datasets Genomes page', 'edam_operations': ['operation_0224'], 'edam_topics': [], 'form_style': 'special', 'hidden': '', 'id': 'ncbi_datasets_source', 'is_workflow_compatible': False, 'labels': [], 'link': '/tool_runner/data_source_redirect?tool_id=ncbi_datasets_source', 'min_width': -1, 'model_class': 'DataSourceTool', 'name': 'NCBI Datasets Genomes', 'panel_section_id': 'get_data', 'panel_section_name': 'Get Data', 'target': '_top', 'version': '13.14.0', 'xrefs': []}
{
    "tool": "galaxy_tools",
    "arguments": {
        "number_of_tools": "5"
    }
}
```

**Explanation:**

-   The first part lists 5 Galaxy tools with their metadata (e.g., id, name, description).
-   The JSON at the end (`{"tool": "galaxy_tools", "arguments": {"number_of_tools": "5"}}`) is the LLM’s response, indicating it called the `galaxy_tools` tool with the argument `number_of_tools=5`.

## LLM Configuration

The chatbot uses the Grok LLM with the following settings:

-   **Model:** `llama-3.2-90b-vision-preview`
-   **API Key:** Set via `LLM_API_KEY` in `.env`.
-   **Endpoint:** `https://api.groq.com/openai/v1/chat/completions`
-   **Parameters:**
    -   `temperature`: 0.7
    -   `max_tokens`: 4096
    -   `top_p`: 1
    -   `stream`: False
    -   `stop`: None

## Troubleshooting

### Import Errors:

-   Ensure `utils/` contains all required files (`fetch_tool_source_code.py`, etc.) and `__init__.py`.
-   If running `galaxy_tools.py` standalone fails, use:

    ```bash
    cd /home/biniam/Desktop/Projects/mcp-xp/groq/
    python3 -m bioblend_server.galaxy_tools
    ```

### Galaxy Connection Issues:

-   Verify `GALAXY_URL` and `GALAXY_API_KEY` are correct.
-   Check network connectivity to https://usegalaxy.eu/.

### LLM Errors:

-   Ensure `LLM_API_KEY` is valid and you have access to the `llama-3.2-90b-vision-preview` model.
-   Check `grok_chatbot.log` for detailed error messages.

### No Output:

-   If the `You:` prompt doesn’t appear, check `grok_chatbot.log` for errors during server initialization.

## Additional Notes

-   **Logging:** Logs are written to `grok_chatbot.log` for debugging.
-   **Extending Functionality:** Add more tools to `bioblend_server/server.py` by extending the `list_tools()` function.
-   **Testing Standalone:**
    -   Test `galaxy_tools.py` alone:

        ```bash
        cd /home/biniam/Desktop/Projects/mcp-xp/groq/
        python3 -m bioblend_server.galaxy_tools
        ```
```
