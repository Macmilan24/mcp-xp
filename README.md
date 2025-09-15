# Chatbot with Galaxy Integration

This project integrates a chatbot with a custom MCP server (`bioblend_server`) to fetch tools from a Galaxy instance (e.g., https://usegalaxy.eu/). The chatbot uses various LLM providers (Azure, Groq) to process user requests and interacts with Galaxy via the bioblend library.

## Project Overview

**Purpose:** Fetch Galaxy tools and their metadata using a chatbot interface with support for multiple LLM providers.

**Components:**

- `main.py`: The FastAPI application that provides REST endpoints for interacting with the chatbot.
- `AI/chatbot.py`: The core chatbot implementation that interfaces with LLM providers and MCP servers.
- `AI/llm_Config/`: Configuration and implementation for different LLM providers.
- `AI/bioblend_server/`: An MCP server that connects to Galaxy and fetches tools.
- `utils/`: Helper functions for fetching and processing Galaxy tool data.

**Dependencies:** Python 3.8+, FastAPI, uvicorn, bioblend, mcp, httpx, python-dotenv, and other packages listed in `requirements.txt`.

## Prerequisites

### Python Environment:

- Ensure Python 3.8 or higher is installed.
- Set up a virtual environment:

  ```bash
  mcp-xp/
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Install Dependencies:

- Install required packages:

  ```bash
  pip install bioblend mcp httpx python-dotenv
  ```

- If using `requirements.txt`, run:

  ```bash
  pip install -r requirements.txt
  ```

## Setup

### 1. Configure Environment Variables

- Create or edit the `.env` file in the project root with the following content:

  ```
  GROQ_API_KEY=<your-groq-api-key>
  AZURE_API_KEY=<your-azure-api-key>
  GALAXY_API_KEY=<your-galaxy-api-key>
  GALAXY_URL=https://usegalaxy.eu/
  ```

  - `GROQ_API_KEY`: Your API key for the Groq LLM provider.
  - `AZURE_API_KEY`: Your API key for the Azure OpenAI service (for GPT-4o).
  - `GALAXY_API_KEY`: Your API key for the Galaxy instance (e.g., from https://usegalaxy.eu/).
  - `GALAXY_URL`: The URL of the Galaxy instance (default: https://usegalaxy.eu/).

### 2. Configure Servers

- Ensure `app/AI/servers_config.json` contains:

  ```json
  {
    "mcpServers": {
      "galaxyTools": {
        "command": "python3",
        "args": ["-m", "app.AI.bioblend_server"],
        "env": {}
      }
    }
  }
  ```

  This configures the `bioblend_server` to run as an MCP server named `galaxyTools`.

### 3. Configure LLM Providers

- The LLM providers are configured in `app/AI/llm_Config/llm_config.json`. This file defines the available LLM providers and their settings:

  ```json
  {
    "providers": {
      "azure": {
        "api_key": "your_azure_api_key",
        "base_url": "https://models.inference.ai.azure.com",
        "model": "gpt-4o",
        "provider": "azure",
        "temperature": 0.7,
        "max_tokens": 150,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": null,
        "stream": true,
        "stream_options": {
          "include_usage": true
        }
      },
      "groq": {
        "api_key": "your_groq_api_key",
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "provider": "groq",
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 1,
        "stream": true,
        "stop": null
      }
    },
    "default_provider": "azure",
    "cache": {
      "enabled": true,
      "cache_size": 100,
      "cache_expiry": 3600
    }
  }
  ```

## Running the Application

1.  Navigate to the Project Directory:

    ```bash
    cd /path/to/project
    ```

2.  Activate the Virtual Environment (if not already active):

    ```bash
    source .venv/bin/activate
    ```

3.  Run the FastAPI application:

    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```

    The application will start and be available at http://localhost:8000.

## Using the API Endpoints

### 1. Initialize the Chat Session

Before sending messages, you need to initialize the chat session:

```bash
curl -X POST http://localhost:8000/initiate_chat
```
````

**Expected Response:**

```json
{
  "message": "Chat session initiated"
}
```

### 2. Send a Message to a Specific Model

After initializing the chat session, you can send messages to a specific model using the model_id parameter:

```bash
curl -X POST "http://localhost:8000/send_message?model_id=azure" \
  -H "Content-Type: application/json" \
  -d '{"message": "what tools do you have"}'
```

**Expected Response:**

```json
{
  "response": "I can access tools from the Galaxy platform, which is a bioinformatics workflow management system. I can either fetch a list of tools available in the Galaxy instance or provide details about a specific tool using its ID. Let me know if you'd like me to retrieve a list of tools or details about a specific one!"
}
```

### Example Interactions

**Fetching Galaxy Tools:**

```bash
curl -X POST "http://localhost:8000/send_message?model_id=azure" \
  -H "Content-Type: application/json" \
  -d '{"message": "get me 5 galaxy tools"}'
```

**Fetching a Specific Tool by ID:**

```bash
curl -X POST "http://localhost:8000/send_message?model_id=azure" \
  -H "Content-Type: application/json" \
  -d '{"message": "get me the tool with id upload1"}'
```

**Listing Available Tools:**

```bash
curl -X POST "http://localhost:8000/send_message?model_id=azure" \
  -H "Content-Type: application/json" \
  -d '{"message": "what tools do you have"}'
```

## LLM Configuration System

The application uses a flexible configuration system to support multiple LLM providers:

### LLM Configuration Files

1. **llm_config.json**: Located at `app/AI/llm_Config/llm_config.json`, this file defines the available LLM providers and their settings. Each provider has its own configuration section with parameters like model name, API endpoint, temperature, etc.

2. **llmConfig.py**: Located at `app/AI/llm_Config/llmConfig.py`, this file implements the provider-specific classes:
   - `LLMModelConfig`: Base class for all LLM configurations
   - `GROQConfig` and `AZUREConfig`: Provider-specific configuration classes
   - `LLMProvider`: Abstract base class for all LLM providers
   - `GroqProvider` and `AzureProvider`: Concrete implementations for each provider

### Application Flow

1. When the application starts, it loads the configuration from `llm_config.json`
2. The `/initiate_chat` endpoint initializes a chat session and loads all configured providers
3. When a message is sent to `/send_message?model_id=azure`, the application:
   - Checks if the chat session is initialized
   - Retrieves the specified provider (e.g., Azure)
   - Sends the message to the provider's API
   - Returns the response

This architecture allows for easy addition of new LLM providers by:

1. Adding a new provider configuration to `llm_config.json`
2. Implementing provider-specific classes in `llmConfig.py`

## Troubleshooting

### Import Errors:

- Ensure `utils/` contains all required files (`fetch_tool_source_code.py`, etc.) and `__init__.py`.
- If running `galaxy_tools.py` standalone fails, use:

  ```bash
  cd /mcp-xp/groq/
  python3 -m bioblend_server.galaxy_tools
  ```

### Galaxy Connection Issues:

- Verify `GALAXY_URL` and `GALAXY_API_KEY` are correct.
- Check network connectivity to https://usegalaxy.eu/.

### LLM Errors:

- Ensure `GROQ_API_KEY` and `AZURE_API_KEY` are valid and you have access to the configured models.
- Check application logs for detailed error messages.

### No Output:

- If the API doesn't respond, check that you've initialized the chat session first.
- Verify that the FastAPI application is running correctly.

## Additional Notes

- **Logging:** Logs are written to the application logs for debugging.
- **Extending Functionality:** Add more tools to `bioblend_server/server.py` by extending the `list_tools()` function.
- **Testing Standalone:**

  - Test `galaxy_tools.py` alone:

    ```bash
    cd /path/to/project
    python3 -m app.AI.bioblend_server.galaxy_tools
    ```

```

