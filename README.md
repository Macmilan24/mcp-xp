# Galaxy Interaction Agent

This project provides a robust FastAPI application that serves as a bridge between a user and a Galaxy instance. It features a sophisticated agent powered by multiple Large Language Models (LLMs) to interact with Galaxy, execute tools and workflows, and manage data. The application uses a custom Model Context Protocol (MCP) server for seamless communication with the Galaxy platform via the `bioblend` library, and implements Retrieval-Augmented Generation (RAG) for intelligent tool discovery.

## Table of Contents

- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
  - [Authentication](#authentication)
  - [Agent Endpoints](#agent-endpoints)
  - [Histories & Data Endpoints](#histories--data-endpoints)
  - [Workflows Endpoints](#workflows-endpoints)
  - [Tools Endpoints](#tools-endpoints)
  - [Invocations Endpoints](#invocations-endpoints)
- [Troubleshooting](#troubleshooting)

## Key Features

*   **Dynamic Galaxy Interaction:** Execute tools and workflows, manage histories, and handle datasets on a Galaxy instance programmatically.
*   **Intelligent Agent (RAG-powered):** A conversational agent that uses Semantic Search (via **Qdrant**) to intelligently find and recommend the right Galaxy tools and workflows based on user queries.
*   **Multi-LLM Support:** Easily configure and switch between different LLM providers like OpenAI, Google Gemini, Azure, and Groq.
*   **Secure Authentication:** User authentication is handled via JWT, with Galaxy API keys securely encrypted.
*   **Asynchronous Operations:** Built with FastAPI and `asyncio` for high-performance, non-blocking I/O.
*   **Caching with Redis:** Caching of invocations, workflows, and rate limiting to improve performance.
*   **Real-time Updates:** WebSocket support for real-time updates on long-running tasks like workflow and tool execution.
*   **Dynamic Form Generation:** Automatically generates HTML forms for Galaxy tools and workflows.

## Project Structure

The project is organized into the following key directories and files:

*   **`main.py`**: The entry point of the FastAPI application. It defines the API endpoints, middleware, and application lifecycle.
*   **`app/AI/`**: Contains the core logic for the AI agent, including:
    *   `chatbot.py`: Manages the chat sessions and orchestration between the LLM and the MCP server.
    *   `llm_config/`: Configuration files and classes for the different LLM providers.
    *   `bioblend_server/`: The MCP server that exposes Galaxy tools to the LLM.
*   **`app/api/`**: Defines the API endpoints, middleware, and data schemas.
*   **`app/orchestration/`**: Handles the caching (via Redis) and background tasks for invocations.
*   **`app/utils/`**: Utility functions, such as importing published workflows.

## Prerequisites

Before you begin, ensure you have the following installed and running:

*   **Python 3.8+**
*   **Redis** (for caching and rate limiting)
*   **Qdrant** (for vector storage and semantic search)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/rejuve-bio/mcp-xp.git
    cd mcp-xp
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  **Create a `.env` file** in the root of the project and add the following environment variables.

    ```env
    # Galaxy Configuration
    GALAXY_URL=https://usegalaxy.eu/
    GALAXY_API_KEY=<your-galaxy-admin-api-key>

    # LLM API Keys (Fill in the ones you intend to use)
    OPENAI_API_KEY=<your-openai-api-key>
    GEMINI_API_KEY=<your-gemini-api-key>
    AZURE_API_KEY=<your-azure-api-key>
    GROQ_API_KEY=<your-groq-api-key>

    # Infrastructure & Security
    SECRET_KEY=<a-secure-random-string-for-encryption>
    REDIS_PORT=6379
    QDRANT_CLIENT=http://localhost:6333 
    ```
    *(Note: `QDRANT_CLIENT` should point to your Qdrant instance URL).*

2.  **Configure the MCP Server** in `app/AI/servers_config.json`. This file tells the application how to communicate with the `bioblend_server`.

    ```json
    {
      "mcpServers": {
        "galaxyTools": {
          "url": "http://localhost:8897",
          "headers": {
            "Authorization": "Bearer <your-jwt-token>"
          }
        }
      }
    }
    ```
    *Note: The `Authorization` header will be dynamically updated by the application during runtime.*

3.  **Configure LLM Providers** in `app/AI/llm_config/llm_config.json`. Here, you can define the settings for each LLM provider.

    ```json
    {
        "providers": {
            "openai": {
                "model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
                "provider": "openai",
                ...
            },
            "gemini": {
                "model": "gemini-2.0-flash",
                "embedding_model": "embedding-001",
                "provider": "gemini",
                ...
            }
        },
        "default_provider": "openai"
    }
    ```

## Running the Application

1.  **Start the MCP Server:**
    Open a terminal and run the following command from the project root:
    ```bash
    python3 -m app.bioblend_server
    ```

2.  **Start the FastAPI Application:**
    In a separate terminal, run the following command:
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
    The application will be available at `http://localhost:8000`.

## API Endpoints

The API is protected by JWT authentication. You first need to register a user to get a token.

### Authentication

**`POST /register-user`**

Registers a new user on the Galaxy instance (or fetches an existing one) and returns an encrypted API token.

*   **Query Parameters:**
    *   `email` (str): The user's email address.
    *   `password` (str): The user's password.

*   **Example Response:**
    ```json
    {
      "username": "user",
      "api_token": "<your-encrypted-api-token>"
    }
    ```

### Agent Endpoints

*You must include the `api_token` in the `Authorization` header as a Bearer token for all subsequent requests.*

**`POST /send_message`**

Sends a message to the Galaxy Agent. The agent will use RAG (via Qdrant) to find relevant tools if necessary.

*   **Example Request:**
    ```bash
    curl -X POST http://localhost:8000/send_message \
    -H "Authorization: Bearer <your-api-token>" \
    -H "Content-Type: application/json" \
    -d '{"message": "find me a tool to align fastq files"}'
    ```

### Histories & Data Endpoints

*   **`GET /api/histories/`**: List all Galaxy histories.
*   **`POST /api/histories/create`**: Create a new Galaxy history.
*   **`POST /api/histories/{history_id}/upload-file`**: Upload a file to a specific history.
*   **`POST /api/histories/{history_id}/upload-collection`**: Upload files and create a dataset collection (list, paired, or list:paired).

### Workflows Endpoints

*   **`GET /api/workflows/`**: List all available workflows.
*   **`POST /api/workflows/upload-workflow`**: Upload a workflow from a `.ga` file.
*   **`GET /api/workflows/{workflow_id}/form`**: Get a dynamic HTML form for a workflow.
*   **`POST /api/workflows/{workflow_id}/histories/{history_id}/execute`**: Execute a workflow.

### Tools Endpoints

*   **`GET /api/tools/{tool_id}/form`**: Get a dynamic HTML form for a tool.
*   **`POST /api/tools/{tool_id}/histories/{history_id}/execute`**: Execute a tool.

### Invocations Endpoints

*   **`GET /api/invocation/`**: List all workflow invocations.
*   **`GET /api/invocation/{invocation_id}/result`**: Get the result of a specific invocation.
*   **`DELETE /api/invocation/DELETE`**: Delete one or more invocations.

## Troubleshooting

*   **Authentication Errors:** Ensure you have a valid `api_token` and that it is correctly included in the `Authorization` header as a Bearer token.
*   **Connection Issues (Galaxy):** Verify that the `GALAXY_URL` is correct and reachable.
*   **Connection Issues (Redis/Qdrant):** Ensure your Redis and Qdrant instances are running and the `REDIS_PORT` and `QDRANT_CLIENT` env variables are correct.
*   **LLM Errors:** Check that your LLM API keys are correct and that you have access to the specified models in `llm_config.json`.