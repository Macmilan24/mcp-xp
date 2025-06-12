import asyncio
import json
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any
import httpx
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from app.AI.llm_Config.llmConfig import GroqProvider, AzureProvider,GeminiProvider, GROQConfig, AZUREConfig, GEMINIConfig
from datetime import datetime



class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.api_key = os.getenv("LLM_API_KEY")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_server_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, "r") as f:
            return json.load(f)

    @staticmethod
    def load_llm_config(file_path: str) -> dict[str, Any]:
        """Load LLM configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing LLM configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, "r") as f:
            return json.load(f)



class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]}
            if self.config.get("env")
            else None,
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
            # logging.info(f"Server {self.name} initialized successfully.")
        except Exception as e:
            # logging.error(f"Error initializing server {self.name}: {e}")
            print(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> list[Any]:
        if not self.session:
            # logging.error(f"Server {self.name} not initialized")
            raise RuntimeError(f"Server {self.name} not initialized")

        # logging.debug("getting tools")
        tools_response = await self.session.list_tools()
        if not tools_response:
            # logging.warning(f"No tools found for server {self.name}")
            return []
        tools = []
        # logging.info(f"Raw tools response from {self.name}: {tools_response}")

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    # logging.info(f"Server: {self.name}, Tool: {tool.name}, Description: {tool.description}")
                    tools.append(Tool(tool.name, tool.description, tool.inputSchema))
        print("tools fetched")
        print(len(tools))
        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] = {},
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                # logging.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments)

                return result

            except Exception as e:
                attempt += 1
                # logging.warning(
                #     f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                # )
                if attempt < retries:
                    # logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    # logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                # logging.error(f"Error during cleanup of server {self.name}: {e}")
                pass


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
        self, name: str, description: str, input_schema: dict[str, Any] = {}
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = (
                    f"- {param_name}: {param_info.get('description', 'No description')}"
                )
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
                    Tool: {self.name}
                    Description: {self.description}
                    Arguments:
                    {chr(10).join(args_desc)}
                """


class LLMClient:
    def __init__(self, llm_providers: dict):
        self.providers = llm_providers  # Creates an instance of the providers



class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], llm_client: LLMClient, user_ip: str = None) -> None:
        self.servers: list[Server] = servers
        self.llm_client: LLMClient = llm_client
        self.memory = []
        self.system_message = None
        self.messages = None
        self.user_ip = user_ip
        self.log_filename = f"chat_session_{self.user_ip}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self._initialize_log_file()

    @classmethod
    async def create(cls, servers: list[Server], llm_client: LLMClient, user_ip: str = None) -> "ChatSession":
        self = cls(servers, llm_client, user_ip)
        
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                except Exception as e:
                    print(f"Failed to initialize server {server.name}: {e}")
                    await self.cleanup_servers()
                    raise RuntimeError(f"Server initialization failed: {e}") from e

            # Gather tools
            all_tools = []
            try:
                for server in self.servers:
                    tools = await server.list_tools()
                    print(f"Type of tools: {type(tools)}")
                    print(type(tools), type(tools[0]), tools[0])
                    all_tools.extend(tools)
                tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])
            except Exception as e:
                print(f"Error listing tools: {e}")
                raise RuntimeError(f"Tool listing failed: {e}") from e

            # Compose system message
            self.system_message = (
                "You are a helpful assistant with access to these tools:\n\n"
                f"{tools_description}\n"
                "Choose the appropriate tool based on the user's question. "
                "If no tool is needed, reply directly.\n\n"
                "IMPORTANT: When you need to use a tool, you must ONLY respond with "
                "the exact JSON object format below, nothing else:\n"
                "{\n"
                '    "tool": "tool name",\n'
                '    "arguments": {\n'
                '        "argument name": "value"\n'
                "    }\n"
                "}\n\n"
                "After receiving a tool's response:\n"
                "1. Transform the raw data into a natural, conversational response\n"
                "2. Keep responses concise but informative\n"
                "3. Focus on the most relevant information\n"
                "4. Use appropriate context from the user's question\n"
                "5. Avoid simply repeating the raw data\n\n"
                "Please use only the tools that are explicitly defined above."
            )

            self.messages = [{"role": "user", "content": self.system_message}]
            return self

        except Exception:
            await self.cleanup_servers()
            raise

    def _initialize_log_file(self):
        """Create the log file when the session starts."""
        try:
            with open(self.log_filename, "w") as log_file:
                log_data = {
                    "user_ip": self.user_ip,
                    "session_start": datetime.now().isoformat(),
                    "messages": []
                }
                json.dump(log_data, log_file)
                log_file.write("\n")  # Ensure each session is on a new line
        except Exception as e:
            print(f"Error initializing log file: {e}")

    async def cleanup_servers(self) -> None:
        """Clean up all server resources safely."""
        if not self.servers:
            return

        for server in self.servers:
            try:
                await server.cleanup()
            except Exception as e:
                # Silently handle cleanup errors to ensure all servers get cleanup attempts
                print(f"Error cleaning up server {server.name}: {e}")
                pass

    async def process_llm_response(self, llm_response: str) -> str:
        """Process the LLM response and execute tools if needed.

        Args:
            llm_response: The response from the LLM.

        Returns:
            The result of tool execution or the original response.
        """
        import json

        try:
            tool_call = json.loads(llm_response)
            if "tool" in tool_call and "arguments" in tool_call:
                # logging.info(f"Executing tool: {tool_call['tool']}")
                # logging.info(f"With arguments: {tool_call['arguments']}")

                for server in self.servers:
                    tools = await server.list_tools()
                    if any(tool.name == tool_call["tool"] for tool in tools):
                        try:
                            result = await server.execute_tool(
                                tool_call["tool"], tool_call["arguments"]
                            )

                            if isinstance(result, dict) and "progress" in result:
                                progress = result["progress"]
                                total = result["total"]
                                percentage = (progress / total) * 100
                                # logging.info(
                                #     f"Progress: {progress}/{total} "
                                #     f"({percentage:.1f}%)"
                                # )

                            return f"Tool execution result: {result}"
                        except Exception as e:
                            error_msg = f"Error executing tool: {str(e)}"
                            # logging.error(error_msg)
                            return error_msg

                return f"No server found with tool: {tool_call['tool']}"
            return llm_response
        except json.JSONDecodeError:
            return llm_response

    async def respond(self, model_id: str, user_input: str) -> str:
        """Handle a user input and return the assistant's response."""
        # Process user input
        try:
            self.messages.extend(self.memory)
            self.messages.append({"role": "user", "content": user_input})
            self.memory.append({"role": "user", "content": user_input})
            providers = self.llm_client.providers

            # Validate model_id
            if model_id not in providers:
                raise ValueError(f"Invalid model_id: {model_id}. Available providers: {list(providers.keys())}")

            # Get LLM response
            llm_response = await providers[model_id].get_response(self.messages)

            # Process potential tool calls
            result = await self.process_llm_response(llm_response)

            if result != llm_response:
                # Tool was used; get a final response
                self.messages.append({"role": "model", "content": llm_response})
                self.messages.append({"role": "model", "content": result})
                final_response = await providers[model_id].get_response(self.messages)
                self.messages.append({"role": "model", "content": final_response})
                self.memory.append({"role": "model", "content": final_response})
                return final_response
            else:
                self.messages.append({"role": "model", "content": llm_response})
                self.memory.append({"role": "model", "content": llm_response})
                return llm_response

        except Exception as e:
            print(f"Error processing response: {e}")
            raise RuntimeError(f"Response processing failed: {e}") from e

        finally:
            await self.cleanup_servers()

async def initialize_session(user_ip: str) -> ChatSession:
    """Initialize and return the chat session."""
    config = Configuration()  # Assumes Configuration class exists
    server_config = config.load_server_config("app/AI/servers_config.json")
    servers = [
        Server(name, srv_config)
        for name, srv_config in server_config["mcpServers"].items()
    ]
    llm_providers = get_providers()
    llm_client = LLMClient(llm_providers)
    chat_session = await ChatSession.create(servers, llm_client, user_ip)
    return chat_session



def get_providers():
    """Retrieve registered LLM providers from llm_config.json."""
    provider_registry = {}
    with open("app/AI/llm_Config/llm_config.json", "r") as f:
        llm_config = json.load(f)
        for provider_name, provider_config in llm_config["providers"].items():
            
            if provider_name == "groq": provider_class = GroqProvider(GROQConfig(provider_config))
            elif provider_name == "azure": provider_class = AzureProvider(AZUREConfig(provider_config))
            elif provider_name == "gemini": provider_class = GeminiProvider(GEMINIConfig(provider_config))
            else: raise ValueError(f"Unknown provider: {provider_name}")
            provider_registry[provider_name] = provider_class
    return provider_registry
