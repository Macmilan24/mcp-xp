import json
import logging

from datetime import datetime
from app.AI.server import Server

from app.AI.llm_config.groq_config import GROQConfig
from app.AI.llm_config.azure_config import AZUREConfig
from app.AI.llm_config.gemini_config import GEMINIConfig

from app.AI.provider.groq_provider import GroqProvider
from app.AI.provider.azure_provider import AzureProvider
from app.AI.provider.gemini_provider import GeminiProvider

from app.config import Configuration

class LLMClient:
    def __init__(self, llm_providers: dict):
        self.providers = llm_providers  # Creates an instance of the providers

logger=logging.getLogger()

class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], llm_client: LLMClient, user_ip: str = None) -> None:
        self.servers: list[Server] = servers
        self.llm_client: LLMClient = llm_client
        self.memory = []
        self.system_message = None
        self.messages: list = None
        self.user_ip = user_ip
        self.tools = None 
        self.log_filename = f"chat_session_{self.user_ip}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.logger=logging.getLogger(__class__.__name__)

    @classmethod
    async def create(cls, servers: list[Server], llm_client: LLMClient, user_ip: str = None) -> "ChatSession":
        self = cls(servers, llm_client, user_ip)
        
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                except Exception as e:
                    self.logger.error(f"Failed to initialize server {server.name}: {e}")
                    await self.cleanup_servers()
                    raise RuntimeError(f"Server initialization failed: {e}") from e

            # Gather tools
            all_tools = []
            try:
                for server in self.servers:
                    tools = await server.list_tools()
                    self.logger.info(f"Type of tools: {type(tools)} for {server.name}")
                    self.tools=tools.tools
                    all_tools.extend(tools.tools)
                tools_description = [tool for tool in all_tools]
            except Exception as e:
                self.logger.error(f"Error listing tools: {e}")
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
                self.logger.info("server cleared up successfully.")
            except Exception as e:
                # Silently handle cleanup errors to ensure all servers get cleanup attempts
                self.logger.error(f"Error cleaning up server {server.name}: {e}")
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
            tool_call = json.loads(llm_response) if isinstance(llm_response,str) else llm_response
            self.logger.info(f" the tool called: {tool_call}")
            if "tool" in tool_call and "arguments" in tool_call:
                # logging.info(f"Executing tool: {tool_call['tool']}")
                # logging.info(f"With arguments: {tool_call['arguments']}")

                for server in self.servers:
                    tools = self.tools  # Use stored tools
                    
                    if any(tool.name == tool_call["tool"] for tool in tools):
                        try:
                            result = await server.execute_tool(
                                tool_call["tool"], tool_call["arguments"]
                            )
                            # print(f'result from tool call: {result}')
                            if isinstance(result, dict) and "progress" in result:
                                progress = result["progress"]
                                total = result["total"]
                                percentage = (progress / total) * 100
                                # logging.info(
                                #     f"Progress: {progress}/{total} "
                                #     f"({percentage:.1f}%)"
                                # )
                            if isinstance(result.content, list) and len(result.content) > 0:

                                content_type = result.content[0].type
                                content_text = result.content[0].text
                                annotations = result.content[0].annotations

                                self.logger.info(f" The return type: {content_type} Annotations: {annotations} Content : {content_text} type: {type(content_text)}")
                            
                                if isinstance(content_text, str):
                                    try:
                                        content_text = json.loads(content_text)
                                        self.logger.info(f"Parsed JSON content, type: {type(content_text)}")
                                    except json.JSONDecodeError:
                                        self.logger.warning("Content is not valid JSON.")

                                if isinstance(content_text, dict) and content_text.get("action_link"):
                                    return content_text

                                else:
                                    return f"""
                                            You are required to respond **strictly and exclusively** based on the following Tool Execution Result:
                                            **{content_text}**

                                            **Instructions:**
                                            1. If the Tool Execution Result is complete and directly answers the query, **return it exactly as-is**. Do **not** paraphrase, summarize, interpret, or alter it in any way.
                                            2. If the Tool Execution Result is **incomplete, unclear, or insufficient**, respond only using the information it contains—**do not draw on external knowledge or assumptions**.
                                            3. Your response must remain **self-contained**, with no reference to outside sources, general knowledge, or unrelated context.
                                            4. If appropriate, you may suggest **guidance or next steps**, but only when clearly warranted by the Tool Execution Result, and only if they can be logically and explicitly **derived from the given context**.
                                            5. Never introduce new information, explanations, or assumptions beyond what is directly stated in the Tool Execution Result.

                                            **Default behavior:**
                                            Always return the Tool Execution Result **verbatim**, unless doing so would leave the query unresolved **based solely on the result itself**.
                                            """
                            else:
                                return "Tool execution result: No content returned."
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
            # self.messages.extend(self.memory)
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
                self.messages.append({"role": "assistant", "content": f"{llm_response}"})
                if isinstance(result, dict):
                    self.messages.append({"role": "user", "content":"\n".join(f"{key}: {value}" for key, value in result.items())})
                else:
                    self.messages.append({"role": "user", "content": result})

                if isinstance(result, dict) and result.get("action_link"):
                    final_response = result
                    try:
                        
                        self.messages.append({"role": "assistant", "content": f"Generated form for {final_response['entity']} name: {final_response['name']}, return form link: {final_response['action_link']}"})
                        self.memory.append({"role": "assistant", "content": f"Generated form for {final_response['entity']} name: {final_response['name']}, return form link: {final_response['action_link']}"})
                    except Exception as e:
                        raise                
                else:
                    final_response = await providers[model_id].get_response(self.messages)
                    self.messages.append({"role": "assistant", "content": final_response})
                    self.memory.append({"role": "assistant", "content": final_response})
                # print(self.messages)
                return final_response
            else:
                self.messages.append({"role": "assistant", "content": llm_response})
                self.memory.append({"role": "assistant", "content": llm_response})
                return llm_response

        except Exception as e:
            self.logger.error(f"Error processing response: {e}")
            raise RuntimeError(f"Response processing failed: {e}") from e

        # finally:
        #     await self.cleanup_servers()


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
    logger.info('Initializng session')
    chat_session = await ChatSession.create(servers, llm_client, user_ip)
    return chat_session



def get_providers():
    """Retrieve registered LLM providers from llm_config.json."""
    provider_registry = {}
    with open("app/AI/llm_config/llm_config.json", "r") as f:
        llm_config = json.load(f)
        for provider_name, provider_config in llm_config["providers"].items():
            
            if provider_name == "groq": provider_class = GroqProvider(GROQConfig(provider_config))
            elif provider_name == "azure": provider_class = AzureProvider(AZUREConfig(provider_config))
            elif provider_name == "gemini": provider_class = GeminiProvider(GEMINIConfig(provider_config))
            else: raise ValueError(f"Unknown provider: {provider_name}")
            provider_registry[provider_name] = provider_class
    return provider_registry
