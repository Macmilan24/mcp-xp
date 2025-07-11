from bioblend import galaxy
from app.bioblend_server.informer.informer import GalaxyInformer

class GalaxyClient:

    def __init__(self, galaxy_url, galaxy_api_key):
        self.galaxy_url = galaxy_url
        self.galaxy_api_key = galaxy_api_key
        self.gi = galaxy.GalaxyInstance(url=self.galaxy_url, key=self.galaxy_api_key)
        self.limit = 2
        self.offset = 0
        self.config_client = galaxy.config.ConfigClient(self.gi)
        self.tool_client = galaxy.tools.ToolClient(self.gi)

    def whoami(self):
        return self.config_client.whoami()

    def get_tools(self, limit=None, offset=None):
        """
        Get a list of tools from the Galaxy instance with optional limit and offset.
        
        Args:
            limit (int, optional): Number of tools to return. Defaults to self.limit.
            offset (int, optional): Start index. Defaults to self.offset.

        Returns:
            tuple: (total_tool_count, list_of_tools)
        """
        limit = limit if limit is not None else self.limit
        offset = offset if offset is not None else self.offset

        all_tools = self.tool_client.get_tools()
        
        if not all_tools:
            return [], []

        total = len(all_tools)
        tools = all_tools[offset:offset + limit]

        if total > 0:
            return total, tools
        else:
            return "No tools found for this Galaxy Instance"

    def get_tool(self, tool_id):
        """
        Get a specific tool by its ID from the galaxy instance
        """
        tool = self.tools_client.show_tool(tool_id=tool_id)
        return (tool)
    
# informer(information retriever) tool as a tool for the MCP server
async def get_galaxy_information(query: str, query_type: str, entity_id: str =None):
    """
    Fetch detailed information on Galaxy tools, workflows, datasets,
    and invocations—including their usage, current state, and any related 
    queries—handling all information requests about Galaxy entities, based on 
    the query_type(tool, workflow, dataset) and the query
    
    Args: query(str): The user query message that needs response
          query_type(str): The tyoe of galaxy entity the query needs a rsponse 
                           for with either of 3 values: tool, dataset, workflow
          entity_id(str) optional : The entity id

    return: dictionary: {
                        query: the users query message
                        retrieved_content: galaxy information details for the entity
                        response: Response for the users query message
                        }
    """

    informer= GalaxyInformer(query_type)
    galaxy_response= await informer.get_entity_info(search_query = query, entity_id = entity_id)
    return galaxy_response.get('response')

# if __name__ == "__main__":
#     # tools()
#    import asyncio
#    asyncio.run(get_galaxy_information(query="find me tools to convert bed file into gtf file", query_type='tool'))