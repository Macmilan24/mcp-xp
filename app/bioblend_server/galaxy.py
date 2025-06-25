from app.config import GALAXY_URL, GALAXY_API_KEY
from bioblend import galaxy
import pprint

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

# TODO:: Maybe should be removed
def setup_instance():
    
    gi = galaxy.GalaxyInstance(url=GALAXY_URL, key=GALAXY_API_KEY)

    hl = gi.histories.get_histories()
    con = galaxy.config.ConfigClient(gi)

    # test the credit have been created successfully
    con.get_config()
    print(con.whoami())

    return gi

def tools(gi, number_of_tools=10):

    
    gi = setup_instance()
    tools = galaxy.tools.ToolClient(gi)

    tool_list = tools.show_tool(tool_id="upload1")
    # pprint.pprint(tool_list)
    single_tool = tools.get_tools()
    tools = []
    for tool in single_tool[:number_of_tools]:
        print(tool)
        
        tools.append(pprint.pformat(tool) + "\n")
    # return "\n".join(tools)
    return tools
    
    
def get_tools(number_of_tools=10):
    """
    Get the tools from the galaxy instance
    """
    gi = setup_instance()
    print("gi ", gi)
    t = tools(gi, int(number_of_tools))
    print("t ",t)
    if t:
        return t
    else:
        return "none found"

def get_tool(id):

    """
    Get a specific tool by its ID from the galaxy instance
    """
    gi = setup_instance()
    tools_client = galaxy.tools.ToolClient(gi)
    tool = tools_client.show_tool(tool_id=id)
    return pprint.pformat(tool)

if __name__ == "__main__":
    tools()
    