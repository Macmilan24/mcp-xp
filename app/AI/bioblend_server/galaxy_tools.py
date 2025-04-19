print("bioblend_server galaxytools.py")
from app.config import GALAXY_URL, GALAXY_API_KEY
from bioblend import galaxy
import os
import pprint

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
    return "\n".join(tools)
    
    
def get_tools(number_of_tools=10):
    """
    Get the tools from the galaxy instance
    """
    gi = setup_instance()
    t = tools(gi, int(number_of_tools))
    return t

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
    