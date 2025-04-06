
from utils.fetch_tool_source_code import fetch_galaxy_tool_source_code
from utils.input_output_xml_data_extractor import extract_input_details, extract_output_details
from utils.extract_inputs_outputs_xml import extract_inputs_outputs
from config import GALAXY_URL, GALAXY_API_KEY
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
    
    return """{
        "name": "Unzip collection",
        "description": "",
        "category": "Collection Operations",
        "help": "Synopsis\nThis tool takes a paired collection and \"unzips\" it into two simple dataset collections (lists of datasets).\n\nDescription\n1. **Functionality**\n   - Given a paired collection of forward and reverse reads, this tool separates them into two distinct collections.\n   - The first output collection contains all forward reads, and the second output collection contains all reverse reads.\n\n2. **Use Case**\n   - Useful for processing paired-end sequencing data.\n   - Enables downstream analysis by handling forward and reverse reads separately.\n\nThis tool simplifies paired dataset management, allowing for more flexible analysis workflows in Galaxy."
    },"""
def get_tools(number_of_tools=10):
    """
    Get the tools from the galaxy instance
    """
    gi = setup_instance()
    t = tools(gi, int(number_of_tools))
    return t


if __name__ == "__main__":
    tools()
    