from bioblend import galaxy
from utils.fetch_tool_source_code import fetch_galaxy_tool_source_code
from utils.input_output_xml_data_extractor import extract_input_details,extract_output_details
from utils.extract_inputs_outputs_xml import extract_inputs_outputs
from config import GALAXY_URL, GALAXY_API_KEY
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

def get_tools(gi):

    
    gi = setup_instance()
    tools = galaxy.tools.ToolClient(gi)

    tool_list = tools.show_tool(tool_id="upload1")
    # pprint.pprint(tool_list)
    single_tool = tools.get_tools()

    for tool in single_tool[:5]:
        id  = tool.get("id")
        response = fetch_galaxy_tool_source_code(tool_id=id,galaxy_url=galaxy_url)
        if response:
            source_xml = response.text
            inputs, outputs = extract_inputs_outputs(source_xml)

        if inputs:
            tool["xml_input"] = extract_input_details(inputs)
        if outputs:
            tool["xml_output"] = extract_output_details(outputs)


def tools():
    gi = setup_instance()
    tools = get_tools(gi)
    return tools


if __name__ == "__main__":
    tools()
    