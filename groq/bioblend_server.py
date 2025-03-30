from bioblend import galaxy
from fetch_tool_source_code import fetch_galaxy_tool_source_code
from input_output_xml_data_extractor import extract_input_details,extract_output_details
from extract_inputs_outputs_xml import extract_inputs_outputs
from convert_to_html_form import converter
from dotenv import load_dotenv
import os
import pprint

# Load environment variables from .env file
load_dotenv()

galaxy_url = os.getenv("GALAXY_URL")
galaxy_api_key = os.getenv("GALAXY_API_KEY")

gi = galaxy.GalaxyInstance(url=galaxy_url, key=galaxy_api_key)

hl = gi.histories.get_histories()
con = galaxy.config.ConfigClient(gi)

# test the credit have been created successfully
con.get_config()
print(con.whoami())


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


# testing the source code have been fetched
# pprint.pprint(single_tool[:8])

# test to the converter fuction
# for i in range(5):
  # pprint.pprint(converter(single_tool[i]))