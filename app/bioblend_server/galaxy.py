import logging 
import os
from dotenv import load_dotenv
load_dotenv()

from bioblend import galaxy
from bioblend.galaxy.objects import GalaxyInstance
from app.bioblend_server.informer.informer import GalaxyInformer

class GalaxyClient:

    def __init__(self):
        self.galaxy_url = os.getenv('GALAXY_URL')
        self.galaxy_api_key = os.getenv('GALAXY_API_KEY')
        self.gi_object = GalaxyInstance(url=self.galaxy_url, api_key=self.galaxy_api_key)
        self.gi_client= self.gi_object.gi
        self.limit = 2
        self.offset = 0
        self.config_client = galaxy.config.ConfigClient(self.gi_client)
        self.tool_client = galaxy.tools.ToolClient(self.gi_client)
        self.logger=logging.getLogger(__class__.__name__)

    def whoami(self):
        return self.config_client.whoami()

    
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