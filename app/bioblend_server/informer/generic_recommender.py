import logging
from typing import Literal
from app.bioblend_server.informer.manager import InformerManager

class GenericRecommender:
    def __init__(self, entity_type: Literal["tool", "workflow"]):
        
        self.entity_type = entity_type
        self.log = logging.getLogger(__class__.__name__)
        self.manager = InformerManager()
        
    async def scrape_tool():
        # TODO: Add tool scraper logic here.
        pass
    
    async def scrape_workflow():
        # TODO: Add workflow tool scraper logic here.
        pass
    
    async def store_to_collection(self, scraped_data: list[dict]):
        collection_name = f"generic_galaxy_{self.entity_type}"
        self.log.info(f"Storing scraped galaxy data to Qdrant.")
        await self.manager.embed_and_store_entities(entities = scraped_data, collection_name = collection_name)
        self.log.info(f"content embedded and stored in {collection_name} succefully.")
        
    def recommend(self, query_vector:list):
        collection_name = f"generic_galaxy_{self.entity_type}"
        try:
            results = self.manager.search_by_vector(collection=collection_name, query_vector=query_vector, entity_type=self.entity_type)
        except Exception as e:
            self.log.error(f"Error occured recommending from generic scraped data: {e}")
            return {}
        
        return results