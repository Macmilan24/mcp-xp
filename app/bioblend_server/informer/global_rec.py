from sys import path
path.append(".")

import logging
from typing import Literal

from app.bioblend_server.informer.manager import InformerManager
from app.bioblend_server.informer.scrapers.tool_scraper import GalaxyToolScraper
from app.bioblend_server.informer.scrapers.workflow_scraper import GalaxyWorkflowScraper

class GlobalRecommender:
        
    def __init__(self):
        self.manager: InformerManager = None
        self.log = logging.getLogger(__class__.__name__)
        self.tool_scraper = GalaxyToolScraper()
        self.workflow_scraper = GalaxyWorkflowScraper()     
    
    @classmethod
    async def create(cls):
        self = cls()
        self.manager = await InformerManager().create()
        return self
    
    async def store_scraped_tools(self):
         scraped_tools = await self.tool_scraper.scrape_tool()
         await self.tool_scraper.close()
         await self.store_to_collection(scraped_tools, "tool")
         
    async def store_scraped_workflows(self):
         scraped_workflows = await self.workflow_scraper.scrape_workflows()
         await self.store_to_collection(scraped_workflows, "workflow")
               
    async def store_to_collection(self, scraped_data: list[dict], entity_type: Literal["tool", "workflow"]):
        collection_name = f"generic_galaxy_{entity_type}"
        self.log.info(f"Storing scraped galaxy data to Qdrant.")
        await self.manager.embed_and_store_entities(entities = scraped_data, collection_name = collection_name)
        self.log.info(f"content embedded and stored in {collection_name} succefully.")