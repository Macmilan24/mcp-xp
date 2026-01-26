import asyncio
import logging

from sys import path
path.append(".")

from app.bioblend_server.informer.global_rec import GlobalRecommender
from app.bioblend_server.informer.utils import InformerTTLs

class BackgroundIndexer:
    
    def __init__(self):
        self.log = logging.getLogger(__class__.__name__)
        self.running = False
        self.global_rec: GlobalRecommender = None 
        
    async def run_once(self):
        self.global_rec = await GlobalRecommender().create()
        await asyncio.gather(self.global_rec.store_scraped_tools(), self.global_rec.store_scraped_workflows())

    async def run_loop(self):
        """The main background loop that runs forever until cancelled."""
        
        self.log.info("Background scraper loop started.")
        self.global_rec = await GlobalRecommender().create()
        self.running = True

        try:
            while self.running:
                self.log.info("Scraping and Indexing Global Recommender metadata.")

                try:
                    await asyncio.gather(
                        self.global_rec.store_scraped_tools(),
                        self.global_rec.store_scraped_workflows()
                    )
                    
                    self.log.info("Cycle complete. Sleeping.")
                
                except Exception as e:
                    # Catch application-level errors so the loop doesn't die entirely
                    self.log.error(f"Error during scraping cycle: {e}", exc_info=True)
                    
                # Wait for next cycle
                await asyncio.sleep(InformerTTLs.LIFESPAN.value)

        except asyncio.CancelledError:
            self.log.info("Background task received cancellation signal.")
            self.running = False
            raise
        finally:
            self.log.info("Background scraper loop shutdown complete.")