from __future__ import annotations

import logging
from dotenv import load_dotenv
from typing import List

load_dotenv()

from sys import path
path.append('.')

from app.bioblend_server.galaxy import GalaxyClient
from bioblend.galaxy.objects.wrappers import History


class HistoryManager:

    def __init__(self, galaxy_client: GalaxyClient):
        self.galaxy_client = galaxy_client
        self.gi_user = self.galaxy_client.gi_object 
        self.log = logging.getLogger(self.__class__.__name__)

    def create(self, name:str = None) -> History:
        history=self.gi_user.histories.create(name=name)
        return history
    
    def get(self, history_id:str)-> History:
        history=self.gi_user.histories.get(history_id)
        return history
    
    def purge(self, history: History):
        history.delete(purge=True)

    def delete(self, history: History):
        history.delete()
    
    def list_history(self)-> List[History]:
        history = self.gi_user.histories.list()
        return history 