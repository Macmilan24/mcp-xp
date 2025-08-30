import logging
import os
from dotenv import load_dotenv

load_dotenv()

from bioblend import galaxy
from bioblend.galaxy.objects import GalaxyInstance


# Per request instantiation
class GalaxyClient:

    def __init__(self, user_api_key: str):
        self.galaxy_url = os.getenv("GALAXY_URL")
        self.admin_api_key = os.getenv("GALAXY_API_KEY")
        self.user_api_key = user_api_key

        # Galaxy instance with administarative access, for some functionalities
        self.gi_admin = GalaxyInstance(url=self.galaxy_url, api_key=self.admin_api_key)
        # Galaxy instance for users.
        self.gi_object = GalaxyInstance(url=self.galaxy_url, api_key=self.user_api_key)

        self.gi_client = self.gi_object.gi

        self.config_client = galaxy.config.ConfigClient(self.gi_client)
        self.logger = logging.getLogger(__class__.__name__)

    def whoami(self):
        return self.config_client.whoami()
