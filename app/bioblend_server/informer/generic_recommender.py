import json
import os
from sys import path
path.append(".")
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from bioblend.galaxy import GalaxyInstance
import requests
from datetime import datetime
import yaml
import random
from tqdm import tqdm
import logging
from typing import Literal
from app.bioblend_server.informer.manager import InformerManager
from dotenv import load_dotenv
import httpx


class GenericRecommender:
    def __init__(self, entity_type: Literal["tool", "workflow"]):
        
        self.entity_type = entity_type
        self.log = logging.getLogger(__class__.__name__)
        self.manager = InformerManager()
        
        load_dotenv()
        config_file = "config.yml"
        if not os.path.exists(config_file):
            self.log.error(f"Configuration file {config_file} not found.")
            raise FileNotFoundError(f"Configuration file {config_file} not found.")
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)
            
        # Get Galaxy scraping credentials from environment variables
        self.galaxy_url = os.getenv("GALAXY_SCRAPING_URL")
        self.galaxy_url_api_key = os.getenv("GALAXY_SCRAPING_API_KEY")
        
        # Validate credentials
        if not self.galaxy_url or not self.galaxy_url_api_key:
            self.log.error("Galaxy scraping URL or API key not found in environment variables.")
            raise ValueError("Galaxy scraping URL or API key not found in environment variables.")
        # Initialize Galaxy Instance
        self.gi = GalaxyInstance(url=self.galaxy_url, key=self.galaxy_url_api_key)

        # github token for future use
        self.github_token = os.getenv("GITHUB_TOKEN", None)
        self.github_API_URL = "https://api.github.com/repos/galaxyproject/iwc/contents/workflows"
        self.raw_base_url = "https://raw.githubusercontent.com/galaxyproject/iwc/main/workflows"
        self.maximum_workflow_fetch = 50
        if self.github_token:
            self.headers = {
                "Authorization": f"token {self.github_token}"
            }
        else:
            self.log.warning("No GITHUB_TOKEN found in environment variables. Using unauthenticated requests may hit rate limits.")
            self.headers = {}
    async def scrape_tool(self):
        # setting variables
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = f"data/galaxy_tools_{timestamp}.json"
        max_workers = self.config["processing"]["max_workers"]
        tool_limit = self.config["processing"].get("tool_limit", None)
        categories_config = self.config.get("categories", {})
        self.log.info(f"Scraping galaxy tools data...")
        

        
        if os.path.exists(output_file):
            self.log.info(f"Output file {output_file} already exists. Skipping scraping.")
        else:
            self.log.info(f"Fetching tool list from Galaxy instance at {self.galaxy_url}...")
            tools = self.gi.tools.get_tools()
            self.log.info(f"Total tools fetched: {len(tools)}")
            
            # Group tools By Category
            tools_by_category = {}
            for tool in tools:
                category = tool.get("panel_section_name", "Uncategorized")
                tools_by_category.setdefault(category, []).append(tool)
                
            # Deduplicate tools
            tool_map = {}
            for category, cat_tools in tools_by_category.items():
                percentage = categories_config.get(category, {}).get("percentage", 1)
                if not isinstance(percentage, (int, float)) or percentage < 0:
                    print(f"Invalid percentage for {category}: {percentage}. Using 100%.")
                    percentage = 1
                percentage = min(percentage, 1)
                num_tools = int(len(cat_tools) * percentage)
                if num_tools == 0 and percentage > 0:
                    num_tools = 1
                self.log.info(f"Selecting {num_tools} tools from category '{category}' ({len(cat_tools)} total tools, {percentage*100}% specified).")
                selected_tools = random.sample(cat_tools, num_tools) if num_tools < len(cat_tools) else cat_tools
                
                for tool in selected_tools:
                    tool_id = tool.get("id")
                    
                    if tool_id not in tool_map:
                        tool_map[tool_id] = {"base": tool, "categories": set()}
                    tool_map[tool_id]["categories"].add(category)
                    
            # Final Filtered tools 
            filtered_tools = []
            for entry in tool_map.values():
                tool = entry["base"]
                tool["categories"] = list(entry["categories"])
                filtered_tools.append(tool)
            if tool_limit:
                filtered_tools = filtered_tools[:tool_limit]
  
            self.log.info(f"Fetching Details for {len(filtered_tools)} tools using {max_workers} workers...")
            
        # Scrape tool details concurrently
        tools_json = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tool = {executor.submit(self.fetch_tools_detail, tool): tool for tool in filtered_tools}
            for future in tqdm(as_completed(future_to_tool), total=len(future_to_tool), desc="Scraping Tools"):
                tool = future_to_tool[future]
                try:
                    tool_data = future.result()
                    if tool_data:
                        tools_json.append(tool_data)
                except Exception as e:
                    self.log.error(f"Error processing tool {tool.get('id', 'unknown')}: {e}")
        # Save scraped data to JSON file
        with open(output_file, "w") as f:
            json.dump(tools_json, f, indent=4)
        self.log.info(f"Scraped data saved to {output_file}.")
           

    # Scrape helper function
    def fetch_tools_detail(self, tool: dict) -> dict | None:
        tool_id = tool.get("id", "")
        try:
            tools_details = self.gi.tools.show_tool(tool_id, io_details=True)
            
            help_text = ""
            raw_tool_url = f"{self.galaxy_url}/api/tools/{tool_id}/raw_tool_source?key={self.galaxy_url_api_key}"
            response = requests.get(raw_tool_url)
            response.raise_for_status()
            tool_xml = response.text
            
            root = ET.fromstring(tool_xml)
            help_elem = root.find("help")
            if help_elem is not None and help_elem.text:
                help_text = help_elem.text.strip()
                self.log.info(f"Content Extracted from help section for tool {tool_id}.")
            else:
                self.log.info(f"No help section found for tool {tool_id}.")
                
            return {
                "id": tool_id, 
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "categories": tool.get("categories", []),
                "version": tool.get("version", ""),
                "help": help_text
            }
        except Exception as e:
            self.log.error(f"Error fetching details for tool {tool_id}: {e}")
            return None
        

    # Async GET request to GitHub API
    async def github_api_get(self, url):
        async with httpx.AsyncClient(headers=self.headers) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()

    # Async fetch raw file contents
    async def fetch_file_contents(self, path):
        raw_url = f"{self.raw_base_url}/{path}"
        async with httpx.AsyncClient(headers=self.headers) as client:
            response = await client.get(raw_url)
            response.raise_for_status()
            return response.text
        
    # Parse .ga workflow content
    async def parse_ga_content(self, ga_text):
        try:
            data = json.loads(ga_text)
            workflow_name = data.get("name", "unknown")
            steps = data.get("steps", {})
            tools_used = []

            for step in steps.values():
                if step.get("type") != "tool":
                    continue
                repo = step.get("tool_shed_repository", {})
                tool_info = {
                    "id": step.get("tool_id", ""),
                    "name": step.get("name", ""),
                    "version": step.get("tool_version", ""),
                    "owner": repo.get("owner", ""),
                    "category": repo.get("name", ""),
                    "tool_shed_url": repo.get("tool_shed", "")
                }
                if tool_info not in tools_used:
                    tools_used.append(tool_info)

            return {
                "workflow_name": workflow_name,
                "number_of_steps": len(steps),
                "tools_used": tools_used
            }
        except Exception as e:
            self.log.error(f"Failed to parse .ga JSON: {e}")
            return {}

    # Scan repository folder for workflows
    async def scan_repo(self, category, repo_name):
        base_path = f"{category}/{repo_name}"
        url = f"{self.github_API_URL}/{category}/{repo_name}"

        try:
            repo_contents = await self.github_api_get(url)
        except Exception as e:
            self.log.error(f"Failed to get repo contents for {base_path}: {e}")
            return None

        workflow_files = []
        files_present = set()
        directories_present = set()
        readme_content = None

        for item in repo_contents:
            name = item["name"]
            if item["type"] == "file":
                files_present.add(name)
                if name.endswith(".ga"):
                    ga_text = await self.fetch_file_contents(f"{base_path}/{name}")
                    ga_info = await self.parse_ga_content(ga_text)

                    # Add file info and raw download link
                    ga_info.update({
                        "file_name": name,
                        "raw_download_url": f"{self.raw_base_url}/{base_path}/{name}"
                    })

                    workflow_files.append(ga_info)

                if name == "README.md":
                    try:
                        readme_content = await self.fetch_file_contents(f"{base_path}/README.md")
                    except Exception as e:
                        self.log.error(f"Failed to fetch README for {base_path}: {e}")

            elif item["type"] == "dir":
                directories_present.add(name)

        return {
            "category": category.lower(),
            "workflow_repository": repo_name.lower(),
            "workflow_files": workflow_files,
            "has_test_data": "test-data" in directories_present,
            "has_dockstore_yml": ".dockstore.yml" in files_present,
            "has_readme": "README.md" in files_present,
            "readme_content": readme_content,
            "has_changelog": "CHANGELOG.md" in files_present
        }

    # Main scraper
    async def scrape_workflows(self):
        self.log.info("Scraping Galaxy workflows from GitHub...")

        categories = await self.github_api_get(self.github_API_URL)
        all_data = []
        workflow_count = 0

        for category_item in categories:
            if category_item["type"] != "dir":
                continue
            category = category_item["name"]
            self.log.info(f"Scanning category: {category}")

            repos = await self.github_api_get(category_item["url"])
            for repo in repos:
                if repo["type"] != "dir":
                    continue

                repo_name = repo["name"]
                repo_data = await self.scan_repo(category, repo_name)
                if repo_data:
                    all_data.append(repo_data)
                    workflow_count += len(repo_data["workflow_files"])
                    if self.maximum_workflow_fetch and workflow_count >= self.maximum_workflow_fetch:
                        self.log.info(f"Reached maximum workflow fetch limit of {self.maximum_workflow_fetch}. Stopping.")
                        break
            if self.maximum_workflow_fetch and workflow_count >= self.maximum_workflow_fetch:
                break

        # Save output
        os.makedirs("data", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = f"data/galaxy_workflows_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(all_data, f, indent=4)
        self.log.info(f"Scraped workflow data saved to {output_file}.")
                
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
    
if __name__ == "__main__":
    import asyncio
    # recommender = GenericRecommender(entity_type="tool")
    # asyncio.run(recommender.scrape_tool())
    
    recommender_workflow = GenericRecommender(entity_type="workflow")
    asyncio.run(recommender_workflow.scrape_workflows())
    