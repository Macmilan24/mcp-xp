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
import re
from pathlib import Path


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
        self.maximum_workflow_fetch = None
        if self.github_token:
            self.headers = {
                "Authorization": f"token {self.github_token}"
            }
        else:
            self.log.warning("No GITHUB_TOKEN found in environment variables. Using unauthenticated requests may hit rate limits.")
            self.headers = {}

        self.raw_dir = Path("data")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
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

            name = tool.get("name", "").strip()
            description = tool.get("description", "").strip()
            categories = tool.get("categories", [])
            version = tool.get("version", "").strip()

            # Construct a language-model-friendly content field
            content_parts = [
                f"Tool Name: {name}.",
                f"Description: {description}." if description else "",
                f"Version: {version}." if version else "",
                f"Categories: {', '.join(categories)}." if categories else "",
                f"Help Section: {help_text}." if help_text else "",
            ]

            # Filter empty entries and join into a cohesive, well-structured paragraph
            content = " ".join([part for part in content_parts if part]).strip()

            # You can make this more natural-language friendly for embeddings:
            content = (
                f"This is a Galaxy tool named '{name}'. "
                f"It is described as follows: {description if description else 'No description available.'} "
                f"The tool belongs to the following categories: {', '.join(categories) if categories else 'Uncategorized'}. "
                f"The current version of the tool is {version if version else 'unspecified'}. "
                f"Here is the help or usage information: {help_text if help_text else 'No help content provided.'}"
            )

            return {
                "id": tool_id,
                "name": name,
                "description": description,
                "categories": categories,
                "version": version,
                "help": help_text,
                "content": content
            }

        except Exception as e:
            self.log.error(f"Error fetching details for tool {tool_id}: {e}")
            return None
        

    async def github_api_get(self, url: str) -> dict | list:
            async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.json()

    async def fetch_file_contents(self, path: str) -> str:
        raw_url = f"{self.raw_base_url}/{path}"
        async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
            resp = await client.get(raw_url)
            resp.raise_for_status()
            return resp.text

    # ---------------------------
    # Parse .ga content
    # ---------------------------
    async def parse_ga_content(self, ga_text: str) -> dict:
        """Return dict with workflow_name, number_of_steps, tools_used (list of dicts)."""
        try:
            data = json.loads(ga_text)
            workflow_name = data.get("name", "unknown")
            steps = data.get("steps", {}) or {}
            tools_used: list = []

            for step in steps.values():
                if step.get("type") != "tool":
                    continue
                repo = step.get("tool_shed_repository", {}) or {}
                tool_info = {
                    "id": step.get("tool_id", "") or "",
                    "name": step.get("name", "") or "",
                    "version": step.get("tool_version", "") or "",
                    "owner": repo.get("owner", "") or "",
                    "category": repo.get("name", "") or "",
                    "tool_shed_url": repo.get("tool_shed", "") or ""
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
            return {"workflow_name": "unknown", "number_of_steps": 0, "tools_used": []}

    # ---------------------------
    # Repo scanner
    # ---------------------------
    async def scan_repo(self, category: str, repo_name: str) -> dict | None:
        base_path = f"{category}/{repo_name}"
        api_url = f"{self.github_API_URL}/{category}/{repo_name}"

        try:
            repo_contents = await self.github_api_get(api_url)
        except Exception as e:
            self.log.error(f"Failed to list repo contents for {base_path}: {e}")
            return None

        workflow_files = []
        files_present = set()
        directories_present = set()
        readme_content: str | None = None

        for item in repo_contents:
            name = item.get("name", "")
            itype = item.get("type", "")
            if itype == "file":
                files_present.add(name)
                if name.endswith(".ga"):
                    # fetch and parse .ga content
                    try:
                        ga_text = await self.fetch_file_contents(f"{base_path}/{name}")
                        ga_info = await self.parse_ga_content(ga_text)
                    except Exception as e:
                        self.log.error(f"Error fetching/parsing {base_path}/{name}: {e}")
                        continue

                    # add filename and raw URL
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
                        readme_content = None

            elif itype == "dir":
                directories_present.add(name)

        return {
            "category": category.lower(),
            "workflow_repository": repo_name.lower(),
            "workflow_files": workflow_files,
            "planemo_tests": [],  # left for compatibility (could be filled if needed)
            "has_test_data": "test-data" in directories_present,
            "has_dockstore_yml": ".dockstore.yml" in files_present,
            "has_readme": "README.md" in files_present,
            "readme_content": readme_content,
            "has_changelog": "CHANGELOG.md" in files_present
        }


    def clean_readme(self, text: str) -> str:
        """Normalize and clean README/help text for embedding."""
        if not isinstance(text, str):
            return ""
        # Remove HTML/XML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Replace multiple underline/equal sequences
        text = re.sub(r"[=_]{2,}", " ", text)
        # Remove box drawing / ASCII table characters
        text = re.sub(r"[│║╔╗╚╝╠╣═╦╩╬┼─┐└┘┌┴┬├┤]+", " ", text)
        # Remove markdown table lines (+ | -)
        text = re.sub(r"[\+\|\-]{2,}", " ", text)
        # Remove stray markdown special chars
        text = re.sub(r"[*_`#\[\]]", "", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def preprocess_scraped(self, raw_data: list) -> list:
        """
        Convert raw scraped data (list of repo dicts) into a preprocessed, compact format:
        - category
        - workflow_repository
        - workflow_name (first non-empty workflow_files.workflow_name)
        - tool_names (unique flattened list from all workflow_files)
        - readme_cleaned
        - raw_download_url (first non-empty found)
        - content (structured text for embeddings)
        """
        preprocessed = []
        for repo in raw_data:
            tool_names = []
            chosen_workflow_name = None
            chosen_raw_url = None

            # Iterate all workflow_files and aggregate tool names; choose first non-empty name & raw_url
            for wf in repo.get("workflow_files", []):
                # Capture first non-empty workflow name
                if not chosen_workflow_name:
                    name = wf.get("workflow_name") or ""
                    if name:
                        chosen_workflow_name = name
                # Capture first non-empty raw URL
                if not chosen_raw_url:
                    url = wf.get("raw_download_url") or ""
                    if url:
                        chosen_raw_url = url
                # Collect tool names
                for tool in wf.get("tools_used", []):
                    tname = tool.get("name")
                    if tname:
                        tool_names.append(tname)

            # Make unique while preserving order
            seen = set()
            unique_tool_names = []
            for t in tool_names:
                if t not in seen:
                    seen.add(t)
                    unique_tool_names.append(t)

            readme = repo.get("readme_content", "") or ""
            cleaned_readme = self.clean_readme(readme)

            # Well-structured LLM-friendly content field
            content = (
                f"This is a Galaxy workflow.\n"
                f"Workflow Name: {chosen_workflow_name or repo.get('workflow_repository', 'Unknown')}.\n"
                f"Category: {repo.get('category', 'Uncategorized')}.\n"
                f"Description (from README): {cleaned_readme.strip()}.\n"
                f"This workflow uses the following tools: {', '.join(unique_tool_names) if unique_tool_names else 'No tools listed'}.\n"
                f"You can download the raw workflow definition from: {chosen_raw_url or 'N/A'}.\n"
                f"End of workflow description."
            )

            preprocessed.append({
                "category": repo.get("category", ""),
                "workflow_repository": repo.get("workflow_repository", ""),
                "workflow_name": chosen_workflow_name or repo.get("workflow_repository", ""),
                "tool_names": unique_tool_names,
                "readme_cleaned": cleaned_readme,
                "raw_download_url": chosen_raw_url or "",
                "content": content,
            })

        return preprocessed


    # ---------------------------
    # Main scraping flow
    # ---------------------------
    async def scrape_workflows(self):
        """
        Scrape Galaxy workflows from the IWC GitHub repository, preprocess them in-memory,
        and save a single preprocessed JSON file suitable for downstream LLM or embedding use.
        """
        self.log.info(" Starting Galaxy IWC workflow scraping...")

        try:
            categories = await self.github_api_get(self.github_API_URL)
        except Exception as e:
            self.log.error(f"Failed to fetch top-level categories: {e}")
            return

        all_data = []
        workflow_count = 0

        # Iterate through all top-level directories (categories)
        for category_item in categories:
            if category_item.get("type") != "dir":
                continue

            category = category_item.get("name")
            self.log.info(f"📂 Scanning category: {category}")

            try:
                repos = await self.github_api_get(category_item.get("url"))
            except Exception as e:
                self.log.error(f"⚠️ Failed to list repos for category {category}: {e}")
                continue

            # Iterate through repositories inside category
            for repo in repos:
                if repo.get("type") != "dir":
                    continue

                repo_name = repo.get("name")
                repo_data = await self.scan_repo(category, repo_name)
                if repo_data:
                    all_data.append(repo_data)
                    workflow_count += len(repo_data.get("workflow_files", []))

                    if self.maximum_workflow_fetch and workflow_count >= self.maximum_workflow_fetch:
                        self.log.info(f"Reached maximum workflow fetch limit ({self.maximum_workflow_fetch}). Stopping.")
                        break

            if self.maximum_workflow_fetch and workflow_count >= self.maximum_workflow_fetch:
                break

        self.log.info(f"✅ Completed scraping {workflow_count} workflows across {len(all_data)} repositories.")

        # --- Single-step preprocessing and saving ---
        try:
            self.log.info("🧹 Preprocessing scraped workflows...")
            preprocessed = self.preprocess_scraped(all_data)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.raw_dir / f"galaxy_iwc_workflows_preprocessed_{timestamp}.json"

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(preprocessed, f, indent=2, ensure_ascii=False)

            self.log.info(f"📦 Preprocessed workflow data saved to: {output_path}")
            return str(output_path)

        except Exception as e:
            self.log.error(f"Error during preprocessing or saving: {e}")
            raise

                
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
    