import re
import os
import ast
from rapidfuzz import process, fuzz
import json
import numpy as np
import logging
import redis
import httpx
from dotenv import load_dotenv
from typing import Dict, Any, Union
from urllib.parse import quote
import sys
import asyncio

load_dotenv()
sys.path.append('.')

from app.log_setup import configure_logging
from app.bioblend_server.galaxy import GalaxyClient
from app.AI.provider.gemini_provider import GeminiProvider
from app.AI.provider.openai_provider import OpenAIProvider
from app.bioblend_server.informer.prompts import RETRIEVE_PROMPT, SELECTION_PROMPT, EXTRACT_KEY_WORD, FINAL_RESPONSE_PROMPT
from app.AI.llm_config._base_config import LLMModelConfig


class GalaxyInformer:
    """
    A Tool to retrieve and summarize information about Galaxy
    entities (tools, workflows, datasets) using a combination of bioblend based API calls,
    caching, fuzzy search, and Retrieval-Augmented Generation (RAG).
    """
    def __init__(self, galaxy_client: GalaxyClient, entity_type: str):
        """Initializes the GalaxyInformer with non-blocking assignments."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.entity_type = entity_type.lower()
        self.galaxy_client = galaxy_client
        self.gi_user = self.galaxy_client.gi_client
        self.gi_admin = self.galaxy_client.gi_admin
        self.username = self.galaxy_client.whoami
        self.llm = None
        self.redis_client = None
        self.manager = None
        self._entity_config = {
            'dataset': {
                'get_method': self._get_datasets,
                'search_fields': ['name', 'dataset_id'],
                'id_field': 'dataset_id'
            },
            'tool': {
                'get_method': self._get_tools,
                'search_fields': ['name', 'tool_id'],
                'id_field': 'tool_id'
            },
            'workflow': {
                'get_method': self._get_workflows,
                'search_fields': ['name', 'workflow_id'],
                'id_field': 'workflow_id'
            }
        }
        self.logger.info(f'Initializing the galaxy informer for entity type: {entity_type} for user {self.username}')

    @classmethod
    async def create(cls, galaxy_client: GalaxyClient, entity_type: str, llm_provider = os.getenv("CURRENT_LLM", "gemini")):
        """Asynchronous factory to create and fully initialize a GalaxyInformer instance."""

        from app.bioblend_server.informer.manager import InformerManager 
        
        self = cls(galaxy_client, entity_type)

        configure_logging()
        
        with open('app/AI/llm_config/llm_config.json', 'r') as f:
            model_config_data = json.load(f)
        
        if llm_provider == "gemini":
            gemini_cfg = LLMModelConfig(model_config_data['providers']['gemini'])
            self.llm = GeminiProvider(model_config=gemini_cfg)
        elif llm_provider == "openai":
            openai_cfg = LLMModelConfig(model_config_data['providers']['openai'])
            self.llm = OpenAIProvider(model_config=openai_cfg)
            
        self.redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=os.getenv("REDIS_PORT"), db=0, decode_responses=True)
        self.manager = await InformerManager.create()
        return self

    async def get_embedding_model(self, input):
        with open('app/AI/llm_config/llm_config.json', 'r') as f:
            model_config_data = json.load(f)
        openai_cfg = LLMModelConfig(model_config_data['providers']['openai'])
        llm = OpenAIProvider(model_config=openai_cfg)
        return await llm.embedding_model(input)
    
    async def get_response(self, message):
        # Accept either a raw string or already-formatted list[dict]
        if isinstance(message, str):
            message = [{"role": "user", "content": message}]
        return await self.llm.get_response(message)
    
    async def run_get_request(self, url, headers, params):
        """
        Sends an asynchronous HTTP GET request to the specified URL with the given headers and query parameters.
        """
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url=url, headers=headers, params=params)
            response.raise_for_status()
        return response.json()
    
    async def run_post_request(self, url, headers=None, data=None, json_data=None, params=None):
        """
        Makes an asynchronous HTTP POST request using httpx.
        """
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                url=url,
                headers=headers,
                data=data,
                json=json_data,
                params=params
            )
            response.raise_for_status()
            return response.json()
    
    def extract_filename(self, path: str) -> str:
        """
        Extracts a filename from a full path string.
        """
        self.logger.info(f'Extracting file name from path.')
        match = re.search(r'([^/]+)$', path)
        return match.group(1) if match else path

    def _get_datasets(self) -> list[dict]:
        """
        Retrieves all datasets from both Galaxy libraries and histories.
        """
        dataset_list = []
        self.logger.info('Gathering datasets from libraries...')
        try:
            libraries = self.gi_user.libraries.get_libraries()
            for library in libraries:
                library_details = self.gi_user.libraries.show_library(library['id'], contents=True)
                for lib in library_details:
                    if lib['type'] == 'file':
                        name = self.extract_filename(lib['name'])
                        dataset_list.append({
                            "dataset_id": lib['id'],
                            "name": name,
                            "full_path": lib['name'],
                            "type": lib["type"],
                            "source": "library",
                            "content": (
                                        f"Library dataset titled '{name}' (Galaxy ID: {lib['id']}) "
                                        f"stored under path '{lib['name']}' in the data galaxy instance library. "
                                        f"Dataset type is '{lib['type']}', indicating a Galaxy data file (e.g., FASTQ, BAM, VCF, TXT, etc.). "
                                        f"This file originates from a curated library, suggesting reuse in standardized workflows or reference analyses. "
                                     )
                            })
        except Exception as e:
            self.logger.error(f"Failed to retrieve library datasets: {e}")

        self.logger.info('Gathering datasets from histories...')
        try:
            # Note: get_datasets() can be slow. Consider filtering if possible.
            for data in self.gi_user.datasets.get_datasets(deleted=False, state='ok'):
                if data['type'] == 'file':
                    dataset_list.append({
                        "dataset_id": data['id'],
                        "name": data['name'],
                        "full_path": data['url'],
                        "type": data['type'],
                        "source": "history",
                        "content": (
                                    f"History dataset titled '{data['name']}' (Galaxy ID: {data['id']}) retrieved from user history. "
                                    f"Located at URL '{data['url']}'. Type: '{data['type']}', representing a file in a Galaxy history."
                                    f"This dataset likely resulted from a prior step in a bioinformatics workflow, tool execution output, or an uploaded to the history by the user."
                                )
                    })
        except Exception as e:
            self.logger.error(f"Failed to retrieve history datasets: {e}")
            
        return dataset_list

    def _get_tools(self) -> list[dict]:
        """  Retrieves all tools available in the Galaxy instance.  """

        def is_data_manager_tool(tool_dict, threshold=90) -> bool:
            """ Detects if a Galaxy tool is likely a data manager tool. """
            fields_to_check = [
                tool_dict.get('id', ''),
                tool_dict.get('name', ''),
                tool_dict.get('config_file', ''),
            ]

            # Combine all text into one lowercase string
            combined_text = ' '.join(fields_to_check).lower()

            # First, use regex to quickly check for obvious matches
            if re.search(r'data.*manager|manager.*data', combined_text):
                return True

            # Fuzzy check separately for "data" and "manager"
            has_data = fuzz.partial_ratio("data", combined_text) >= threshold
            has_manager = fuzz.partial_ratio("manager", combined_text) >= threshold

            if has_data and has_manager:
                return True
            else:
                return False
   
        tools=[]
        self.logger.info('Gathering all available tools...')
        for tool in self.gi_user.tools.get_tools():
            tool_type="data manager" if is_data_manager_tool(tool) else "Regular"
            tool_info={
                    'description': tool.get('description', None),
                    'tool_id': tool['id'],
                    'name': tool['name'],
                    'tool_type': tool_type,
                    'content': (
                                f"Galaxy tool named '{tool['name']}' (ID: {tool['id']}) designed to perform a specific function within a bioinformatics workflow. "
                                f"Tool description: '{tool.get('description', 'No description provided')}'. "
                            )
                             }
            if tool_type == "data manager":
                tool_info['content'] += (
                           f'This Galaxy tool named {tool_info["name"]} is a Galaxy Data Manager tool, meaning it is designed to either create, modify, or manage reference datasets that are used by other tools within the Galaxy platform.'
                    )
            elif tool_type == "Regular" :
                tool_info['content'] += (f'This Galaxy tool named  {tool_info["name"]} is a regular galaxy tool')

            tools.append(tool_info)

        return tools

    def _get_workflows(self) -> list[dict]:
        """
        Retrieves all published workflows in the Galaxy instance.
        """
        self.logger.info('Gathering all published workflows...')
        workflows = []
        for wf in self.gi_user.workflows.get_workflows(published=True):
             desc = str(wf.get('annotations') or wf.get('description') or 'No description available.')
             workflows.append({
                'description': desc,
                'model_class': wf['model_class'],
                'owner': wf['owner'],
                'workflow_id': wf['id'],
                'name': wf['name'],
                'content': (
                    f"Galaxy workflow titled '{wf['name']}' (ID: {wf['id']}), authored by user '{wf['owner']}'. "
                    f"This is a workflow composed of multiple bioinformatics tools connected in a pipeline. "
                    f"Workflow class: '{wf['model_class']}'. Purpose: {desc}. "
                    f"Workflows typically automate multi-step processes."
                )
            })
        return workflows

    def get_all_entities(self) -> list[dict]:
        """
        Public method to get all entities based on the configured type.
        """
        return self._entity_config[self.entity_type]['get_method']()

    async def _semantic_search(self, query: str, collection_name: str) -> dict:
        """
        Performs semantic search using the integrated vector manager.
        """
        self.logger.info(f"Performing semantic search for query: '{query}'")
        try:
            if isinstance(query, str):
                query=[query]
            embeddings = await self.get_embedding_model(query)
            embed = np.array(embeddings)
            query_embedding = embed.reshape(-1, 1536).tolist()[0]
            # Search vector database
            results = self.manager.search_by_vector(
                collection=collection_name,
                query_vector=query_embedding,
                entity_type=self.entity_type
            )
            return {k: results[k] for k in sorted(results.keys())[:10]}
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return {}

    def _parse_list_from_llm(self, list_str: str) -> list:
        """
        Extracts and safely parses a Python list from an LLM-generated string.
        """
        try:
            # Remove markdown code block markers (e.g., ```python ... ```)
            cleaned_str = re.sub(r"^\s*```(?:python)?\s*|\s*```\s*$", "", list_str.strip(), flags=re.IGNORECASE | re.MULTILINE)
            parsed = ast.literal_eval(cleaned_str.strip())
            return parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError) as e:
            self.logger.error(f"Could not parse list from LLM output: {e}")
            return []

    async def _extract_fuzzy_search_keywords(self, query: str) -> list:
        """
        Uses an LLM to extract relevant keywords for a fuzzy search.
        """
        prompt = EXTRACT_KEY_WORD.format(query = query)
        try:
            keywords_str = await self.get_response(prompt)
            self.logger.info("Extracted keywords for fuzzy search.")
            result_list = keywords_str if isinstance(keywords_str, list) else self._parse_list_from_llm(keywords_str)
            return result_list
        except Exception as e:
            self.logger.error(f"Failed to generate keywords from LLM: {e}")
            return [query] # Fallback to the original query

    def _fuzzy_search(self, query: str, entities: list[dict], config: list, threshold=85) -> list:
        """Fuzzy search with priority fields, improved for efficiency and clarity."""

        self.logger.info('Fuzzy search for the query by priority fields')
        
        # Prepare priority candidates
        priority_candidates = []
        entity_map = {}  # Map: field string -> entity
        
        for entity in entities:
            for field in config['search_fields']:
                if field in entity and isinstance(entity[field], str):
                    field_value = entity[field]
                    priority_candidates.append(field_value)
                    entity_map[field_value] = entity
        
        # Run fuzzy match across all priority fields at once
        results = process.extract(query, priority_candidates, scorer=fuzz.WRatio, limit=10)
        
        # Filter by threshold and collect matches
        priority_matches = []
        for match_str, score, _ in results:
            if score >= threshold:
                priority_matches.append((entity_map[match_str], score))
        
        if priority_matches:
            self.logger.info(f'Found {len(priority_matches)} matches in priority fields for search query: {query}')
            return sorted(priority_matches, key=lambda x: x[1], reverse=True)[:5]
        
        # No good priority matches found → fallback to all other fields
        self.logger.info(f'No matches found in the priority fields for search query: {query}, searching in all fields as a fallback')
        
        fallback_candidates = []
        fallback_map = {}
        
        for entity in entities:
            for key, value in entity.items():
                if key not in config['search_fields'] and isinstance(value, str):
                    fallback_candidates.append(value)
                    fallback_map[value] = entity
        
        results = process.extract(query, fallback_candidates, scorer=fuzz.WRatio, limit=10)
        
        fallback_matches = []
        for match_str, score, _ in results:
            if score >= threshold:
                fallback_matches.append((fallback_map[match_str], score))
        
        if fallback_matches:
            self.logger.info(f'Found {len(fallback_matches)} matches in fallback fields for search query: {query}')
            return sorted(fallback_matches, key=lambda x: x[1], reverse=True)[:5]
        
        self.logger.info(f'No matches found in either priority or fallback fields for search query: {query}')
        return []  # Always return a list, even if empty


    async def refresh_and_cache_entities(self) -> list[dict]:
        """
        Fetches fresh entity data from Galaxy, then caches it in Redis and
        updates the Qdrant vector store.
        """
        self.logger.info(f'No valid cache found for {self.entity_type}, fetching fresh data.')
        entities = self.get_all_entities()

        # If Nothing is found in the galaxy return the emoty result
        if not entities:
            return None
        
        if self.entity_type  == "tool":
            collection_name = f'Galaxy_{self.entity_type}'
        else:
            collection_name = f'Galaxy_{self.entity_type}_{self.username}'
        
        # Cache in Redis with a 10-hour TTL
        try:
            if self.entity_type  == "tool":
                self.redis_client.setex(collection_name, 86400, json.dumps(entities))
                self.logger.info(f'Saved {self.entity_type} entities to Redis.')
                
            elif self.entity_type  == "workflow":
                self.redis_client.setex(collection_name, 18000, json.dumps(entities))
                self.logger.info(f'Saved {self.entity_type} entities to Redis.')
                
            elif self.entity_type == "dataset":
                self.redis_client.setex(collection_name, 3600, json.dumps(entities))
                self.logger.info(f'Saved {self.entity_type} entities to Redis.')
                
        except redis.RedisError as e:
            self.logger.error(f'Failed to save entities to Redis: {e}')
        except Exception as e:
            self.logger.error(f'Failed to save entities to Redis: {e}')
        
        # Store in Qdrant vector database
        try:
            # Delete old collection adn add to new collection.
            self.manager.delete_collection(collection_name=collection_name)
            await self.manager.embed_and_store_entities(
                entities=entities,
                collection_name=collection_name
            )
            self.logger.info(f'Saved {self.entity_type} entities to Qdrant.')
        except Exception as e:
            self.logger.error(f'Failed to save entities to Qdrant: {e}')
            
        return entities

    async def get_cached_or_fresh_entities(self) -> list[dict]:
        """
        Attempts to retrieve entities from Redis cache, falling back to fetching
        fresh data if the cache is empty or invalid.
        """
        try:
            if self.entity_type == "tool":
                entities_str = self.redis_client.get(f'Galaxy_{self.entity_type}')
            else:
                entities_str = self.redis_client.get(f'Galaxy_{self.entity_type}_{self.username}')
            if entities_str:
                self.logger.info(f'Retrieved cached {self.entity_type} entities from Redis.')
                return json.loads(entities_str)
        except (redis.RedisError, json.JSONDecodeError) as e:
            self.logger.error(f'Redis cache retrieval failed: {e}. Fetching fresh data.')
        
        return await self.refresh_and_cache_entities()

    async def search_entities(self, query: str, threshold=85) -> dict:
        """
        Conducts a hybrid search using both fuzzy and semantic techniques,
        then uses an LLM to select and rank the best results.
        """
        entities = await self.get_cached_or_fresh_entities()

        if not entities:
            return None

        # 1. Prepare collection name for semantic search
        if self.entity_type == "tool":
            collection_name = f'Galaxy_{self.entity_type}'
        else:
            collection_name = f'Galaxy_{self.entity_type}_{self.username}'

        # 2. Extract keywords
        keywords = await self._extract_fuzzy_search_keywords(query)

        # Run fuzzy and semantic searches concurrently
        async def run_fuzzy_searches():
            """Run fuzzy searches concurrently for all keywords."""
            tasks = [
                asyncio.to_thread(
                    self._fuzzy_search,
                    keyword,
                    entities,
                    self._entity_config[self.entity_type],
                    threshold
                )
                for keyword in keywords
            ]
            fuzzy_batches = await asyncio.gather(*tasks)
            # Flatten all fuzzy results into one list
            fuzzy_results = [item for batch in fuzzy_batches for item in batch]
            # Remove duplicates
            unique_fuzzy_results = list({
                item[f'{self._entity_config[self.entity_type]["id_field"]}']: item
                for item, score in fuzzy_results
            }.values())
            return unique_fuzzy_results

        # Launch both searches concurrently
        fuzzy_task = asyncio.create_task(run_fuzzy_searches())
        semantic_task = asyncio.create_task(self._semantic_search(query=query, collection_name=collection_name))

        # Wait for both to complete
        unique_fuzzy_results, semantic_results = await asyncio.gather(fuzzy_task, semantic_task)

        # 3. LLM-based Re-ranking and Selection (unchanged)
        self.logger.info("Using LLM to select the best results from hybrid search.")
        prompt = SELECTION_PROMPT.format(input=query, tuple_items=unique_fuzzy_results, dict_items=semantic_results)

        try:
            selected_str = await self.get_response(prompt)
            return selected_str
        except (json.JSONDecodeError, Exception) as e:
            self.logger.error(f"Failed to select entities using LLM: {e}")
            combined = {v[self._entity_config[self.entity_type]['id_field']]: v for k, v in semantic_results.items()}
            for item in unique_fuzzy_results:
                combined[item[self._entity_config[self.entity_type]['id_field']]] = item
            return dict(list(combined.items())[:3])

        
    async def _show_tool(self, tool_id):
        """
        Replacement for bioblends show_tool function for richer information retrieval, 
        making direct get http request to the galaxy api
        """
        headers = {'x-api-key' : self.galaxy_client.user_api_key}
        url = f'{self.galaxy_client.galaxy_url}/api/tools/{tool_id}/build'
        params= {'history_id': self.gi_user.histories.get_histories()[-1]['id']}

        try:
            tool = await self.run_get_request(url=url, headers=headers, params=params)
        except Exception as e:
            self.logger.error(f'error fetching toolds via direct api call: {e}')
            raise 
        return tool

    def _prune_empty_nested(self, obj: Union[Dict[str, Any], list]) -> Union[Dict[str, Any], list]:
        """Recursively prune empty key-value pairs, lists, and dicts from nested structures."""
        
        if isinstance(obj, dict):
            to_remove = []
            for k, v in list(obj.items()):
                if isinstance(v, (dict, list)):
                    self._prune_empty_nested(v)  # Recurse first
                    if isinstance(v, (dict, list)) and len(v) == 0:
                        to_remove.append(k)
                elif v is None or (isinstance(v, str) and not v.strip()):
                    to_remove.append(k)
            for k in to_remove:
                obj.pop(k, None)
        elif isinstance(obj, list):
            obj[:] = [item for item in (self._prune_empty_nested(item) for item in obj) if item not in [None, '', [] , {}]]
        return obj

    async def _clean_tool_metadata(self, raw_tool: Dict[str, Any]) -> Dict[str, Any]:
        """Clean tool metadata for RAG compatibility."""

        tool = raw_tool.copy()

        # Step 1: Drop only junk (UI/runtime/internal keeps all semantics)
        universal_drops = [
            'model_class', 'icon', 'hidden', 'config_file', 'panel_section_id', 'form_style',
            'sharable_url', 'message', 'tool_errors', 'job_id', 'job_remap', 'history_id',
            'display', 'action', 'method', 'enctype', 'help_format'
        ]
        for field in universal_drops:
            tool.pop(field, None)
        
        # Step 1.5: Recursively prune empty nested structures
        self._prune_empty_nested(tool)
        
        # Step 2: Enrich inputs
        if 'inputs' in tool:
            enriched_inputs = []
            for inp in tool['inputs']:
                # Keep all core fields; drop only per-input UI junk
                input_junk = ['model_class', 'argument', 'help_format', 'refresh_on_change', 'hidden', 
                            'is_dynamic', 'tag', 'text_value']  # These are form-specific
                for junk in input_junk:
                    inp.pop(junk, None)
                
                if 'optional' in inp:
                    inp['required'] = not inp['optional']
                
                # flatten options lightly
                if 'options' in inp and isinstance(inp['options'], list):

                    flattened_opts = []
                    for opt in inp['options']:
                        if isinstance(opt, list) and len(opt) >= 2:
                            flattened_opts.append({'label': opt[0], 'value': opt[1], 'selected': opt[2] if len(opt) > 2 else False})
                        else:
                            flattened_opts.append(opt)
                    inp['options'] = flattened_opts
                
                if 'edam' in inp and inp['edam']:
                    pass
                
                enriched_inputs.append(inp)
            tool['inputs'] = enriched_inputs
        
        # Step 3: Full help preservation (strip HTML/tags for clean text, no truncation)
        if 'help' in tool:
            help_text = tool['help']

            clean_help = re.sub(r'<[^>]+>', '', help_text)  # Basic tag removal
            clean_help = re.sub(r'\n\s*\n', '\n\n', clean_help.strip())  # Normalize whitespace
            tool['help_text'] = clean_help
            del tool['help']
               
        # Step 5: Add universal derived fields
        tool['tool_id'] = tool.get('id', 'unknown')
        tool['tool name'] = tool.get('name', 'unknown')
        if 'name' in tool:
            del tool['name']
            
        if 'id' in tool:
            del tool['id']
        tool['category'] = tool.get('panel_section_name', tool.get('labels', ['General'])[0])
        
        tool_id = tool['tool_id']
        
        # Add link to for execution.
        tool_url = quote(tool_id, safe = "")
        tool['Link to execute tool'] = f"{self.galaxy_client.galaxy_url}/?tool_id={tool_url}&version=latest"
        
        if 'panel_section_name' in tool:
            del tool['panel_section_name']
              
        # Flatten arrays for brevity if huge
        for k in ['versions', 'requirements', 'labels', 'xrefs', 'edam_operations', 'edam_topics']:
            if k in tool and isinstance(tool[k], list) and len(tool[k]) > 10:
                tool[k + '_truncated'] = tool[k][:10] + ['...']
        
        return tool
    
    def _clean_workflow_metadata(self, raw_workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and enrich workflow metadata for RAG compatibility."""
        
        workflow = raw_workflow.copy() 
        
        # Step 1: Remove universal non-semantic fields
        universal_drops = [
            'model_class', 'id', 'create_time', 'update_time', 'url', 'published', 'deleted', 'hidden',
            'owner', 'latest_workflow_uuid', 'number_of_steps', 'show_in_tool_panel', 'creator_deleted',
            'doi', 'email_hash', 'readme', 'help', 'slug', 'source_metadata', 'name'
        ]
        for field in universal_drops:
            workflow.pop(field, None)
        
        # Step 1.5: Recursively prune empty nested structures across the entire workflow
        self._prune_empty_nested(workflow)
        
        # Step 2: Enrich inputs by converting dictionary to sorted list, adding derived fields, and removing junk
        if 'inputs' in workflow and isinstance(workflow['inputs'], dict):
            enriched_inputs = []
            for key, inp in sorted(workflow['inputs'].items(), key=lambda x: int(x[0])):
                # Remove input-specific non-semantic fields
                input_junk = ['uuid', 'value']
                for junk in input_junk:
                    inp.pop(junk, None)
                
                # Derive 'required' field based on 'optional' if present
                inp['required'] = not inp.get('optional', False)
                
                # Prune the individual input after modifications
                self._prune_empty_nested(inp)
                
                enriched_inputs.append(inp)
            workflow['inputs'] = enriched_inputs
        
        # Step 3: Clean annotation by stripping HTML tags, normalizing whitespace, and renaming to 'description'
        if 'annotation' in workflow:
            annot_text = workflow['annotation']
            # Remove HTML tags
            clean_annot = re.sub(r'<[^>]+>', '', annot_text)
            # Normalize multiple newlines and strip leading/trailing whitespace
            clean_annot = re.sub(r'\n\s*\n', '\n\n', clean_annot.strip())
            if 'description' not in workflow:
                workflow['description'] = clean_annot
            else:
                workflow['description'] += f" {clean_annot}"
                
            workflow.pop('annotation', None)
        
        
        # Step 4: Enrich steps by converting dictionary to sorted list, cleaning tool_inputs, and adding derived count
        if 'steps' in workflow and isinstance(workflow['steps'], dict):
            enriched_steps = []
            for step_id, step in sorted(workflow['steps'].items(), key=lambda x: int(x[0])):
                
                # Remove step-specific non-semantic fields
                step_junk = ['when', '__page__', '__rerun_remap_job_id__']
                for junk in step_junk:
                    step.pop(junk, None)
                               
                # For subworkflows, recursively clean the metadata
                if step.get('type') == 'subworkflow':
                    step = self._clean_workflow_metadata(step)
                
                # Prune the individual step after modifications
                self._prune_empty_nested(step)
                
                # structure tool info better
                if step.get('type') == 'tool':
                    
                    if 'tool_id' in step and step['tool_id']:
                        parts = step['tool_id'].split('/')
                        if len(parts) >= 5:  # Minimum for toolshed/repos/owner/repo/tool structure
                            tool_info = {
                                'full_path': step['tool_id'],
                                'owner': parts[-4],
                                'repo': parts[-3],
                                'name': parts[-2],
                            }
                            # Add version and inputs
                            if 'tool_version' in step:
                                tool_info['version'] = re.sub(r'\+galaxy\d+$', '', step['tool_version'])
                            if 'tool_inputs' in step:
                                tool_info['tool_inputs'] = step['tool_inputs']
                            step['tool_info'] = tool_info
                    
                    step.pop('tool_id', None)
                    step.pop('tool_version', None)
                    step.pop('tool_inputs', None)
                
                enriched_steps.append(step)
            workflow['steps'] = enriched_steps
            # Add derived field for step count
            workflow['number_of_steps'] = len(enriched_steps)
        
        # Step 5: Clean creator list by removing non-essential fields per person and pruning
        if 'creator' in workflow and isinstance(workflow['creator'], list):
            for person in workflow['creator']:
                # Remove person-specific non-essential fields
                person_junk = [
                    'class', 'address', 'alternateName', 'email', 'faxNumber', 'image', 'telephone',
                    'url', 'familyName', 'givenName', 'honorificPrefix', 'honorificSuffix', 'jobTitle'
                ]
                for junk in person_junk:
                    person.pop(junk, None)
            # Prune the creator list after individual modifications
            self._prune_empty_nested(workflow['creator'])
        
        # Step 6: Add derived fields for enhanced metadata
        workflow['workflow_id'] = raw_workflow.get('id', 'unknown')
        workflow['workflow name'] = raw_workflow.get('name', 'unknown')
        workflow['category'] = workflow.get('tags', ['General'])[0] if 'tags' in workflow else 'General'
        
        # Add link to workflow for execution.
        workflow_id = workflow['workflow_id']
        if workflow_id != 'unknown':
            workflow['Link to execute workflow'] = f"{self.galaxy_client.galaxy_url}/workflows/run?id={workflow_id}"
                      
        # Final prune to ensure no lingering empty structures
        self._prune_empty_nested(workflow)
        
        return workflow

    def _show_workflow(self, workflow_id):
        """Retrieve and clean workflow metadata"""
        workflow = self.gi_user.workflows.show_workflow(workflow_id)
        # Clean metadata of workflow
        cleaned_workflow_data = self._clean_workflow_metadata(workflow)
        
        return json.dumps(cleaned_workflow_data, ensure_ascii=False)
    
    async def _show_tool(self, tool_id):
        """
        Replacement for bioblends show_tool function for richer information retrieval, 
        making direct get http request to the galaxy api, nd then cleaning retrieved metadata.
        """
        headers = {'x-api-key' : self.galaxy_client.user_api_key}
        url = f'{self.galaxy_client.galaxy_url}/api/tools/{tool_id}/build'
        params= {'history_id': self.gi_user.histories.get_histories()[-1]['id']}

        try:
            tool = await self.run_get_request(url=url, headers=headers, params=params)
            cleaned_tool_data = await self._clean_tool_metadata(tool)
        except Exception as e:
            self.logger.error(f'error fetching toolds via direct api call: {e}')
            raise 
        return  json.dumps(cleaned_tool_data, ensure_ascii=False)
    
    def _show_dataset(self, dataset_id):
        dataset = self.gi_user.datasets.show_dataset(dataset_id)
        return json.dumps(dataset, ensure_ascii=False)

    async def get_entity_details(self, entity_id: str) -> dict:
        detail_methods = {
            'dataset': self._show_dataset,
            'tool': self._show_tool,
            'workflow': self._show_workflow
        }

        try:
            self.logger.info(f"Fetching details for {self.entity_type} with ID: {entity_id}")
            method = detail_methods[self.entity_type]

            if asyncio.iscoroutinefunction(method):
                return await method(entity_id)
            return method(entity_id)
        
        except Exception as e:
            self.logger.error(f"Could not retrieve details for {self.entity_type}:{entity_id}: {e}")
            return {"error": "Failed to retrieve details."}

    async def generate_final_response(self, query: str, retrieved_contents: list):
        """ Generates a final, user-facing natural language response based on the retrieved and processed information. """
        
        async def content_response(query, content): 
            prompt = RETRIEVE_PROMPT.format(query=query, retrieved_content=content)
            response_text = await self.get_response(prompt)
            self.logger.info(f"Context response: {response_text} \n\n")
            return response_text
        
        tasks = [content_response(query, content) for content in retrieved_contents]
        task_results = await asyncio.gather(*tasks)
        query_responses = "\n\n\n\n".join(str(r) for r in task_results if r)

        self.logger.info('Generating final response.')
        prompt = FINAL_RESPONSE_PROMPT.format(query=query, query_responses=query_responses)
        response_text = await self.get_response(prompt)
        self.logger.info(f"Final response: {response_text}")
        return response_text


    async def get_entity_info(self, search_query: str, entity_id: str = None) -> dict:
        """
        The main public method to orchestrate the entire information retrieval process.
        """
        id_field = self._entity_config[self.entity_type]['id_field']
        found_entities=None
        if entity_id:
            self.logger.info(f'Direct ID provided: {entity_id}. Retrieving details directly.')
            entities = await self.get_cached_or_fresh_entities()
            # Validate the id against the entity id in galaxy instance
            retrived_entity = next((e for e in entities if e[id_field] == entity_id), None)
            if retrived_entity!=None:
                found_entities={'0': {'name': retrived_entity['name'], id_field : retrived_entity[id_field] }}
            # else: 
            #     raise 

        if found_entities is None:
            self.logger.info(f"No ID provided or invalid ID. Searching for entities based on query: '{search_query}'")
            found_entities = await self.search_entities(query=search_query)

        if found_entities is None:
            return await self.generate_final_response(search_query, {"message": "No relevant items found."})
        
        self.logger.info(f"Relevant {self.entity_type}s found are: {[v.get('name') for v in found_entities.values()]}")

        tasks = []

        for _, item_stub in enumerate(found_entities.values()):
            item_id = item_stub.get(id_field)
            if item_id:
                tasks.append(self.get_entity_details(item_id))
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return await self.generate_final_response(search_query, results)