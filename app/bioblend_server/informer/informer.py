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
import sys

load_dotenv()
sys.path.append('.')

from app.log_setup import configure_logging
from app.bioblend_server.galaxy import GalaxyClient
from app.AI.provider.gemini_provider import GeminiProvider
from app.bioblend_server.informer.prompts import RETRIEVE_PROMPT, SELECTION_PROMPT, INVOCATION_PROMPT, EXTRACT_KEY_WORD
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
        self.user_info = self.galaxy_client.whoami()
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
        self.logger.info(f'Initializing the galaxy informer for entity type: {entity_type} for user {self.user_info["id"]}')

    @classmethod
    async def create(cls, galaxy_client: GalaxyClient, entity_type: str):
        """Asynchronous factory to create and fully initialize a GalaxyInformer instance."""

        from app.bioblend_server.informer.manager import InformerManager 
        
        self = cls(galaxy_client, entity_type)

        configure_logging()
        
        with open('app/AI/llm_config/llm_config.json', 'r') as f:
            model_config_data = json.load(f)

        gemini_cfg = LLMModelConfig(model_config_data['providers']['gemini'])
        self.llm = GeminiProvider(model_config=gemini_cfg)
        self.redis_client = redis.Redis(host='localhost', port=os.getenv("REDIS_PORT"), db=0, decode_responses=True)
        self.manager = await InformerManager.create()
        return self

    async def get_embedding_model(self, input):
        return await self.llm.gemini_embedding_model(input)
    
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
            query_embedding = embed.reshape(-1, 768).tolist()[0]
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
            return self._parse_list_from_llm(keywords_str)
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
        
        collection_name = f'Galaxy_{self.entity_type}_{self.user_info["id"]}'
        
        # Cache in Redis with a 10-hour TTL
        try:
            self.redis_client.setex(collection_name, 36000, json.dumps(entities))
            self.logger.info(f'Saved {self.entity_type} entities to Redis.')
        except redis.RedisError as e:
            self.logger.error(f'Failed to save entities to Redis: {e}')
        
        # Store in Qdrant vector database
        try:
            # --- Calls to the new vector manager ---
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
            entities_str = self.redis_client.get(f'Galaxy_{self.entity_type}_{self.user_info["id"]}')
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
        
        # 1. Fuzzy Search
        keywords = await self._extract_fuzzy_search_keywords(query)
        fuzzy_results = []
        for keyword in keywords:
            fuzzy_results.extend(self._fuzzy_search(query=keyword, entities=entities, config=self._entity_config[self.entity_type], threshold=threshold))
        # Remove duplicates
        unique_fuzzy_results = list({item[f'{self._entity_config[self.entity_type]["id_field"]}']: item for item, score in fuzzy_results}.values())

        # 2. Semantic Search
        collection_name = f'Galaxy_{self.entity_type}_{self.user_info["id"]}'
        semantic_results = await self._semantic_search(query=query, collection_name=collection_name)
        
        # 3. LLM-based Re-ranking and Selection
        self.logger.info("Using LLM to select the best results from hybrid search.")
        prompt = SELECTION_PROMPT.format(input=query, tuple_items=unique_fuzzy_results, dict_items=semantic_results)
        
        try:
            selected_str = await self.get_response(prompt)
            # The selection prompt should ask for a JSON object of the top relevant results
            return selected_str
        except (json.JSONDecodeError, Exception) as e:
            self.logger.error(f"Failed to select entities using LLM: {e}")
            # Fallback: combine and return top results manually if LLM fails
            combined = {v[self._entity_config[self.entity_type]['id_field']]:v for k,v in semantic_results.items()}
            for item in unique_fuzzy_results:
                combined[item[self._entity_config[self.entity_type]['id_field']]] = item
            return dict(list(combined.items())[:3])
        
    def _show_tool(self, tool_id):
        """
        Replacement for bioblends show_tool function for richer information retrieval, 
        making direct get http request to the galaxy api
        """
        headers = {'x-api-key' : self.galaxy_client.user_api_key}
        url = f'{self.galaxy_client.galaxy_url}/api/tools/{tool_id}/build'
        params= {'history_id': self.gi_user.histories.get_histories()[-1]['id']}

        try:
            tool = asyncio.run(self.run_get_request(url=url, headers=headers, params=params))
        except Exception as e:
            self.logger.error(f'error fetching toolds via direct api call: {e}')
            raise 
        return tool


    def get_entity_details(self, entity_id: str) -> dict:
        """
        Retrieves full details for a single entity by its ID.
        """

        detail_methods = {
            'dataset': self.gi_user.datasets.show_dataset,
            'tool': lambda id: self._show_tool(tool_id=id),
            'workflow': self.gi_user.workflows.show_workflow
        }
        
        try:
            self.logger.info(f"Fetching details for {self.entity_type} with ID: {entity_id}")
            return detail_methods[self.entity_type](entity_id)
        except Exception as e:
            self.logger.error(f"Could not retrieve details for {self.entity_type}:{entity_id}: {e}")
            return {"error": "Failed to retrieve details."}


    async def get_invocation_details(self, query: str, workflow_id: str = None):
        """Function to check and extract specific invocation details from the galaxy instance"""

        invocation_result = None
        
        def is_close_to_invocation(query: str) -> bool:
            query_words = query.split()
            target_word = "invocation"
            reference_word = "invoke"

            # Create a threshold from the ratio of the two words
            threshold = fuzz.ratio(target_word, reference_word)  # similarity score

            # Check each word against 'invocation'
            for word in query_words:
                score = fuzz.ratio(word, target_word)
                if score >= threshold:
                    return True
            return False
        try:
            self.logger.info('Checking if query is invocation related')
            if is_close_to_invocation(query=query):
                self.logger.info('Extracting invocation information')
                if workflow_id:
                    invocations=self.gi_user.invocations.get_invocations(workflow_id=workflow_id)
                else:
                    invocations=self.gi_user.invocations.get_invocations()

                # Prompt LLM to select the correct match for invocation.
                prompt=INVOCATION_PROMPT.format(query=query, invocations=invocations)
                response= await self.get_response(prompt)
                response=response.strip()
                self.logger.warning(f" the response of the LLM: {response}")

                if response == "No matches":
                    return None
                else:

                    # Extract and structure essential parts of invocation information

                    invocation_detail= self.gi_user.invocations.show_invocation(invocation_id=response)
                    invocation_report= self.gi_user.invocations.get_invocation_report(invocation_id=response)
                    invocation_report['messages']= invocation_detail['messages']
                    invocation_steps= [{
                                        'update_time': step.get('update_time', None),
                                        'job_id': step.get('job_id', None),
                                        'workflow_step_label': step.get('workflow_step_label', None),
                                        'order_index': step.get('order_index'),
                                        'state':step.get('state')
                                    } for step in invocation_detail.get('steps')]
                    invocation_inputs=[{
                                        'id': value.get('id'),
                                        'label': value.get('label'),
                                        'src': value.get('src')
                                        } for key,value in invocation_detail.get('inputs', None).items() ]
                    invocation_input_step_parameters=invocation_detail.get('input_step_parameters', None)
                    invocation_outputs=[key for key,value in invocation_detail.get('outputs', None).items()]
                    invocation_output_collection= [key for key,value in invocation_detail.get('output_collections', None).items()]
                    

                    invocation_result={
                        'invocation id': response,
                        'workflow id': workflow_id,
                        'invocation inputs': invocation_inputs,
                        'invocation input step parameters': invocation_input_step_parameters,
                        'invocation steps': invocation_steps,
                        'invocation outputs' : invocation_outputs,
                        'invocation collection outputs': invocation_output_collection,
                        'invocation final report' : invocation_report
                    }
        except Exception as e:
            self.logger.error(f"Error handling invocationss: {e}")
            raise e
            
        return invocation_result

    async def generate_final_response(self, query: str, retrieved_content: dict) -> dict:
        """
        Generates a final, user-facing natural language response based on the
        retrieved and processed information.
        """
        self.logger.info('Generating final response with LLM.')
        prompt = RETRIEVE_PROMPT.format(query=query, retrieved_content=retrieved_content)
        
        response_text = await self.get_response(prompt)
        
        return {
            'query': query,
            'retrieved_content': retrieved_content,
            'response': response_text
        }


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
        
        self.logger.info(f'found entities are : {found_entities} and are type {type(found_entities)}')
        # Check and retrieve invocation detail if query is asking about workflow invocation.
        
         # Collate all details
        response_details = {}

        if self.entity_type == 'workflow':
            invocation_info =await self.get_invocation_details(query = search_query, workflow_id= found_entities['0'][id_field])
            # register found information.
            if invocation_info:
                response_details['retrieved invocation details for the workflow'] = invocation_info
            else:
                response_details['retrieved invocation details for the workflow'] = "No matching invocation details found for the workflow "

        for i, item_stub in enumerate(found_entities.values()):
            item_id = item_stub.get(id_field)
            if item_id:
                details = self.get_entity_details(item_id)
                response_details[f'found {self.entity_type} {i}'] = details
        
        return await self.generate_final_response(search_query, response_details)



# To test the information retriever tool separetly.
if __name__ == "__main__":
    import asyncio
    async def main():
        user_api_key = "2b80f888032970d458302d74f6bff8ef" # demo galaxty api key
        galaxy_client = GalaxyClient(user_api_key)
        informer = await GalaxyInformer.create(galaxy_client, "tool")
        input_query = """Find a tool to convert gff files to bed"""
        information = await informer.get_entity_info(search_query=input_query)
                                                    #   entity_id='toolshed.g2.bx.psu.edu/repos/iuc/rgrnastar/rna_star/2.7.11a+galaxy1')

        print("--- Final Response ---")
        print(information['response'])      
    asyncio.run(main())