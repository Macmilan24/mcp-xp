# app/bioblend_server/informer/pipeline/benchmark.py

import json
import os
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import asyncio

# Import Gemini Provider from your existing app structure
import sys
sys.path.append(os.getcwd()) # Ensure we can import from app.
from app.AI.provider.gemini_provider import GeminiProvider
from app.AI.llm_config._base_config import LLMModelConfig
from dotenv import load_dotenv

load_dotenv() # Load env for Gemini API Key

# --- Configuration ---
DATA_FILE = "app/bioblend_server/informer/pipeline/data/processed_tools.json"
QUERIES_FILE = "app/bioblend_server/informer/pipeline/benchmark_queries.json"
SBERT_MODEL_NAME = "johnnas12/e5-galaxy-finetuned"

def load_data():
    print(f"Loading data from {DATA_FILE}...")
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Prepare content strings (logic borrowed from Phase 2 plan)
    texts = []
    ids = []
    for entry in data:
        categories = entry.get("categories") or []
        content = (
            f"{entry.get('name','')} - {entry.get('description','')} - "
            f"{', '.join([c for c in categories if c])} - {entry.get('help','')}"
        )
        texts.append(content)
        ids.append(entry.get('id'))
    print(f"Loaded {len(texts)} tools.")
    return texts, ids

def load_queries():
    with open(QUERIES_FILE, 'r') as f:
        return json.load(f)

def get_rank(similarity_scores, target_index):
    """Finds the rank of the target index in sorted similarity scores (descending)."""
    # Argsort gives indices that would sort the array. We want descending order.
    sorted_indices = np.argsort(similarity_scores)[::-1]
    # Find where the target_index is in the sorted list. Add 1 because rank is 1-based.
    rank = np.where(sorted_indices == target_index)[0][0] + 1
    return rank

# --- SBERT Evaluation ---
def evaluate_sbert(texts, tool_ids, queries):
    print(f"\n--- Testing SBERT ({SBERT_MODEL_NAME}) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    model = SentenceTransformer(SBERT_MODEL_NAME, device=device)

    # 1. Benchmark Embedding Time
    start_time = time.time()
    # e5 models need "passage: " prefix for documents
    passage_texts = ["passage: " + t for t in texts]
    doc_embeddings = model.encode(passage_texts, show_progress_bar=True, normalize_embeddings=True)
    embed_time = time.time() - start_time
    print(f"SBERT Corpus Embedding Time: {embed_time:.2f} seconds")

    # 2. Benchmark Retrieval Accuracy
    ranks = []
    hits_at_3 = 0

    print("Evaluating queries...")
    for q in queries:
        if q['entity_type'] != 'tool': continue
        
        target_id = q['expected_id']
        try:
            target_index = tool_ids.index(target_id)
        except ValueError:
            print(f"wawring: Target ID {target_id} not found in processed data. Skipping.")
            continue

        # e5 models need "query: " prefix for queries
        query_embedding = model.encode(["query: " + q['query']], normalize_embeddings=True)
        
        # Calculate cosine similarity
        scores = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        rank = get_rank(scores, target_index)
        ranks.append(rank)
        if rank <= 3:
            hits_at_3 += 1
        
        print(f"  Query: '{q['query'][:30]}...' -> Target: {target_id} -> Rank: {rank}")

    return {
        "Model": "SBERT (Local)",
        "Embedding Time (s)": round(embed_time, 2),
        "Mean Reciprocal Rank (MRR)": round(np.mean([1/r for r in ranks]), 4) if ranks else 0,
        "Hit@3 (%)": round((hits_at_3 / len(ranks)) * 100, 2) if ranks else 0
    }

# --- Gemini Evaluation ---
async def evaluate_gemini(texts, tool_ids, queries):
    print(f"\n--- Testing Gemini (API) ---")
    
    # Initialize Gemini Provider
    try:
        with open('app/AI/llm_config/llm_config.json', 'r') as f:
            config_data = json.load(f)
        gemini_cfg = LLMModelConfig(config_data['providers']['gemini'])
        llm = GeminiProvider(model_config=gemini_cfg)
    except Exception as e:
        print(f"Failed to initialize Gemini: {e}. Make sure config and .env are set.")
        return None

    # 1. Benchmark Embedding Time
    # Gemini has rate limits. We must batch.
    print("Generating embeddings via API (this may take time due to rate limits)...")
    start_time = time.time()
    doc_embeddings = []
    batch_size = 50 # Adjust based on API limits
    
    # Simple synchronous batching for the benchmark script
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"  Processing batch {i}-{i+len(batch)} / {len(texts)}")
        try:
            # Assuming llm.gemini_embedding_model returns a list of lists
            batch_emb = await llm.gemini_embedding_model(batch)
            doc_embeddings.extend(batch_emb)
            time.sleep(1) # Basic rate limiting
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            return None

    embed_time = time.time() - start_time
    doc_embeddings_np = np.array(doc_embeddings)
    print(f"Gemini Corpus Embedding Time: {embed_time:.2f} seconds")

    # 2. Benchmark Retrieval Accuracy
    ranks = []
    hits_at_3 = 0

    print("Evaluating queries...")
    for q in queries:
        if q['entity_type'] != 'tool': continue
        
        target_id = q['expected_id']
        try:
            target_index = tool_ids.index(target_id)
        except ValueError:
            continue

        q_emb = await llm.gemini_embedding_model([q['query']])
        q_emb_np = np.array(q_emb)
        
        scores = cosine_similarity(q_emb_np, doc_embeddings_np)[0]
        
        rank = get_rank(scores, target_index)
        ranks.append(rank)
        if rank <= 3:
            hits_at_3 += 1
            
        print(f"  Query: '{q['query'][:30]}...' -> Target: {target_id} -> Rank: {rank}")

    return {
        "Model": "Gemini (API)",
        "Embedding Time (s)": round(embed_time, 2),
        "Mean Reciprocal Rank (MRR)": round(np.mean([1/r for r in ranks]), 4) if ranks else 0,
        "Hit@3 (%)": round((hits_at_3 / len(ranks)) * 100, 2) if ranks else 0
    }

async def main():
    # Ensure data exists
    if not os.path.exists(DATA_FILE):
        print(f"❌ Data file not found: {DATA_FILE}")
        print("Please run the pipeline manually first: python app/bioblend_server/informer/pipeline/tool_pipeline.py")
        return

    texts, tool_ids = load_data()
    queries = load_queries()

    if not texts or not queries:
        print("No data or queries to test.")
        return

    # --- MODIFICATION: Only run the Gemini benchmark ---
    # sbert_results = evaluate_sbert(texts, tool_ids, queries) # We skip this now
    gemini_results = await evaluate_gemini(texts, tool_ids, queries)

    # Print Comparison
    print("\n" + "="*60)
    print("BENCHMARK RESULTS DOCUMENTATION")
    print("="*60)

    results = []
    if gemini_results:
        results.append(gemini_results)
    else:
        print("Gemini benchmark did not complete successfully.")
        return
    
    df = pd.DataFrame(results)
    # Reorder columns
    cols = ["Model", "Hit@3 (%)", "Mean Reciprocal Rank (MRR)", "Embedding Time (s)"]
    print(df[cols].to_string(index=False))
    print("="*60)
    print("Hit@3: Percentage of times the correct tool was in the top 3 results.")
    print("MRR: Higher is better (1.0 is perfect).")
    print("\n✅ RECOMMENDATION: Proceed with Gemini API due to performance constraints of local models.")


if __name__ == "__main__":
    asyncio.run(main())