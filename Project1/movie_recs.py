# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "pymongo",
#     "python-dotenv",
#     "requests",
#     "huggingface-hub",
#     "numpy",
# ]
# ///

# Actual code

import os
import pymongo
from dotenv import load_dotenv # pyright: ignore[reportMissingImports]
from pathlib import Path
import requests
from huggingface_hub import InferenceClient
import time

# Loading env var and paths
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
mongo_uri = os.getenv("MONGODB_URL")
hf_token = os.getenv("HF_TOKEN")

if not mongo_uri:
    raise ValueError("MONGODB_URL not found in .env file!")

client = pymongo.MongoClient(
    mongo_uri,
    tls=True,
    tlsAllowInvalidCertificates=True  
)

db = client.sample_mflix
collection = db.movies

# Checking env var working or not

# print(collection.count_documents({}))  
# items = collection.find().limit(10)
# print(items)
# for item in items:
#     print(item["title"])
# print(hf_token)

# Generate embeddings - Using Hugging Face Inference Client
hf_client = InferenceClient(token=hf_token)

def generate_embedding(text: str) -> list[float]:
  result = hf_client.feature_extraction(
    text,
    model="sentence-transformers/all-MiniLM-L6-v2"
  )
  
  # Convert to list if it's not already
  if hasattr(result, 'tolist'):
    embedding = result.tolist()
  else:
    embedding = list(result)
  print(embedding)
  print(f"Generated embedding of length: {len(embedding)}")
  return embedding

# Generate embeddings for movies (run this once to populate the database)
# ALREADY RUN - Comment out after first run to avoid regenerating
# print("\nGenerating embeddings for movies...")
# count = 0
# for doc in collection.find({'plot':{"$exists": True}}).limit(50):
#   try:
#     doc['plot_embedding_hf'] = generate_embedding(doc['plot'])
#     collection.replace_one({'_id': doc['_id']}, doc) # Replaces with updated document containing the embedding
#     count += 1
#     print(f"[{count}/50] Generated embedding for: {doc['title']}")
#     # Add a small delay to avoid rate limiting
#     time.sleep(0.5)
#   except Exception as e:
#     print(f"Error generating embedding for {doc.get('title', 'Unknown')}: {e}")
#     print("Continuing with vector search using already generated embeddings...")
#     break
# print(f"\nTotal embeddings generated: {count}")

# Try different queries! Examples:
# "imaginary characters from outer space at war"
# "a romantic love story"
# "epic adventure with treasure"
# "detective solving mysteries"
# "family comedy"

query = ""

print(f"\nSearching for: {query}")
query_embedding = generate_embedding(query)
print(f"Query embedding generated successfully")

results = collection.aggregate([
  {"$vectorSearch": {
    "queryVector": query_embedding,
    "path": "plot_embedding_hf",
    "numCandidates": 100,
    "limit": 4,
    "index": "PlotSemanticSearch",
      }}
])

print("\n" + "="*80)
print("SEARCH RESULTS:")
print("="*80)

print("\n" + "="*80)
print("RESULTS:", results)
print("="*80)

result_count = 0
for document in results:
    result_count += 1
    print(f'\n[{result_count}] {document["title"]}')
    print(f'Plot: {document["plot"]}')
    print("-"*80)

if result_count == 0:
    print("No results found. Make sure the embeddings are generated in the database and the index is created.")
