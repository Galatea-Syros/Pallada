import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer


# ---------------- CONFIG ----------------
OPENAI_MODEL = "text-embedding-3-small"
LOCAL_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
local_model = SentenceTransformer(LOCAL_EMBED_MODEL)

CHROMA_PERSIST_DIR = ".venv/chroma_db"
COLLECTION_NAME = "pdf_documents"
# -----------------------------------------
# Load Chroma DB
client_chroma = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
try:
    collection = client_chroma.get_collection(COLLECTION_NAME)
except Exception as e:
    print(f"Warning: Could not load collection '{COLLECTION_NAME}': {e}")
    collection = None


def debug_chroma_query(collection, query_text, model):
    """
    Debugs embedding & Chroma setup to identify issues with garbage results.
    """

    print("\n=== Step 1. Generate Query Embedding ===")
    query_embedding = model.encode([query_text], convert_to_numpy=True).tolist()
    print(f"Query embedding shape: {np.array(query_embedding).shape}")
    print(f"First 5 dims: {query_embedding[0][:5]}")

    print("\n=== Step 2. Peek at Collection Embeddings ===")
    peeked = collection.peek()  # default 10 items
    sample_embedding = peeked["embeddings"][0]
    print(f"Collection embedding shape: {np.array(sample_embedding).shape}")
    print(f"First 5 dims: {sample_embedding[:5]}")


    print("\n=== Step 3. Compare Dimensions ===")
    qdim = len(query_embedding[0])
    cdim = len(peeked["embeddings"][0])
    if qdim != cdim:
        print(f"❌ DIMENSION MISMATCH! Query dim={qdim}, Collection dim={cdim}")
    else:
        print("✅ Dimensions match.")

    print("\n=== Step 4. Collection Metadata (distance metric) ===")
    try:
        print(collection.metadata)
    except Exception:
        print("⚠️ Could not fetch collection metadata. Check Chroma version.")

    print("\n=== Step 5. Run a Test Query ===")
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    print("Results IDs:", results.get("ids", []))
    print("Distances:", results.get("distances", []))
    print("Docs preview:", [doc[:100] for doc in results.get("documents", [[]])[0]])

    return results

results = debug_chroma_query(collection, "What is quantum computing?", local_model)