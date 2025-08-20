import chromadb


CHROMA_PERSIST_DIR = ".venv/chroma_db"
COLLECTION_NAME = "pdf_documents"
# Load Chroma DB
client_chroma = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
try:
    collection = client_chroma.get_collection(COLLECTION_NAME)
except Exception as e:
    print(f"Warning: Could not load collection '{COLLECTION_NAME}': {e}")
    collection = None

print(collection.count())   # how many docs
print(collection.peek()["embeddings"][0][:5])  # sample embedding


# Delete it entirely
client_chroma.delete_collection(COLLECTION_NAME)