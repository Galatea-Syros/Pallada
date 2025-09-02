from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain_community.embeddings import JinaEmbeddings

# ---------------- CONFIG ----------------
OPENAI_MODEL = "text-embedding-3-small"
load_dotenv()
LOCAL_EMBED_MODEL = "jinaai/jina-embeddings-v3"
local_model = JinaEmbeddings(
    model=LOCAL_EMBED_MODEL,
    jina_api_key=os.getenv("JINA_API_KEY")  # must be set in .env
)

CHROMA_PERSIST_DIR = ".venv/chroma_db"
COLLECTION_NAME = "pdf_documents"

# -----------------------------------------

# --- Init OpenAI client ---
load_dotenv()
client_oa = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  # Fixed typo: was OPEN_AI_API_KEY
if not client_oa.api_key:
    raise RuntimeError("Set OPENAI_API_KEY in env")

# Load Chroma DB
client_chroma = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
try:
    collection = client_chroma.get_collection(COLLECTION_NAME)
except Exception as e:
    print(f"Warning: Could not load collection '{COLLECTION_NAME}': {e}")
    collection = None

# Load small model (good for testing only)
generator = pipeline("text-generation", model="distilgpt2")


def hf_chat(prompt: str):
    output = generator(prompt, max_length=300, do_sample=True, temperature=0.7)
    return output[0]["generated_text"]


app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str

# ROOT ENDPOINT
@app.get("/")
def read_root():
    return {
        "message": "FastAPI RAG backend is running!",
        "status": "online",
        "endpoints": {
            "health": "/",
            "rag_query": "/rag"
        }
    }

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "RAG API"}

# Test endpoint that doesn't require the heavy ML models
@app.post("/test")
def test_endpoint(req: QueryRequest):
    return {
        "question": req.question,
        "answer": f"Test response for: {req.question}",
        "status": "success"
    }

# Your original RAG endpoint
@app.post("/rag")
def rag_query(req: QueryRequest):
    try:
        if collection is None:
            return {"error": "Collection not available"}

        # 1. Embed the query using the same local model
        query_embedding = local_model.embed_query(req.question)

        # 2. Query Chroma for nearest neighbors
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        # 3. Check if we found any documents
        if not results["documents"][0]:
            return {
                "question": req.question,
                "answer": "No relevant documents found in the database.",
                "sources": [],
                "status": "success"
            }

        # 4. Prepare context and sources
        context_text = "\n".join(results["documents"][0])

        # Prepare sources with metadata
        sources = []
        for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
        )):
            source = {
                "id": i + 1,
                 "Title": os.path.splitext(metadata["filename"])[0] if metadata and metadata.get("filename") else None,
                "content": doc,
                "metadata": metadata if metadata else {},
                "similarity_score": round(1 - distance, 4) if distance else None,  # Convert distance to similarity
                "preview": doc[:200] + "..." if len(doc) > 200 else doc
            }
            sources.append(source)

        # 5. Generate answer
        prompt = f"Answer the question using only the context below:\n\n{context_text}\n\nQuestion: {req.question}\nAnswer:"
        output = generator(prompt, max_length=300, do_sample=True, temperature=0.7)

        # Extract just the answer part (remove the prompt)
        full_response = output[0]["generated_text"]
        if "Answer:" in full_response:
            answer = full_response.split("Answer:")[-1].strip()
        else:
            answer = full_response

        return {
            "question": req.question,
            "answer": answer,
            "sources": sources,
            "status": "success"
        }

    except Exception as e:
        return {
            "error": f"An error occurred: {str(e)}",
            "status": "error"
        }


# This is important - make sure this runs
if __name__ == "__main__":
    import uvicorn
    print("Starting server...")
    print("Available endpoints:")
    print("- GET  / (root)")
    print("- GET  /health")
    print("- POST /test")
    print("- POST /rag")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

