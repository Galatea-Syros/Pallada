import os
import time
import uuid
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from openai import OpenAI
import chromadb
from tenacity import retry, wait_exponential, stop_after_attempt
from sentence_transformers import SentenceTransformer
from typing import List
import re
import unicodedata
from transformers import AutoTokenizer

# ---------------- CONFIG ----------------
OPENAI_MODEL = "text-embedding-3-small"
LOCAL_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
local_model = SentenceTransformer(LOCAL_EMBED_MODEL)

CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 100
EMBEDDING_BATCH = 50

CHROMA_PERSIST_DIR = ".venv/chroma_db"
PDF_DIR = "./Deliverables (phase 0)"
COLLECTION_NAME = "pdf_documents"
# -----------------------------------------

# --- Init OpenAI client ---
load_dotenv()
client_oa = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
if not client_oa.api_key:
    raise RuntimeError("Set OPENAI_API_KEY in env")

# --- Init Chroma ---
client_chroma = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
if COLLECTION_NAME in [c.name for c in client_chroma.list_collections()]:
    collection = client_chroma.get_collection(COLLECTION_NAME)
else:
    collection = client_chroma.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}   # <<< use cosine for mpnet
)


# --- Token-aware splitter ---
class TokenAwareTextSplitter:
    def __init__(self, model_name: str, chunk_size: int, chunk_overlap: int):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            start += self.chunk_size - self.chunk_overlap
        return chunks

splitter = TokenAwareTextSplitter(LOCAL_EMBED_MODEL, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS)

def clean_text(text: str) -> str:
    # Normalize unicode (important for accented Greek characters)
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters (line breaks, form feeds, etc.)
    text = re.sub(r"[\r\n\t\f]+", " ", text)

    # Remove weird leftover PDF artifacts (non-printables)
    text = re.sub(r"[^\x20-\x7E\u0370-\u03FF\u1F00-\u1FFF]+", " ", text)
    # ↑ keeps ASCII + Greek unicode ranges

    # Collapse multiple spaces
    text = re.sub(r"\s{2,}", " ", text)

    # Trim leading/trailing spaces
    return text.strip()

# --- Embedding function with retry ---
@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    # Clean texts: remove None, ensure str type
    clean_texts = [(t if isinstance(t, str) else str(t)) for t in texts]
    clean_texts = [t if t.strip() != "" else " " for t in clean_texts]

    try:
        #resp = client_oa.embeddings.create(
        #    model=OPENAI_MODEL,
        #    input=clean_texts
        #)
        embeddings = local_model.encode(
            clean_texts,
            convert_to_numpy=True,  # Keep as list of lists
            show_progress_bar=True
        )
    except Exception as e:
        print(f"Embedding API call failed: {e}")
        raise

    # Convert numpy array to list of lists if needed
    embeddings = embeddings.tolist()

    return embeddings

# --- Element processing ---
def prepare_chunks_from_elements(elements, source_metadata: Dict) -> List[Dict]:
    chunks = []
    for el in elements:
        # Element type (old/new unstructured)
        el_type = getattr(el, "element_type", type(el).__name__)

        # Page number (old dict-style vs new dataclass)
        page = None
        if hasattr(el, "metadata") and el.metadata is not None:
            if hasattr(el.metadata, "get"):  # old style dict-like
                page = el.metadata.get("page_number")
            elif hasattr(el.metadata, "page_number"):
                page = el.metadata.page_number

        # Try to get element text
        text = ""
        if hasattr(el, "get_text") and callable(el.get_text):
            raw_text  = el.get_text() or ""
        elif hasattr(el, "text"):
            raw_text  = el.text or ""
        else:
            raw_text  = str(el)
        # Clean (normalize, remove artifacts, keep meaning)
        text = clean_text(raw_text)
        # Skip empty
        if not text.strip():
            continue

        # Split into token-aware chunks
        pieces = splitter.split_text(text) if len(text) > 0 else []

        for idx, piece in enumerate(pieces):
            md = {
                **source_metadata,
                "page_number": page,
                "element_type": el_type,
                "chunk_index": idx,
            }
            chunks.append({"id": str(uuid.uuid4()), "text": piece, "metadata": md})

    return chunks

# --- PDF ingestion ---
def ingest_pdf(pdf_path: Path):
    src_meta = {
        "filename": pdf_path.name,
        "filepath": str(pdf_path.resolve()),
        "creation_date": time.ctime(pdf_path.stat().st_ctime),
    }

    print(f"📄 Processing {pdf_path.name} ...")
    elements = partition_pdf(str(pdf_path), languages=["ell"], strategy="hi_res")
    chunks = prepare_chunks_from_elements(elements, src_meta)

    if not chunks:
        print(f"⚠️ No text extracted from {pdf_path.name}")
        return

    ids_batch, texts_batch, metas_batch = [], [], []

    for i, chunk in enumerate(chunks):
        ids_batch.append(chunk["id"])
        texts_batch.append(chunk["text"])
        metas_batch.append(chunk["metadata"])

        # When batch full or last chunk, process
        if len(texts_batch) >= EMBEDDING_BATCH or i == len(chunks) - 1:
            print(f"🔹 Embedding {len(texts_batch)} chunks from {pdf_path.name} (batch {i // EMBEDDING_BATCH + 1})")
            embeddings = get_embeddings_batch(texts_batch)
            collection.add(
                ids=ids_batch,
                embeddings=embeddings,
                documents=texts_batch,
                metadatas=metas_batch,
            )
            ids_batch, texts_batch, metas_batch = [], [], []

# --- Folder ingestion ---
def ingest_folder_recursive(root_folder: str):
    root_path = Path(root_folder)
    pdf_files = sorted(root_path.rglob("*.pdf"))

    print(f"Found {len(pdf_files)} PDFs under {root_folder}")
    for idx, pdf in enumerate(pdf_files, 1):
        print(f"\n[{idx}/{len(pdf_files)}] Processing {pdf.relative_to(root_path)}")
        try:
            ingest_pdf(Path(pdf))
        except Exception as e:
            print(f"❌ Failed ingest for {pdf}: {e}")

# --- Verification ---
def verify_collection():
    print("\n--- Chroma Verification ---")
    count = collection.count()
    print(f"Total documents stored: {count}")

    if count > 0:
        peek = collection.peek(limit=3)
        for doc, meta in zip(peek["documents"], peek["metadatas"]):
            print(f"\nText preview: {doc[:100]!r}")
            print(f"Metadata: {meta}")

        # Use the same mpnet model for query embedding
        query_text = "project architecture description"
        query_embedding = local_model.encode([query_text], convert_to_numpy=True).tolist()

        qres = collection.query(
            query_embeddings=query_embedding,
            n_results=3
        )

        print("\n--- Sample query results ---")
        for doc, meta in zip(qres["documents"][0], qres["metadatas"][0]):
            print(f"Match: {doc[:100]!r}")
            print(f"Metadata: {meta}")


ingest_folder_recursive(PDF_DIR)
verify_collection()
