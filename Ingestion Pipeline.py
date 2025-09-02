from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
import chromadb
import os
from collections import defaultdict
import uuid
from typing import List, Dict
from pathlib import Path
import time
import unicodedata
import re
from langchain_community.embeddings import JinaEmbeddings
# ---------------- CONFIG ----------------

load_dotenv()
LOCAL_EMBED_MODEL = "jinaai/jina-embeddings-v3"
local_model = JinaEmbeddings(
    model=LOCAL_EMBED_MODEL,
    jina_api_key=os.getenv("JINA_API_KEY")  # must be set in .env
)


CHUNK_SIZE_TOKENS = 8000
CHUNK_OVERLAP_TOKENS = 1500
MAX_TOKENS = 8192  # Jina v3 limit
EMBEDDING_BATCH = 50
GREEK_TOKENIZER_MODEL = "nlpaueb/bert-base-greek-uncased-v1"

CHROMA_PERSIST_DIR = ".venv/chroma_db"
PDF_DIR = "./Deliverables (phase 0)"
COLLECTION_NAME = "pdf_documents"
# -----------------------------------------


# --- Init Chroma ---
client_chroma = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
if COLLECTION_NAME in [c.name for c in client_chroma.list_collections()]:
    collection = client_chroma.get_collection(COLLECTION_NAME)
else:
    collection = client_chroma.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}   # <<< use cosine for mpnet
)


# Tokenizer for splitting (Greek Roberta/BERT recommended)
from transformers import AutoTokenizer
SPLIT_TOKENIZER_MODEL = "nlpaueb/bert-base-greek-uncased-v1"
splitter_tokenizer = AutoTokenizer.from_pretrained(SPLIT_TOKENIZER_MODEL, model_max_length=10**9)

# --- Utilities ---
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\r\n\t\f]+", " ", text)
    text = re.sub(r"[^\x20-\x7E\u0370-\u03FF\u1F00-\u1FFF]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

class TokenAwareTextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.tokenizer = splitter_tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        chunks = []
        start = 0
        step = self.chunk_size - self.chunk_overlap
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens).strip()
            if chunk_text:
                chunks.append(chunk_text)
            start += step
        return chunks

splitter = TokenAwareTextSplitter(CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS)

# --- Merge elements per page ---
def merge_elements_by_page(elements):
    page_texts = defaultdict(str)
    for el in elements:
        # Safely get page number
        metadata = getattr(el, "metadata", None)
        page = None
        if metadata is not None:
            if isinstance(metadata, dict):
                page = metadata.get("page_number")
            else:
                page = getattr(metadata, "page_number", None)
        if page is None:
            continue

        # Get text
        text = getattr(el, "get_text", lambda: getattr(el, "text", str(el)))()
        text = clean_text(text)
        if text:
            page_texts[page] += " " + text

    return {page: t.strip() for page, t in page_texts.items()}

# --- Prepare chunks ---
def prepare_chunks_from_elements(elements, source_metadata: Dict) -> List[Dict]:
    merged_pages = merge_elements_by_page(elements)
    chunks = []
    for page, text in merged_pages.items():
        # Split only if exceeding max tokens
        tokens = splitter.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if len(tokens) <= MAX_TOKENS:
            pieces = [text]
        else:
            pieces = splitter.split_text(text)

        for idx, piece in enumerate(pieces):
            md = {
                **source_metadata,
                "page_number": page,
                "chunk_index": idx,
            }
            chunks.append({"id": str(uuid.uuid4()), "text": piece, "metadata": md})
    return chunks

# --- Embedding batch ---
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    clean_texts = [(t if isinstance(t, str) else str(t)) for t in texts]
    clean_texts = [t if t.strip() != "" else " " for t in clean_texts]
    embeddings = local_model.embed_documents(clean_texts)
    return embeddings

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

        if len(texts_batch) >= EMBEDDING_BATCH or i == len(chunks) - 1:
            print(f"🔹 Embedding {len(texts_batch)} chunks from {pdf_path.name}")
            embeddings = get_embeddings_batch(texts_batch)
            collection.add(
                ids=ids_batch,
                embeddings=embeddings,
                documents=texts_batch,
                metadatas=metas_batch,
            )
            ids_batch, texts_batch, metas_batch = [], [], []

# --- Folder ingestion remains the same ---
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
        query_embedding = [local_model.embed_query(query_text)]

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
