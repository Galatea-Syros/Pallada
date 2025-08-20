# 👋 Welcome all dipshits  

This is **Pallada**, a RAG system for our firm.  

### 🔧 How she works  
- Processes PDFs in Python using **Tesseract OCR**  
- Splits text with a **Hugging Face tokenizer**  
- Creates embeddings with **multilingual-mpnet**  
- Stores vectors in **ChromaDB**  
- Receives input from a **mock-up HTML frontend** via **FastAPI**  
- Finds the top **5 nearest neighbors** (subject to change) using cosine similarity  
- Provides neighbors as context to an **LLM** (currently **GPT-2** for testing, but will be swapped for a real model via API)  
- Returns the response **alongside sources and metadata** (with trimming for visual clarity)  

> She don’t cook, she don’t clean, and she fw Cardi.  

---

## 📝 Wannados  

1. **Swap the LLM**  
   - Connect Pallada to a proper LLM.  
   - Current GPT-2 rants are charming but mostly useless.  

2. **Expand the database**  
   - Move beyond the toy set of ~20 PDFs.  
   - Explore **vector quantization** if expansion slows down retrieval.  
   - Consider migrating to a **cloud-native datastore** (e.g., Milvus w/ Zilliz) if storage needs grow.  

3. **Knowledge graph integration**  
   - Layer Pallada on top of a **knowledge graph** for multi-hop retrieval via **GraphRAG**.  
   - Potentially migrate to **Neo4J**, which may make points 2b and 2c moot.  

4. **Chat capabilities**  
   - Enable conversational back-and-forth.  
   - Requires a better LLM first.  

5. **Frontend polish**  
   - Beautify the interface. (…like hell I will).  

6. **Employee-project linkage**  
   - Connect projects to employees.  
   - Allow **direct scheduling of 1:1s** with employees who worked on relevant projects via Pallada’s endpoint.  
