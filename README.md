# 👋 Welcome all respected readers  

This is **Pallada** 🦉, a RAG system for our firm.  

### 🔧 How she works  #TechnicalDeets
- Processes PDFs in Python using **Tesseract OCR**  
- Splits text with a **Hugging Face tokenizer**  
- Creates embeddings with **multilingual-mpnet-v2**  
- Stores vectors in **ChromaDB**  
- Receives input from a **mock-up HTML frontend** via **FastAPI**  
- Finds the top **5 nearest neighbors** (subject to change) using cosine similarity  
- Provides neighbors as context to an **LLM** (currently **GPT-2** for testing, but will be swapped for a real model via API)  
- Returns the response **alongside sources and metadata**
---

## 👩‍💻 For end users #TreatYourRobotRight
- Longer prompts seem to respond better
- Not all sources come from unique files. If the same file appears as multiple sources it probably contains multiple points of interest
- Acccuracy below 71% can be considered garbadge. Ignore such sources.
 
---

## 📝 Wannados #anydaynow

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
   - Make the **filepath** metadatum function as a link
   - Make the **pagenumber** metadatum more visible
   - Hide irrelevant for the end user metadata  
   - Beautify the interface. (…like hell I will).  

7. **Employee-project linkage**  
   - Connect projects to employees. Author metadatum is a start but ideally I'd like the Harvest reports  
   - Allow **direct scheduling of 1:1s** with employees who worked on relevant projects trhough Pallada’s endpoint. Probably some **Outlook** integration.

8. **Replace requirements.txt with poetry**
