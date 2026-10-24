# DocQuery
A local, privacy-first Multi-PDF Question-Answering engine built with Streamlit, LangChain, and Ollama.

## Features
- **Strictly Grounded Q&A**: Answers generated solely from uploaded PDF context with zero external hallucinations.
- **3-Query Expansion**: Uses Qwen to generate 3 alternative synonym/formal queries, maximizing retrieval recall across varied phrasing.
- **Persistent SHA-256 Vector Caching**: Fast loading of FAISS vector indices without re-embedding identical files.
- **Source Attributions**: Clear, verifiable document name and page number citations for every fact.
- **Streaming UI**: Token-by-token response streaming rendered directly in Streamlit.
- **100% Local Execution**: Zero cloud API dependencies, zero data leakage, and air-gapped readiness.

## Tech Stack
- **Python 3.10+** - Core programming language
- **Streamlit** - Web application and streaming frontend
- **LangChain Core / Community** - RAG orchestration and LCEL pipeline
- **Ollama (`qwen2.5-coder:7b`)** - Local generative reasoning engine
- **FAISS (CPU)** - High-performance local dense vector store
- **Hugging Face (`all-MiniLM-L6-v2`)** - Local sentence embeddings
- **pypdf** - PDF text extraction

## Architecture
```text
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        DocQuery Architecture                                      │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────┐          1. User Prompt          ┌────────────────────────────────────────┐
  │   Streamlit UI   │ ───────────────────────────────► │       Query Expansion Pipeline         │
  │   (Frontend)     │                                  │   (Prompt 1: Qwen2.5-Coder:7b)         │
  └──────────────────┘                                  └───────────────────┬────────────────────┘
           ▲                                                                │
           │                                                                ▼ 2. Generates 3 Queries
           │                                                    ┌───────────────────────────┐
           │                                                    │   Expanded Search Set     │
           │                                                    │  [Original + 3 Alternates]│
           │                                                    └─────────────┬─────────────┘
           │                                                                  │
           │                                                                  ▼ 3. Multi-Vector Search
           │                                                    ┌───────────────────────────┐
           │                                                    │    FAISS Vector Store     │
           │                                                    │  (Persistent Hash Cache)  │
           │                                                    └─────────────┬─────────────┘
           │                                                                  │
           │                                                                  ▼ 4. Deduplicate & Gate (≤1.15)
           │ 6. Token Stream & Sources                          ┌───────────────────────────┐
           │                                                    │   Top-K Relevant Chunks   │
           │                                                    └─────────────┬─────────────┘
           │                                                                  │
           │                                                                  ▼ 5. Grounded Context
  ┌────────┴─────────┐         Prompt 2: Synthesis      ┌─────────────────────────────────────────┐
  │   Streamlit UI   │ ◄─────────────────────────────── │           Final RAG Chain               │
  │ (Display Output) │                                  │      (Qwen2.5-Coder Streaming)          │
  └──────────────────┘                                  └─────────────────────────────────────────┘
```

## How It Works
1. **Document Ingestion** - Uploaded PDFs are parsed page-by-page and chunked into overlapping passages.
2. **Deterministic Caching** - Each PDF is fingerprinted with a SHA-256 hash across file bytes and chunk settings to save/load FAISS indices instantly.
3. **3-Query Expansion** - The user's question passes through a dedicated prompt where `qwen2.5-coder:7b` generates exactly 3 alternative queries targeting synonyms, formal terms, and numerical patterns.
4. **Vector Retrieval & Fusion** - FAISS searches across all selected documents using all 4 queries, drops distant matches exceeding the distance threshold, and deduplicates the ranked passages.
5. **Strict Grounded Generation** - The filtered context is fed into the final synthesis prompt, where Qwen streams the answer live with page citations.
6. **Citation Display** - Expanders provide exact document sources, pages, and distance scores for verification.

## Component Details
| Component | Implementation | Purpose |
| :--- | :--- | :--- |
| **Frontend** | Streamlit UI (`app.py`) | File upload, PDF selection, input form, and streaming text |
| **Embeddings** | `all-MiniLM-L6-v2` | Converts text chunks into normalized 384-dimensional dense vectors |
| **Vector Store** | FAISS CPU | In-memory and on-disk indexed similarity search |
| **Reasoning Engine** | `qwen2.5-coder:7b` (via Ollama) | Query expansion and factual response synthesis |
| **Document Parser** | `PyPDFLoader` | Reads and extracts page-level text from uploaded PDFs |

## Prerequisites
- Python 3.10 or higher
- [Ollama](https://ollama.com/) installed and running locally
- Pull the local LLM:
  ```bash
  ollama pull qwen2.5-coder:7b
  ```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/DocQuery.git
cd DocQuery
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Local LLM
Ensure your local Ollama daemon is running:
```bash
ollama list
```
Verify that `qwen2.5-coder:7b` is present in the list.

## Running the Application
```bash
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

## Project Structure
```text
DocQuery/
├── app.py              # Main application logic (UI, caching, RAG chain)
├── requirements.txt    # Python dependencies
├── .gitignore          # Git exclusion rules (caches, virtualenvs, PDFs)
├── README.md           # Project documentation
├── pdfs/               # Uploaded PDF files directory (auto-created)
└── indexes/            # Serialized FAISS vector index folders (auto-created)
```

## How to Learn from This Repo

### Step 1: Understand the Flow
Study `app.py` section by section:
- **Lines 10–20**: Modern LangChain ecosystem imports (`langchain-ollama`, `langchain-huggingface`).
- **Lines 73–108**: Query expansion prompt transforming one question into 3 alternative search vectors.
- **Lines 111–157**: SHA-256 fingerprinting for zero-cost reloading of processed vector stores.
- **Lines 200–225**: Strict anti-hallucination system prompt forcing factual citations.
- **Lines 270–295**: Dynamic LangChain Expression Language (LCEL) streaming to `st.write_stream`.

### Step 2: Key Concepts to Learn
| Concept | Code Location | What You'll Learn |
| :--- | :--- | :--- |
| **Deterministic Caching** | `app.py:get_vectorstore` | Hashing file contents and parameters to prevent duplicate compute |
| **3-Query Expansion** | `app.py:expand_queries` | Increasing recall by retrieving on alternate phrasing and synonyms |
| **Score Filtering** | `app.py:MAX_DISTANCE_THRESHOLD` | Dropping low-confidence chunks before LLM generation |
| **Streaming Generators** | `app.py:st.write_stream` | Lowering time-to-first-token latency in web applications |

### Step 3: Customization Ideas
- **Change the Model**: Swap `LLM_MODEL = "qwen2.5-coder:7b"` to `mistral:7b` or `llama3.2:3b`.
- **Adjust Chunk Settings**: Change `CHUNK_SIZE` and `CHUNK_OVERLAP` to test precision vs. context breadth.
- **Adjust Distance Threshold**: Modify `MAX_DISTANCE_THRESHOLD = 1.15` to make context gating stricter or looser.

## Security & Privacy Notes
- **Local Only**: All embeddings and text generation execute on your machine.
- **Data Protection**: Document chunks and vector indices are stored in local folders (`pdfs/`, `indexes/`) which are ignored by `.gitignore` to prevent committing sensitive documents.

## License
MIT License