## RAG pipeline with LangChain + Groq Llama-4 (example)

This project demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline using:

- Local JSON files stored in `DATA/invoice json/` as the document source
- Embeddings from `sentence-transformers` (all-MiniLM-L6-v2)
- Chroma vector store (local persistence)
- Groq Llama-4 Maverick for generation via `groq-client` (wrappped in a minimal LangChain LLM)

Important: Do NOT hardcode API keys in the repo. Use environment variables.

Requirements
- Python 3.10+
- See `requirements.txt` for Python packages

Setup (PowerShell)

```powershell
.
# Create venv and install deps
.
.
``` 

Environment variables
- `GROQ_API_KEY` — set this to your Groq API key (do not commit it)
- `GROQ_MODEL` — optional, defaults to `meta-llama/llama-4-maverick-17b-128e-instruct`

Running

1. Activate venv
2. python rag.py

Streamlit UI

Run the Streamlit app to ask questions via a web UI:

```powershell
# Activate venv first if not active
.
$env:GROQ_API_KEY='your-key-here'
python -m streamlit run streamlit_app.py
```

In the app:
- Click “(Re)build vectorstore” to index `DATA/invoice json/` on first run.
- Ask your question in the input field.
- Answers stream in as they arrive from the Groq API (OpenAI-style streaming).
- “Sources” section lists the top-matched passages and the source JSON filename.
- Use the sidebar controls:
  - Groq Model: override the model identifier.
  - Temperature: slide from 0.0 (most factual) to 1.0 (more creative). Default 0.0.
  - Context Sources: number of document chunks to retrieve (4-20). More = broader context.
  - Document Order: choose "Serial (by filename)" for consistent filename-based order or "Similarity (by relevance)" for best-match ranking.
  - Prompt Mode: choose between "Invoice Understanding" (extract/normalize to a consistent JSON schema and answer invoice-related queries) and "Factual QA (strict)" (answer only from retrieved context, otherwise reply "I don't know").

References / Grounding
- Groq LLMs and API: https://www.groq.ai/ (use `groq-client` for Python integrations)
- Llama 4 Maverick: model family announced by Meta; consult Groq docs for exact model naming and capabilities
- LangChain docs: https://langchain.com/
- Chroma: https://www.trychroma.com/
- SentenceTransformers: https://www.sbert.net/

Notes
- This is a starter scaffold. For production usage, handle retries, timeouts, batching, access control, and more robust metadata handling.

Streaming
- We call Groq’s OpenAI-compatible Chat Completions endpoint with `stream=True` to receive partial deltas and update the UI incrementally.
- If your network blocks SSE or the API, the UI will show a placeholder and may fall back to a static message.

Offline fallback
- If the API is unreachable (e.g., DNS fail), the app shows a notice like `[Groq API unreachable...]` and still displays retrieved “Sources” so you can inspect context.

Environment
- GROQ_API_KEY — required.
- GROQ_API_URL — optional (defaults to `https://api.groq.com/openai`).
- GROQ_RETRIES — optional retry count (default 3).

References / Grounding
- Groq API (OpenAI-compatible): see Groq docs
- Llama 4 Maverick model family: Groq documentation
- LangChain (used initially; current embedding/retrieval path is direct to Chroma)
- Chroma vector database: https://docs.trychroma.com/
- SentenceTransformers: https://www.sbert.net/

Groq Model Name
Set GROQ_MODEL to the exact published name from Groq (example: `meta-llama/llama-4-maverick-17b-128e-instruct`). If the endpoint returns 404 or model_not_found, verify the model identifier and API base.

Connectivity Test
- Manual script: `python check_groq.py`
- Pytest live (requires env var):
```powershell
$env:RUN_GROQ_LIVE='1'
python -m pytest tests/test_groq_live.py -q
```

Secrets
- A `.env` file was added for convenience. That file contains your Groq API key. This is generally insecure to commit — if this repository will be shared, remove `.env` or rotate the key. `.env` is now listed in `.gitignore`.

Implemented features / changelog
--------------------------------
This project has been iteratively improved during the session. Below is a comprehensive list of implemented features, changes, and debugging steps applied so far:

- Project scaffold
  - `rag.py`: core RAG pipeline, data loading, chunking, embedding, vectorstore creation, retrieval, and a simple CLI QA loop.
  - `streamlit_app.py`: interactive Streamlit UI for querying the vectorstore with streaming LLM responses and source citations.
  - `requirements.txt`, `.gitignore`, `setup_venv.ps1` (venv setup helper), and `.env` (local secrets for convenience).

- Data ingestion and vectorstore
  - Loads JSON invoice files from `DATA/invoice json/` and flattens them into text chunks with metadata containing the source filename.
  - Chunking improved to larger passages (500 token chunks with 100-token overlap) for richer context per vector.
  - Embeddings use `sentence-transformers` (all-MiniLM-L6-v2).
  - Local Chroma vectorstore persisted under `.chroma` using `chromadb.PersistentClient`.

- Retrieval behavior
  - Default retrieval depth increased (12 chunks) and exposed as a sidebar slider (`Context Sources`) in Streamlit (range 4–20).
  - Added `Document Order` option: "Serial (by filename)" (deterministic) or "Similarity (by relevance)" (default similarity ranking).
  - `run_qa` updated to optionally order retrieved chunks by filename.

- Groq LLM integration
  - `GroqLLM` wrapper implemented using OpenAI-compatible chat completions endpoint (`https://api.groq.com/openai/v1/chat/completions`).
  - Streaming support implemented: `generate_stream` yields partial deltas and allows a `chunk_handler` callback to update the UI progressively.
  - Retry/backoff logic added for robustness (configurable via `GROQ_RETRIES`).
  - Security: `.env` is loaded at startup and LLM methods read `GROQ_API_KEY` from the environment at call time to ensure the latest key is used.

- Anti-hallucination and system prompts
  - Default LLM sampling tuned to be conservative: temperature default 0.0, `top_p` 0.1.
  - Two system prompt modes exposed in the UI:
    - "Invoice Understanding": an invoice-focused system prompt instructing the LLM to normalize invoice fields and analyze multiple documents comprehensively.
    - "Factual QA (strict)": strict context-only instruction; if unsupported, the LLM should reply exactly "I don't know".
  - Streamlit passes the selected system prompt on each LLM call.

- Streamlit UI features
  - Rebuild vectorstore button to ingest JSON files and persist vectors locally.
  - Sidebar controls: Groq Model, Temperature slider, Context Sources, Document Order, Prompt Mode.
  - Progressive streaming output display and source expanders listing matched passages with filenames.

- Tests and debugging
  - Unit tests added for core LLM wrapper behavior and chunking; optional live Groq connectivity test (skipped by default).
  - Connectivity checker utility `check_groq.py` to verify model/endpoint reachability.
  - Fixed Chroma API usage and removed deprecated calls.
  - Verified pytest run: 2 passed, 1 skipped (live test).

- Behavior fixes
  - Increased retrieval depth and chunk sizes to include more invoices in context.
  - Implemented serial ordering to give deterministic document order to the LLM when requested.
  - Improved prompting to explicitly tell the LLM to analyze ALL provided document chunks to avoid partial summaries.
  - Ensured the app reads GROQ_API_KEY at request time from environment (and loads `.env` at startup).

Next recommended steps
----------------------
- Add automated tests that validate end-to-end example queries (mocking LLM responses) to ensure the RAG flow handles multi-document queries correctly.
- Add an optional `--order` flag or UI preset to persist the choice of retrieval order as part of saved sessions.
- Consider adding a small post-processing validator that scans the model's JSON output (for invoice extraction) and validates required fields are present/empty strings otherwise.
