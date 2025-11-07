# RAG Chat for Invoices — README

This repository contains a small Retrieval-Augmented Generation (RAG) demo focused on invoice JSONs. It uses local JSON invoice data, embeddings, a local Chroma vectorstore, and Groq-hosted LLMs for generation.

Key components
- `rag.py` — core pipeline: JSON loader, chunking, SentenceTransformer embeddings, Chroma persistence, and a minimal `GroqLLM` wrapper.
- `streamlit_app.py` — Streamlit UI for interactive queries with progressive streaming output and source citations.
- `check_groq.py` — small connectivity tester for your Groq key/model.

Prerequisites
- Python 3.10+
- Create and activate a virtualenv and install dependencies in `requirements.txt`.

Environment
- Use a `.env` file or set environment variables before running the app.
  - `GROQ_API_KEY` — required. Put your Groq key in `.env` (do not commit it to public repos).
  - `GROQ_MODEL` — optional; a default model is used if unset.
  - `GROQ_API_URL` — optional override of the API base (defaults to `https://api.groq.com/openai`).
  - `GROQ_RETRIES`, `GROQ_BACKOFF` — optional retry/backoff settings.

Quick start (PowerShell)
```powershell
# create venv (one-time)
python -m venv venv
.\n+# activate
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# put your key in .env (example)
# GROQ_API_KEY=gsk_...
# GROQ_MODEL=meta-llama/llama-4-maverick-17b-128e-instruct

# run Streamlit UI
python -m streamlit run streamlit_app.py
```

What’s different / current behavior
- Model selector: a dropdown in the sidebar lets you choose from several Groq model identifiers (including meta-llama/llama-4-maverick). The selected model is sent with each request.
- Smart retrieval: the app **uses the Chroma vectorstore and semantic search** to retrieve only the most relevant invoice chunks for each query. This prevents sending the entire dataset as prompt (avoids 413 Payload Too Large errors).
- Context chunk slider: control how many chunks are retrieved per query (default 16, adjustable 8–30). Reduce this if you still see payload-size issues.
- Backend logging: the app prints logs for each API call (timestamp, model, status code, elapsed time, and `Retry-After` on 429s) so you can verify the request and response flow in the terminal.

How to (re)build the vectorstore
1. From the Streamlit UI click `(Re)build vectorstore from JSON files`.
2. Or run the quick builder script in Python (from the repo root):
```powershell
C:/Hackathon/venv/Scripts/python.exe -c "from rag import load_json_files, build_vectorstore; import pathlib; docs=load_json_files(pathlib.Path('DATA')/ 'invoice json'); build_vectorstore(docs)"
```

Connectivity test
- Use the small helper to verify your key and model:
```powershell
python check_groq.py
```
It prints the status code and a short response body.

Troubleshooting
- 413 Payload Too Large: reduce the **Context Chunks** slider; ensure smart retrieval is enabled (it is by default). If your dataset is massive, consider pre-summarizing or increasing chunk-level summarization.
- 429 Too Many Requests: the app now honors `Retry-After` and implements exponential backoff with jitter. If 429s persist, consider rotating the key or ensuring no other processes are consuming the same key.
- Key exposure: `.env` is convenient for development but avoid committing it. If you have committed a key, rotate it immediately.

Testing
- Unit tests: `pytest -q` runs the unit tests. A live Groq connectivity test exists but is skipped unless you set `RUN_GROQ_LIVE=1`.

Files of interest
- `rag.py` — ingestion, embedding, retrieval, `GroqLLM` wrapper with logging and retry/backoff.
- `streamlit_app.py` — the UI: dropdown model selector, temperature, context chunk slider, rebuild button, and streaming UI.
- `check_groq.py` — connectivity tester.

Contributing notes
- Avoid committing large dataset files (PDFs/images) or keys. Use `.gitignore` and only commit code and small example JSONs.

If you want, I can:
- Add a short script to pre-summarize chunks to further reduce token usage.
- Add a developer-only demo script that runs a sample query and prints the full request/response flow to help debugging.

---
Updated to reflect: model dropdown, smart retrieval (semantic search), backend logging, `.env` key usage, and fixes for large-payload errors.
