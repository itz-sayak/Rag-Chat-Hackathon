"""
RAG pipeline using local JSON files, Chroma vectorstore, sentence-transformers embeddings,
and Groq Llama-4 via the groq-client.

Usage:
  - Create a virtualenv and install from requirements.txt
  - Set env var GROQ_API_KEY to your Groq key
  - Optionally set GROQ_MODEL (defaults to "llama-4-maverick")
  - Run: python rag.py

This script:
  - Loads JSON files from DATA/invoice json/
  - Extracts text content
  - Chunks text into passages
  - Builds embeddings with sentence-transformers
  - Stores vectors in chroma
  - Runs a simple retrieval + generation loop

Note: This is an example scaffold. Do NOT hardcode API keys in source.
"""

import os
import json
from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer
import chromadb
from dataclasses import dataclass


@dataclass
class Document:
    page_content: str
    metadata: dict

# Avoid importing LangChain LLM base to reduce tight coupling in this scaffold.


from dotenv import load_dotenv

# Load .env at module import so environment variables are available. Streamlit app also calls load_dotenv on startup.
load_dotenv()

DATA_DIR = Path("DATA") / "invoice json"
CHROMA_DIR = Path(".chroma")

# Model default name
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-4-maverick")


def load_json_files(folder: Path) -> List[Document]:
    docs = []
    for p in sorted(folder.glob("*.json")):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Failed to parse {p}: {e}")
            continue

        # Heuristic: flatten keys into a simple text blob
        parts = []
        def walk(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    walk(v, prefix=prefix + k + ": ")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    walk(v, prefix=prefix + f"[{i}] ")
            else:
                parts.append(prefix + str(obj))

        walk(raw)
        text = "\n".join(parts)
        metadata = {"source": str(p.name)}
        docs.append(Document(page_content=text, metadata=metadata))

    return docs


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


class GroqLLM:
    """Configurable LangChain LLM wrapper for Groq via simple REST calls.

    Supports basic parameters (temperature, max_tokens, top_p). Also provides a
    `generate_stream` helper to iterate chunked responses if the Groq API supports streaming.
    """

    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 0.1,
        system_prompt: str | None = None,
    ):
        # Defaults are tuned to minimize hallucinations
        self.model_name = model_name or GROQ_MODEL
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.top_p = float(top_p)
        self.system_prompt = (
            system_prompt
            or (
                "You are a cautious, factual assistant for question answering over provided documents. "
                "Follow these rules strictly: 1) Use ONLY the supplied Context. 2) If the answer is not "
                "fully supported by the Context, reply exactly: 'I don't know'. 3) Do NOT guess or invent "
                "facts, numbers, or citations. 4) Be concise and include only information from the Context."
            )
        )

    @property
    def _identifying_params(self):
        return {"model_name": self.model_name, "temperature": self.temperature, "max_tokens": self.max_tokens, "top_p": self.top_p}

    @property
    def _llm_type(self):
        return "groq"

    def _call(self, prompt: str, stop=None) -> str:
        import requests, time

        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set. Please add it to your .env or environment.")

        # Use Groq's OpenAI-compatible Chat Completions API
        base_url = os.getenv("GROQ_API_URL", "https://api.groq.com/openai")
        endpoint = base_url.rstrip('/') + "/v1/chat/completions"
        headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False,
        }

        attempts = int(os.getenv("GROQ_RETRIES", "3"))
        backoff = 2.0
        last_err = None
        for attempt in range(1, attempts + 1):
            try:
                resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict) and "choices" in data and data["choices"]:
                    choice = data["choices"][0]
                    # OpenAI-style: choices[0].message.content
                    if "message" in choice and isinstance(choice["message"], dict):
                        return choice["message"].get("content", "")
                    # Some providers also include `text`
                    return choice.get("text", "")
                return json.dumps(data)
            except Exception as e:
                last_err = e
                if "getaddrinfo failed" in str(e).lower():
                    break
                time.sleep(backoff * attempt)

        return f"[Groq API unreachable after {attempts} attempts: {last_err}]"

    def generate_stream(self, prompt: str, chunk_handler=None, timeout: int = 300):
        """Generator that yields partial text chunks from the Groq API when streaming is available.

        chunk_handler: optional callable called with each chunk (for callbacks)
        """
        import requests

        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set. Please add it to your .env or environment.")

        base_url = os.getenv("GROQ_API_URL", "https://api.groq.com/openai")
        endpoint = base_url.rstrip('/') + "/v1/chat/completions"
        headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": True,
        }

        resp = requests.post(endpoint, json=payload, headers=headers, stream=True, timeout=timeout)
        try:
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Groq streaming request failed: {e}; status={getattr(resp, 'status_code', None)}")

        # OpenAI-compatible SSE sends lines like: "data: {json}\n\n" and a final "data: [DONE]"
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw
            if line.startswith("data: "):
                line = line[len("data: ") :]
            if line.strip() == "[DONE]":
                break
            try:
                obj = json.loads(line)
                delta_text = ""
                if isinstance(obj, dict) and "choices" in obj and obj["choices"]:
                    choice = obj["choices"][0]
                    # OpenAI stream uses choices[0].delta.content
                    if "delta" in choice and isinstance(choice["delta"], dict):
                        delta_text = choice["delta"].get("content", "")
                    # Some providers use text
                    delta_text = delta_text or choice.get("text", "")
                chunk = delta_text or ""
            except Exception:
                chunk = ""

            if chunk_handler:
                try:
                    chunk_handler(chunk)
                except Exception:
                    pass
            yield chunk


def build_vectorstore(docs: List[Document]):
    """Build a Chroma collection using sentence-transformers directly.

    Returns the chromadb collection object.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(name="invoices")

    texts = []
    metadatas = []
    ids = []
    for idx, d in enumerate(docs):
        for i, chunk in enumerate(chunk_text(d.page_content, chunk_size=500, overlap=100)):
            texts.append(chunk)
            metadatas.append(d.metadata)
            ids.append(f"doc-{idx}-{i}")

    # Compute embeddings in batch
    embeddings = model.encode(texts, convert_to_numpy=True)

    # Add to collection (PersistentClient automatically persists to path)
    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings.tolist())
    return collection


def load_collection():
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return chroma_client.get_or_create_collection(name="invoices")


def run_qa(vectordb_collection, query: str, k: int = 12, order_by_filename: bool = True) -> str:
    """Retrieve top-k passages from the chroma collection and call GroqLLM to answer."""
    model = SentenceTransformer("all-MiniLM-L6-v2")

    if vectordb_collection is None:
        vectordb_collection = load_collection()

    # compute query embedding
    q_emb = model.encode([query], convert_to_numpy=True)[0].tolist()
    results = vectordb_collection.query(query_embeddings=[q_emb], n_results=k, include=['documents', 'metadatas'])

    # results['documents'] is a list of lists (one per query)
    docs = []
    if results and 'documents' in results:
        # Combine docs with metadata for potential sorting
        doc_meta_pairs = []
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i] if 'metadatas' in results and results['metadatas'] and results['metadatas'][0] and i < len(results['metadatas'][0]) else {}
            doc_meta_pairs.append((doc, meta))
        
        # Sort by filename if requested (default behavior)
        if order_by_filename:
            doc_meta_pairs.sort(key=lambda x: x[1].get('source', ''))
        
        # Extract documents in chosen order
        for doc, meta in doc_meta_pairs:
            docs.append(doc)

    # Create a strict prompt with retrieved context to minimize hallucinations
    context = "\n\n".join(docs)
    prompt = (
        "Answer strictly using ONLY the Context below. If the Context does not contain the exact "
        "information needed, respond exactly with: I don't know. Do not guess or add facts.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )

    llm = GroqLLM()
    return llm._call(prompt)


def main():
    print("Loading documents...")
    docs = load_json_files(DATA_DIR)
    if not docs:
        print(f"No documents found in {DATA_DIR}. Exiting.")
        return

    print(f"Loaded {len(docs)} documents. Building vectorstore...")
    vectordb = build_vectorstore(docs)
    print("Vectorstore built and persisted.")

    while True:
        q = input("Enter a question (or 'exit'): ")
        if not q or q.strip().lower() in ("exit", "quit"):
            break
        ans = run_qa(vectordb, q)
        print("\nAnswer:\n", ans)


if __name__ == '__main__':
    main()
