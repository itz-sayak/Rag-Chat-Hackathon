import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Load .env at startup to ensure GROQ_API_KEY and GROQ_MODEL are available
load_dotenv(override=True)

from rag import load_json_files, build_vectorstore, run_qa, load_collection, GroqLLM

DATA_DIR = Path("DATA") / "invoice json"

st.title("RAG QA over Invoice JSONs")

st.write("This app loads JSON files from `DATA/invoice json/`, builds a Chroma vectorstore (persisted to .chroma), and answers questions using Groq Llama-4.")

# Model selection dropdown with available Groq models
available_models = [
    "llama-3.1-8b-instant",
    "llama-3-3-70b-versatile",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-guard-4-12b",
    "meta-llama/llama-prompt-guard-2-22m",
    "meta-llama/llama-prompt-guard-2-86m",
]
default_model = os.getenv("GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")
default_index = available_models.index(default_model) if default_model in available_models else 2
model_name = st.sidebar.selectbox("Groq Model", options=available_models, index=default_index)
st.sidebar.write(f"Using model: `{model_name}`")

# Temperature toggle (slider)
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.01,
    help="Lower = more deterministic, higher = more creative. 0.0 is recommended to minimize hallucinations.",
)

# Retrieval count for context size control
retrieval_count = st.sidebar.slider(
    "Context Chunks",
    min_value=8,
    max_value=30,
    value=16,
    step=2,
    help="Number of relevant invoice chunks to retrieve. More chunks = better coverage but larger prompts. Reduce if you get '413 Payload Too Large' errors.",
)
st.sidebar.info("💡 Using smart retrieval: only the most relevant invoice chunks are sent to the LLM to avoid payload size limits.")

# System prompt (Invoice Understanding only)
chosen_system_prompt = (
    "You are an intelligent invoice understanding assistant. You will be given invoice data in JSON, text, or "
    "semi-structured format. The structure or field names may vary, but the underlying context is always an invoice "
    "document containing: Invoice metadata (invoice number, date); Company or vendor details; Client details; Line items "
    "(description, quantity, rate, amount, tax, total); Summary (subtotal, tax, discount, grand total); Bank or payment "
    "details; Additional notes or signatures. Your task is to: Identify and normalize these fields even if their labels "
    "differ. Handle missing or null fields gracefully. Infer context where possible. Return a consistent JSON schema with standardized keys and include empty fields when missing. CRITICAL: When provided with multiple documents, analyze ALL of them comprehensively. Do not stop at just a few examples."
)

if st.button("(Re)build vectorstore from JSON files"):
    with st.spinner("Loading documents and building vectorstore — this can take a while"):
        docs = load_json_files(DATA_DIR)
        if not docs:
            st.error(f"No documents found in {DATA_DIR}")
        else:
            build_vectorstore(docs)
            st.success("Vectorstore built and persisted to .chroma")

query = st.text_input("Enter a question")
if query:
    with st.spinner("Retrieving and generating answer..."):
        try:
            # Load the persisted Chroma collection
            collection = load_collection()

            # Build and run streaming generation (conservative sampling to reduce hallucinations)
            llm = GroqLLM(
                model_name=model_name,
                temperature=temperature,
                top_p=0.1,
                system_prompt=chosen_system_prompt,
            )

            placeholder = st.empty()
            text_so_far = [""]

            def handle_chunk(chunk):
                text_so_far[0] += chunk
                # Update the placeholder with progressively growing text
                placeholder.markdown("**Answer:**\n\n" + text_so_far[0])

            # Use semantic search to retrieve relevant invoice chunks from vectorstore
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            q_emb = model.encode([query], convert_to_numpy=True)[0].tolist()
            
            # Retrieve top-k relevant chunks (configurable via sidebar slider)
            results = collection.query(query_embeddings=[q_emb], n_results=retrieval_count, include=['documents', 'metadatas'])
            docs = []
            metas = []
            if results and 'documents' in results:
                # Combine documents with metadata
                doc_meta_pairs = []
                for i, doc in enumerate(results['documents'][0]):
                    meta = results['metadatas'][0][i] if 'metadatas' in results and results['metadatas'] and results['metadatas'][0] and i < len(results['metadatas'][0]) else {}
                    doc_meta_pairs.append((doc, meta))
                
                # Sort by source (filename) for deterministic ordering
                doc_meta_pairs.sort(key=lambda x: x[1].get('source', ''))
                
                # Extract docs and metadata in chosen order
                for doc, meta in doc_meta_pairs:
                    docs.append(doc)
                    metas.append(meta)

            context = "\n\n".join(docs)
            
            # Calculate approximate token count (rough estimate: 1 token ≈ 4 chars)
            estimated_tokens = len(context) // 4
            print(f"Context size: {len(context)} chars, ~{estimated_tokens} tokens, {len(docs)} chunks")
            
            # Build a single invoice-focused prompt (we always use invoice understanding mode)
            prompt = (
                f"You have been provided with data from {len(docs)} invoice document chunks below. "
                f"Analyze ALL the provided invoice data thoroughly. When answering questions about totals, "
                f"counts, or comparisons, examine EVERY piece of data provided, not just a subset.\n\n"
                f"Invoice Data (examine all {len(docs)} chunks):\n{context}\n\n"
                f"Question: {query}\n"
                f"Instructions: Review all {len(docs)} document chunks above before answering. "
                f"If counting or comparing, make sure to consider every invoice provided.\nAnswer:"
            )

            # Stream chunks from the LLM and display them as they arrive
            streamed_any = False
            for chunk in llm.generate_stream(prompt, chunk_handler=handle_chunk):
                streamed_any = True
            if not streamed_any:
                placeholder.markdown("**Answer:**\n\n(No streamed content received — API may be unreachable)")

            # Show source citations
            if docs:
                st.markdown("---")
                st.markdown("**Sources (top matches):**")
                for i, doc in enumerate(docs):
                    meta = metas[i] if i < len(metas) else {}
                    with st.expander(f"Source {i+1}: {meta.get('source','unknown')}"):
                        st.write(doc)

        except Exception as e:
            st.error(f"Error during retrieval/generation: {e}")
