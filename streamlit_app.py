import os
from pathlib import Path
import streamlit as st

from rag import load_json_files, build_vectorstore, run_qa, load_collection, GroqLLM

DATA_DIR = Path("DATA") / "invoice json"

st.title("RAG QA over Invoice JSONs")

st.write("This app loads JSON files from `DATA/invoice json/`, builds a Chroma vectorstore (persisted to .chroma), and answers questions using Groq Llama-4.")

default_model = os.getenv("GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")
model_name = st.sidebar.text_input("Groq Model", value=default_model)
st.sidebar.write("Override the model name above if needed.")

# Temperature toggle (slider)
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.01,
    help="Lower = more deterministic, higher = more creative. 0.0 is recommended to minimize hallucinations.",
)

# Retrieval count
retrieval_count = st.sidebar.slider(
    "Context Sources",
    min_value=4,
    max_value=20,
    value=12,
    step=1,
    help="Number of document chunks to retrieve for context. More sources = broader context but slower responses.",
)

# Retrieval order
retrieval_order = st.sidebar.radio(
    "Document Order",
    options=["Serial (by filename)", "Similarity (by relevance)"],
    index=0,
    help="Serial gives documents in filename order. Similarity ranks by query relevance.",
)

# System prompt presets
SYSTEM_PROMPT_QA = (
    "You are a cautious, factual assistant for question answering over provided documents. "
    "Rules: 1) Use ONLY the supplied Context. 2) If the answer is not fully supported, reply exactly 'I don't know'. "
    "3) Do NOT guess or invent facts, numbers, or citations. 4) Be concise."
)

SYSTEM_PROMPT_INVOICE = (
    "You are an intelligent invoice understanding assistant. You will be given invoice data in JSON, text, or "
    "semi-structured format. The structure or field names may vary, but the underlying context is always an invoice "
    "document containing: Invoice metadata (invoice number, date); Company or vendor details; Client details; Line items "
    "(description, quantity, rate, amount, tax, total); Summary (subtotal, tax, discount, grand total); Bank or payment "
    "details; Additional notes or signatures. Your task is to: Identify and normalize these fields even if their labels "
    "differ. Handle missing or null fields gracefully. Infer context where possible (e.g., “For BKG Office” implies "
    "client = BKG Office). Return a consistent JSON schema with standardized keys: { \n  \"invoice_no\": \"\", \n  \"invoice_date\": \"\", \n  \"vendor_name\": \"\", \n  \"vendor_address\": \"\", \n  \"vendor_pan\": \"\", \n  \"vendor_gstin\": \"\", \n  \"client_name\": \"\", \n  \"client_address\": \"\", \n  \"items\": [ {\"description\": \"\", \"quantity\": \"\", \"rate\": \"\", \"amount\": \"\", \"total\": \"\"} ], \n  \"total_amount\": \"\", \n  \"bank_details\": {\"account_no\": \"\", \"ifsc\": \"\", \"bank_name\": \"\"}, \n  \"notes\": \"\" \n}. If any field is missing, leave it empty but don’t remove it. When responding to user queries (like “who issued invoice SEC 50?” or “list all items above ₹10,000”), use semantic understanding, not exact field names."
)

prompt_mode = st.sidebar.radio(
    "Prompt Mode",
    options=["Invoice Understanding", "Factual QA (strict)"],
    index=0,
    help="Choose the system prompt to guide the model's behavior.",
)
chosen_system_prompt = SYSTEM_PROMPT_INVOICE if prompt_mode == "Invoice Understanding" else SYSTEM_PROMPT_QA

# For invoice understanding mode, add comprehensive analysis instruction
if prompt_mode == "Invoice Understanding":
    chosen_system_prompt = (
        "You are an intelligent invoice understanding assistant. CRITICAL: When provided with multiple documents, "
        "you MUST analyze ALL of them comprehensively. Do not stop at just a few examples. Examine every document "
        "chunk provided. When counting, comparing, or analyzing patterns, process ALL provided documents, not just a subset. "
        "For queries about duplicates, totals, or comprehensive analysis, review every single invoice in the context before answering."
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

            # First, perform retrieval + build prompt using run_qa's approach but using streaming
            # We'll implement a simple retrieval here and then stream the LLM response.
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            q_emb = model.encode([query], convert_to_numpy=True)[0].tolist()
            # Increase retrieval to get broader context from more invoice files
            results = collection.query(query_embeddings=[q_emb], n_results=retrieval_count, include=['documents', 'metadatas'])
            docs = []
            metas = []
            if results and 'documents' in results:
                # Combine documents with metadata for sorting
                doc_meta_pairs = []
                for i, doc in enumerate(results['documents'][0]):
                    meta = results['metadatas'][0][i] if 'metadatas' in results and results['metadatas'] and results['metadatas'][0] and i < len(results['metadatas'][0]) else {}
                    doc_meta_pairs.append((doc, meta))
                
                # Sort based on user preference
                if retrieval_order == "Serial (by filename)":
                    # Sort by filename to give serial order instead of similarity-based random order
                    doc_meta_pairs.sort(key=lambda x: x[1].get('source', ''))
                # If "Similarity", keep the original order (already ranked by similarity)
                
                # Extract docs and metadata in chosen order
                for doc, meta in doc_meta_pairs:
                    docs.append(doc)
                    metas.append(meta)

            context = "\n\n".join(docs)
            
            # Use different prompts based on the selected mode
            if prompt_mode == "Invoice Understanding":
                prompt = (
                    f"You have been provided with data from {len(docs)} invoice document chunks below. "
                    f"Analyze ALL the provided invoice data thoroughly. When answering questions about totals, "
                    f"counts, or comparisons, examine EVERY piece of data provided, not just a subset.\n\n"
                    f"Invoice Data (examine all {len(docs)} chunks):\n{context}\n\n"
                    f"Question: {query}\n"
                    f"Instructions: Review all {len(docs)} document chunks above before answering. "
                    f"If counting or comparing, make sure to consider every invoice provided.\nAnswer:"
                )
            else:
                prompt = (
                    f"You have been provided with {len(docs)} document chunks below. "
                    f"Answer strictly using ONLY ALL the Context provided. Examine every chunk. "
                    f"If the Context does not contain the exact information needed, respond exactly with: I don't know. "
                    f"Do not guess or add facts.\n\n"
                    f"Context ({len(docs)} chunks - examine all):\n{context}\n\nQuestion: {query}\nAnswer:"
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
