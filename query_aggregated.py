"""
Query helper for aggregated invoice data.

Usage:
    from query_aggregated import query_invoices
    answer = query_invoices("What are the unique states mentioned?")
"""

import json
from pathlib import Path
from typing import List, Dict
from rag import GroqLLM

AGGREGATED_FILE = Path("aggregated_invoices.jsonl")


def load_aggregated_invoices() -> List[Dict]:
    """Load all invoices from aggregated_invoices.jsonl."""
    if not AGGREGATED_FILE.exists():
        return []
    
    invoices = []
    with open(AGGREGATED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    invoices.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return invoices


def query_invoices(question: str, llm: GroqLLM = None, max_context_invoices: int = 50) -> str:
    """
    Answer a question using the aggregated invoice dataset.
    
    Args:
        question: natural language question
        llm: GroqLLM instance (creates one if None)
        max_context_invoices: max invoices to include in context (to avoid payload limits)
    
    Returns:
        Answer string from LLM
    """
    invoices = load_aggregated_invoices()
    
    if not invoices:
        return "No aggregated invoice data found. Please run batch processing first."
    
    # Limit context size if needed
    if len(invoices) > max_context_invoices:
        print(f"Note: Dataset has {len(invoices)} invoices, using first {max_context_invoices} for context.")
        context_invoices = invoices[:max_context_invoices]
    else:
        context_invoices = invoices
    
    # Build context
    context = json.dumps(context_invoices, ensure_ascii=False, indent=2)
    
    # Estimate token count
    estimated_tokens = len(context) // 4
    print(f"Context: {len(context_invoices)} invoices, ~{estimated_tokens} tokens")
    
    if estimated_tokens > 6000:
        print("Warning: Large context may cause 413 errors. Consider reducing max_context_invoices.")
    
    # Build prompt
    prompt = f"""You have access to {len(context_invoices)} normalized invoice records (JSON array below). Answer the question using ONLY the data provided. If the answer requires data not present, reply "I don't know".

Invoice data:
{context}

Question: {question}

Answer:"""
    
    # Call LLM
    if llm is None:
        import os
        model_name = os.getenv("GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")
        llm = GroqLLM(
            model_name=model_name,
            temperature=0.1,
            max_tokens=512,
            top_p=0.1,
        )
    
    return llm._call(prompt)
