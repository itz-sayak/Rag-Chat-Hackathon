"""
Batch processing script for data.json: splits into chunks and extracts structured invoice data.

Usage:
    python batch_process.py

This script:
1. Reads DATA/invoice json/data.json (JSONL format)
2. Chunks records into batches (~6000 chars per batch)
3. Sends each batch to Groq LLM with a JSON extraction prompt
4. Saves normalized invoices to aggregated_invoices.jsonl

The output file can then be queried directly (cheap, fast, structured).
"""

import json
import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from rag import GroqLLM

load_dotenv(override=True)

DATA_FILE = Path("DATA") / "invoice json" / "data.json"
OUTPUT_FILE = Path("aggregated_invoices.jsonl")
CHUNK_SIZE_CHARS = 6000  # conservative chunk size to avoid 413

# Strict JSON extraction prompt
EXTRACTION_PROMPT = """You are a precise invoice data extractor. Given invoice data below, extract and return ONLY a JSON array of normalized invoices. Each invoice must follow this schema:

{
  "invoice_no": "string",
  "invoice_date": "YYYY-MM-DD or empty",
  "vendor_name": "string",
  "vendor_gstin": "string or empty",
  "vendor_pan": "string or empty",
  "client_name": "string or empty",
  "items": [
    {
      "description": "string",
      "quantity": number or null,
      "unit": "string or empty",
      "unit_price": number or null,
      "amount": number or null
    }
  ],
  "subtotal": number or null,
  "tax_amount": number or null,
  "total_amount": number or null,
  "currency": "string (INR, USD, etc.) or empty",
  "source": "string (image field or record id)"
}

Rules:
- Return ONLY the JSON array, no explanatory text.
- If a field is missing, use empty string "" for text fields, null for numbers.
- Preserve the "image" field from input as "source".
- If multiple invoices in the chunk, return all in the array.

Invoice data:
"""


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load JSONL file (one JSON object per line)."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping invalid JSON at line {line_num}: {e}")
    return records


def chunk_records(records: List[Dict], max_chars: int) -> List[List[Dict]]:
    """Group records into chunks with total char size ≤ max_chars."""
    chunks = []
    current_chunk = []
    current_size = 0
    
    for record in records:
        record_str = json.dumps(record, ensure_ascii=False)
        record_size = len(record_str)
        
        # If adding this record would exceed the limit, start a new chunk
        if current_size + record_size > max_chars and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0
        
        current_chunk.append(record)
        current_size += record_size
    
    # Add the last chunk if non-empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def extract_chunk(llm: GroqLLM, chunk: List[Dict], chunk_idx: int) -> List[Dict]:
    """Send a chunk to LLM for extraction and parse the JSON response."""
    chunk_text = json.dumps(chunk, ensure_ascii=False, indent=2)
    prompt = EXTRACTION_PROMPT + "\n" + chunk_text
    
    print(f"[Chunk {chunk_idx}] Processing {len(chunk)} records (~{len(chunk_text)} chars)...")
    
    try:
        response = llm._call(prompt)
        
        # Try to parse JSON from response
        # LLM may wrap in markdown code blocks, so strip those
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        extracted = json.loads(response)
        
        if not isinstance(extracted, list):
            extracted = [extracted]
        
        print(f"[Chunk {chunk_idx}] Extracted {len(extracted)} invoices")
        return extracted
    
    except json.JSONDecodeError as e:
        print(f"[Chunk {chunk_idx}] ERROR: Failed to parse LLM JSON response: {e}")
        print(f"[Chunk {chunk_idx}] Response snippet: {response[:500]}")
        return []
    except Exception as e:
        print(f"[Chunk {chunk_idx}] ERROR: {e}")
        return []


def main():
    print("=" * 60)
    print("Batch Invoice Processing Pipeline")
    print("=" * 60)
    
    # Check input file
    if not DATA_FILE.exists():
        print(f"ERROR: Input file not found: {DATA_FILE}")
        return 1
    
    # Load records
    print(f"\n1. Loading records from {DATA_FILE}...")
    records = load_jsonl(DATA_FILE)
    print(f"   Loaded {len(records)} records")
    
    if not records:
        print("   No records to process.")
        return 0
    
    # Chunk records
    print(f"\n2. Chunking records (max {CHUNK_SIZE_CHARS} chars per chunk)...")
    chunks = chunk_records(records, CHUNK_SIZE_CHARS)
    print(f"   Created {len(chunks)} chunks")
    
    # Initialize LLM
    model_name = os.getenv("GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")
    print(f"\n3. Initializing Groq LLM (model: {model_name})...")
    llm = GroqLLM(
        model_name=model_name,
        temperature=0.0,
        max_tokens=2048,  # Allow sufficient tokens for JSON output
        top_p=0.1,
        system_prompt="You are a precise invoice data extraction assistant.",
    )
    
    # Process chunks
    print(f"\n4. Processing {len(chunks)} chunks...")
    all_invoices = []
    
    for i, chunk in enumerate(chunks, 1):
        extracted = extract_chunk(llm, chunk, i)
        all_invoices.extend(extracted)
    
    # Save aggregated output
    print(f"\n5. Saving {len(all_invoices)} invoices to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for invoice in all_invoices:
            f.write(json.dumps(invoice, ensure_ascii=False) + "\n")
    
    print(f"\n✓ Batch processing complete!")
    print(f"  Total invoices extracted: {len(all_invoices)}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"\nYou can now run queries against the aggregated dataset.")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
