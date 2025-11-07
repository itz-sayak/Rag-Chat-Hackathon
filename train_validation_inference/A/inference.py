import torch
import json
import os
import re
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel
from peft import PeftModel
from pdf2image import convert_from_path
from typing import Dict, Optional
from typing import Any, List

# -----------------
# CONFIGURATION
# -----------------
BASE_MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"
# --- ⚠️ UPDATE THIS PATH ---
# This must be the path to your checkpoint folder
# Using checkpoint-800 from the Google Drive link:
# https://drive.google.com/drive/folders/13chxoj6QlbZIiFk85rRYInSVKQ7aQ0j8

# Try explicit path first (user-provided), then fall back to A/checkpoint-800, then Google Drive
EXPLICIT_ADAPTER_PATH = "/mnt/zone/B/train_validation_inference/checkpoint-800/checkpoint-800"
FALLBACK_LOCAL_ADAPTER_PATH = "/mnt/zone/B/train_validation_inference/A/checkpoint-800"
GDRIVE_ADAPTER_PATH = "/content/drive/MyDrive/donut_peft_lora_output/checkpoint-800"

if os.path.exists(EXPLICIT_ADAPTER_PATH):
    ADAPTER_PATH = EXPLICIT_ADAPTER_PATH
elif os.path.exists(FALLBACK_LOCAL_ADAPTER_PATH):
    ADAPTER_PATH = FALLBACK_LOCAL_ADAPTER_PATH
elif os.path.exists(GDRIVE_ADAPTER_PATH):
    ADAPTER_PATH = GDRIVE_ADAPTER_PATH
else:
    ADAPTER_PATH = EXPLICIT_ADAPTER_PATH  # Will be checked in load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default Donut task prompt for CORD v2
TASK_PROMPT = os.environ.get("DONUT_TASK_PROMPT", "<s_cord-v2><s_answer>")  # Common Donut prompt structure

# Optional: adjust generation parameters via environment variables
DEFAULT_NUM_BEAMS = int(os.environ.get("DONUT_NUM_BEAMS", 3))
DEFAULT_MAX_LENGTH = int(os.environ.get("DONUT_MAX_LENGTH", 768))
DEFAULT_TOP_P = float(os.environ.get("DONUT_TOP_P", 0.95))
DEFAULT_TEMPERATURE = float(os.environ.get("DONUT_TEMPERATURE", 1.0))
DEFAULT_EARLY_STOP = os.environ.get("DONUT_EARLY_STOP", "true").lower() in {"1","true","yes"}


def load_model(base_model_path: str, adapter_path: str, device: str) -> (PeftModel, AutoProcessor):
    """
    Loads the base Donut model and applies the PEFT adapter.
    This function should only be called ONCE.
    """
    # Check if adapter path exists
    if not os.path.exists(adapter_path):
        print(f"\n{'='*60}")
        print("❌ ERROR: Model checkpoint not found!")
        print(f"{'='*60}")
        print(f"\nLooking for checkpoint at: {adapter_path}")
        print("\nYou need to download the checkpoint-800 model:")
        print("1. Go to: https://drive.google.com/drive/folders/13chxoj6QlbZIiFk85rRYInSVKQ7aQ0j8")
        print("2. Download the 'checkpoint-800' folder")
        print(f"3. Place it at: {adapter_path}")
        print("\nOR set the ADAPTER_PATH variable in inference.py to your checkpoint location")
        print(f"{'='*60}\n")
        raise FileNotFoundError(f"Checkpoint not found at {adapter_path}")
    
    print(f"Loading base processor from {base_model_path}...")
    processor = AutoProcessor.from_pretrained(base_model_path)
    
    print(f"Loading base model from {base_model_path}...")
    base_model = VisionEncoderDecoderModel.from_pretrained(base_model_path)
    
    print(f"Loading PEFT adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print(f"Moving model to {device}...")
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model, processor


def parse_donut_output(output_string: str) -> Dict:
    """
    Attempt to parse Donut's textual output into JSON.
    Strategy:
    1) Strip special tokens like <s_...> and </s_...>
    2) Find the largest balanced {...} substring
    3) Apply light sanitization (remove trailing commas, fix NaN/None)
    4) json.loads
    Returns a dict with error info if parsing fails.
    """
    if not output_string:
        return {"error": "EmptyOutput", "raw_output": output_string}

    # Remove special tokens like <s_cord-v2>, </s_cord-v2>, <s_answer>, etc.
    cleaned = re.sub(r"</?s_[^>]+>", "", output_string)
    cleaned = re.sub(r"</?parse>", "", cleaned)
    # Remove any remaining <> tokens conservatively (keep text inside if needed)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)

    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Find largest balanced JSON block
    def extract_balanced_json(s: str) -> Optional[str]:
        start_idxs = [m.start() for m in re.finditer(r"\{", s)]
        best = None
        for start in start_idxs:
            depth = 0
            for i in range(start, len(s)):
                if s[i] == '{':
                    depth += 1
                elif s[i] == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = s[start:i+1]
                        if best is None or len(candidate) > len(best):
                            best = candidate
                        break
        return best

    json_str = extract_balanced_json(cleaned)
    if not json_str:
        return {"error": "Failed to find JSON", "raw_output": output_string}

    # Light sanitization: remove trailing commas and fix special literals
    json_sanitized = json_str
    # Replace Python/NaN-like tokens with JSON null
    json_sanitized = re.sub(r"\b(NaN|nan|None)\b", "null", json_sanitized)
    # Remove trailing commas before } or ]
    json_sanitized = re.sub(r",\s*([}\]])", r"\1", json_sanitized)

    try:
        return json.loads(json_sanitized)
    except json.JSONDecodeError:
        # As last resort, try to replace single quotes with double quotes if any
        maybe_fixed = json_sanitized.replace("'", '"')
        maybe_fixed = re.sub(r",\s*([}\]])", r"\1", maybe_fixed)
        try:
            return json.loads(maybe_fixed)
        except Exception:
            return {"error": "JSONDecodeError", "raw_output": json_str}


def heuristic_extract_fields(text: str) -> Dict:
    """Heuristic post-processing when structured JSON was not produced.
    Tries to extract common invoice fields using regex patterns.
    Returns a dict of any fields it can guess.
    """
    fields: Dict[str, Optional[str]] = {}
    if not text:
        return {"note": "empty raw text"}

    # Collapse whitespace
    raw = re.sub(r"\s+", " ", text)

    # Currency / total amount patterns (₹, Rs, INR)
    amount_pattern = re.compile(r"(?:₹|Rs\.?|INR)\s*([0-9]{1,3}(?:[, ][0-9]{2,3})*(?:\.[0-9]{2})?)", re.IGNORECASE)
    amounts = [m.group(1).replace(" ", "") for m in amount_pattern.finditer(raw)]
    if amounts:
        # Choose the largest numeric value as probable grand total
        def to_float(a):
            try:
                return float(a.replace(",", ""))
            except Exception:
                return -1
        amounts_sorted = sorted(amounts, key=to_float, reverse=True)
        fields["candidate_amounts"] = amounts_sorted[:5]
        fields["grand_total_guess"] = amounts_sorted[0]

    # Date patterns
    date_patterns = [
        re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),           # YYYY-MM-DD
        re.compile(r"\b(\d{2}/\d{2}/\d{4})\b"),           # DD/MM/YYYY
        re.compile(r"\b(\d{2}-[A-Za-z]{3}-\d{2,4})\b"),    # DD-Mon-YY(YY)
        re.compile(r"\b(\d{2}\.[A-Za-z]{3}\.?-?\d{2,4})\b"), # DD.Mon.YYYY variants
    ]
    dates_found = []
    for dp in date_patterns:
        dates_found.extend(m.group(1) for m in dp.finditer(raw))
    if dates_found:
        fields["dates"] = list(dict.fromkeys(dates_found))[:5]
        fields["invoice_date_guess"] = fields["dates"][0]

    # Invoice number (heuristic: words like Invoice No / Inv / Bill No etc.)
    inv_pattern = re.compile(r"(?:Invoice|Inv|Bill)\s*(?:No\.?|#)?[:\-]?\s*([A-Z0-9\-/]+)", re.IGNORECASE)
    invoice_numbers = [m.group(1) for m in inv_pattern.finditer(raw) if len(m.group(1)) <= 30]
    if invoice_numbers:
        fields["invoice_numbers"] = list(dict.fromkeys(invoice_numbers))
        fields["invoice_number_guess"] = fields["invoice_numbers"][0]

    # GST / Tax percentages
    gst_pattern = re.compile(r"(\b[0-9]{1,2}\.?[0-9]{0,2})%\s*(?:GST|IGST|CGST|SGST)?", re.IGNORECASE)
    gst_rates = [m.group(1) for m in gst_pattern.finditer(raw)]
    if gst_rates:
        fields["gst_rates_detected"] = list(dict.fromkeys(gst_rates))

    # If we found nothing, include a note
    if not fields:
        return {"note": "No heuristic fields matched", "sample_raw_head": raw[:120]}
    return fields


def _to_number(val: Optional[str]) -> Optional[float]:
    if not val:
        return None
    try:
        return float(val.replace(",", "").replace(" ", ""))
    except Exception:
        return None


def _find_labeled_amount(raw: str, labels: List[str]) -> Optional[float]:
    # label before amount
    pat1 = re.compile(rf"(?i)\b({'|'.join(labels)})\b[\s:*-]*[₹RsINR\.]?\s*([0-9]{1,3}(?:[, ][0-9]{2,3})*(?:\.[0-9]{2})?)")
    m = pat1.search(raw)
    if m:
        return _to_number(m.group(2))
    # amount before label
    pat2 = re.compile(rf"(?i)([₹RsINR\.]?\s*[0-9]{1,3}(?:[, ][0-9]{2,3})*(?:\.[0-9]{2})?)\s*(?:is|=|:)?\s*\b({'|'.join(labels)})\b")
    m = pat2.search(raw)
    if m:
        return _to_number(re.sub(r"[₹RsINR\s]", "", m.group(1)))
    return None


def _find_gstin_all(raw: str) -> List[str]:
    gstin_re = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b")
    return list(dict.fromkeys(gstin_re.findall(raw)))


def _find_hsn_codes(raw: str) -> List[str]:
    # Prefer codes near HSN/SAC labels
    near_re = re.compile(r"(?i)(HSN|SAC)[^\d]{0,10}(\d{4,8})")
    codes = [m.group(2) for m in near_re.finditer(raw)]
    if not codes:
        loose_re = re.compile(r"\b\d{4,8}\b")
        codes = [m.group(0) for m in loose_re.finditer(raw)]
    # Deduplicate while preserving order
    return list(dict.fromkeys(codes))[:20]


def build_structured_schema(raw_text: str, parsed_hint: Optional[Dict] = None) -> Dict[str, Any]:
    raw = re.sub(r"\s+", " ", raw_text or "").strip()

    # Start with defaults
    schema: Dict[str, Any] = {
        "invoice_no": None,
        "invoice_date": None,
        "invoice_amount": None,
        "gstin_company": None,
        "gstin_client": None,
        "hsn_codes": [],
        "items": [],
        "summary": {
            "subtotal": None,
            "tax_amount": None,
            "total_amount": None,
        },
    }

    # Use parsed hint if available
    if parsed_hint and isinstance(parsed_hint, dict) and not parsed_hint.get("error"):
        for k_src, k_dst in [
            ("invoice_no", "invoice_no"),
            ("invoice_number", "invoice_no"),
            ("invoiceDate", "invoice_date"),
            ("invoice_date", "invoice_date"),
            ("total", "invoice_amount"),
            ("invoice_amount", "invoice_amount"),
        ]:
            if k_src in parsed_hint and schema.get(k_dst) in (None, ""):
                schema[k_dst] = parsed_hint[k_src]

    # GSTINs
    gstins = _find_gstin_all(raw)
    if gstins:
        schema["gstin_company"] = gstins[0]
        if len(gstins) > 1:
            schema["gstin_client"] = gstins[1]

    # Invoice number (fallback)
    if not schema["invoice_no"]:
        inv_pattern = re.compile(r"(?i)\b(Invoice|Inv|Bill)\s*(No\.?|#)?\s*[:\-]?\s*([A-Z0-9\-/]+)")
        m = inv_pattern.search(raw)
        if m:
            schema["invoice_no"] = m.group(3)

    # Dates
    if not schema["invoice_date"]:
        for dp in [
            re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
            re.compile(r"\b(\d{2}/\d{2}/\d{4})\b"),
            re.compile(r"\b(\d{2}-[A-Za-z]{3}-\d{2,4})\b"),
        ]:
            dm = dp.search(raw)
            if dm:
                schema["invoice_date"] = dm.group(1)
                break

    # Amounts (label-based)
    subtotal = _find_labeled_amount(raw, ["subtotal", "taxable value", "total before tax"]) or None
    tax_amount = _find_labeled_amount(raw, ["tax amount", "gst amount", "total tax"]) or None
    total_amount = _find_labeled_amount(raw, ["total", "invoice total", "grand total", "total amount"]) or None

    # If no labeled totals, fallback to largest currency number as total
    if total_amount is None:
        currency_nums = re.findall(r"(?:₹|Rs\.?|INR)?\s*([0-9]{1,3}(?:[, ][0-9]{2,3})*(?:\.[0-9]{2})?)", raw, flags=re.IGNORECASE)
        nums = sorted((_to_number(x) for x in currency_nums if _to_number(x) is not None), reverse=True)
        if nums:
            total_amount = nums[0]

    schema["summary"]["subtotal"] = subtotal
    schema["summary"]["tax_amount"] = tax_amount
    schema["summary"]["total_amount"] = total_amount

    # Invoice amount mirrors total_amount
    schema["invoice_amount"] = total_amount

    # HSN codes
    schema["hsn_codes"] = _find_hsn_codes(raw)

    # Items (basic heuristic placeholder – empty if not parsed)
    schema["items"] = []

    return schema


def extract_invoice_data(input_path: str, model: PeftModel, processor: AutoProcessor, 
                         output_dir: str = ".", max_target_len: int = 768) -> Optional[Dict]:
    """
    Extracts data from a SINGLE PDF or image file and returns the parsed JSON data.
    Also saves the output to a JSON file.
    Returns the extracted data dictionary on success, None on failure.
    """
    images = []
    base_name = os.path.basename(input_path).split('.')[0]
    
    try:
        if input_path.lower().endswith(".pdf"):
            print(f"Converting PDF to images: {input_path}")
            images = convert_from_path(input_path, dpi=200)
        elif input_path.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
            images = [Image.open(input_path).convert("RGB")]
        else:
            print(f"Skipping unsupported file type: {input_path}")
            return None
    except Exception as e:
        print(f"Error processing file {input_path}: {e}")
        return None

    tokenizer = processor.tokenizer
    all_page_data = []

    print(f"Processing {len(images)} page(s)...")
    for i, img in enumerate(images):
        pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(DEVICE)

        # Prepare Donut task prompt for decoder input
        decoder_input_ids = tokenizer(
            TASK_PROMPT,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(DEVICE)

        # Generation with Donut-recommended parameters
        generated = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=max_target_len,
            early_stopping=DEFAULT_EARLY_STOP,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_beams=DEFAULT_NUM_BEAMS,
            temperature=DEFAULT_TEMPERATURE if DEFAULT_NUM_BEAMS == 1 else None,
            top_p=DEFAULT_TOP_P if DEFAULT_NUM_BEAMS == 1 else None,
            bad_words_ids=[[tokenizer.unk_token_id]] if tokenizer.unk_token_id is not None else None,
            return_dict_in_generate=True,
        )
        
        output_str = tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)[0]
        parsed_json = parse_donut_output(output_str)
        if isinstance(parsed_json, dict) and parsed_json.get("error"):
            # Add heuristic extraction
            parsed_json["heuristic_extraction"] = heuristic_extract_fields(output_str)
            parsed_json["raw_truncated"] = output_str[:300]
        
        if len(images) > 1:
            parsed_json["page_number"] = i + 1
        
        all_page_data.append(parsed_json)

    # Build structured schema using the first page's raw/truncated if available
    # Prefer parsed_json without error; else use raw_truncated / raw_output
    base_for_schema = None
    if len(all_page_data) == 1:
        candidate = all_page_data[0]
        if candidate.get("error"):
            base_for_schema = candidate.get("raw_truncated") or candidate.get("raw_output") or ""
        else:
            base_for_schema = json.dumps(candidate)
    else:
        # Concatenate pages raw_truncated for schema building
        combined_raw = " ".join([
            (p.get("raw_truncated") or p.get("raw_output") or "") for p in all_page_data
        ])
        base_for_schema = combined_raw

    structured = build_structured_schema(base_for_schema, parsed_hint=all_page_data[0] if all_page_data else None)
    final_data = {
        "schema": structured,
        "pages": all_page_data,
    }

    output_filename = os.path.join(output_dir, f"{base_name}_extracted.json")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=4)
    
    print(f"✅ Extraction complete! Saved to: {output_filename}")
    return final_data


def process_single_invoice(file_path: str, model: PeftModel, processor: AutoProcessor, 
                          output_dir: str = "./invoice_outputs", max_target_len: int = 512) -> Optional[Dict]:
    """
    Processes a single invoice file (PDF, PNG, JPG, TIFF).
    Returns the extracted data dictionary on success, None on failure.
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return None
    
    print(f"Processing invoice: {file_path}")
    
    extracted_data = extract_invoice_data(
        input_path=file_path,
        model=model,
        processor=processor,
        output_dir=output_dir,
        max_target_len=max_target_len
    )
    
    if extracted_data:
        print("\n--- Processing Complete ---")
        print("✅ Successfully extracted data from invoice")
        return extracted_data
    else:
        print("\n--- Processing Failed ---")
        print(f"❌ Failed to extract data from: {file_path}")
        return None


# -----------------
# MAIN EXECUTION
# -----------------
if __name__ == "__main__":
    import sys
    from glob import glob

    MAX_TARGET_LENGTH = 768

    def usage():
        print("\nUsage:")
        print("  python inference.py <file_or_folder_path> [output_directory]")
        print("\nExamples:")
        print("  python inference.py ./pdfs ./outputs")
        print("  python inference.py ./pdfs/invoice1.pdf ./outputs")

    if len(sys.argv) < 2:
        usage()
        sys.exit(1)

    INPUT_PATH = sys.argv[1]
    OUTPUT_JSON_DIR = sys.argv[2] if len(sys.argv) > 2 else "./invoice_outputs"

    print(f"\n{'='*60}")
    print("INVOICE DATA EXTRACTION")
    print(f"{'='*60}\n")
    print(f"Input: {INPUT_PATH}")
    print(f"Output Directory: {OUTPUT_JSON_DIR}\n")

    model, processor = load_model(BASE_MODEL_NAME, ADAPTER_PATH, DEVICE)

    # If directory: gather all files
    if os.path.isdir(INPUT_PATH):
        patterns = ["*.pdf", "*.png", "*.jpg", "*.jpeg", "*.tiff"]
        file_list = []
        for pat in patterns:
            file_list.extend(glob(os.path.join(INPUT_PATH, pat)))
        file_list = sorted(file_list)
        if not file_list:
            print(f"❌ No supported files found in directory: {INPUT_PATH}")
            sys.exit(1)
        print(f"Found {len(file_list)} files. Starting batch processing...\n")
        successes = 0
        for fpath in file_list:
            print(f"--- Processing: {fpath} ---")
            res = process_single_invoice(
                file_path=fpath,
                model=model,
                processor=processor,
                output_dir=OUTPUT_JSON_DIR,
                max_target_len=MAX_TARGET_LENGTH
            )
            if res:
                successes += 1
            print()
        print(f"Batch complete. Successful: {successes}/{len(file_list)}")
        sys.exit(0 if successes else 1)
    else:
        if not os.path.exists(INPUT_PATH):
            print(f"❌ Input file not found: {INPUT_PATH}")
            usage()
            sys.exit(1)
        result = process_single_invoice(
            file_path=INPUT_PATH,
            model=model,
            processor=processor,
            output_dir=OUTPUT_JSON_DIR,
            max_target_len=MAX_TARGET_LENGTH
        )
        if result:
            print("\n" + "="*60)
            print("Extracted Data:")
            print("="*60)
            print(json.dumps(result, indent=2))
            sys.exit(0)
        else:
            sys.exit(1)