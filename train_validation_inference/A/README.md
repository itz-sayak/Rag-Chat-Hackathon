Invoice Extraction — README

Overview
--------
This folder contains scripts to run Donut-based invoice extraction (PEFT/LoRA adapter applied).

What we have done
-----------------
- Prepared `inference.py` to run either on a single file or on a folder containing PDFs/images.
- Added robust parsing logic and a heuristic fallback to extract key invoice fields when the model output isn't strict JSON.
- Created helper scripts to download and run the process (note: Google Drive checkpoint required manual download due to permissions).
- Performed batch inference over `/mnt/zone/B/train_validation_inference/A/pdfs` and stored outputs in `./outputs`.

Primary script to run
---------------------
Use `inference.py`. It accepts either a single file or a folder as the first argument and an optional output directory as the second argument.

Examples:

Process a folder of PDFs:

```bash
python inference.py /mnt/zone/B/train_validation_inference/A/pdfs /mnt/zone/B/train_validation_inference/A/outputs
```

Process a single file:

```bash
python inference.py /mnt/zone/B/train_validation_inference/A/pdfs/1.pdf /mnt/zone/B/train_validation_inference/A/outputs
```

Advanced: set generation parameters via environment variables

```bash
# Increase beams and length
DONUT_NUM_BEAMS=5 DONUT_MAX_LENGTH=1024 DONUT_EARLY_STOP=true python inference.py /path/to/pdfs ./outputs
```

Where outputs are saved
----------------------
- Each input file produces a JSON file named `<basename>_extracted.json` under the specified output directory.
- Each JSON contains two keys: `schema` (the normalized schema we attempted to fill) and `pages` (raw/page-level model outputs and heuristic data).

Recommended next steps
----------------------
1. If you have the original fine-tuning prompt or expected JSON format (training target), provide it — aligning `TASK_PROMPT` in `inference.py` will significantly improve structured output.
2. If outputs are too noisy, consider using OCR (Tesseract) as a fallback and then apply regex/schema parsing on the OCR text.
3. If you want stronger schema extraction (items table parsing), I can implement a table-extraction heuristic and run it on outputs.

Contact / Notes
---------------
If you want me to normalize all existing JSON outputs into the final schema (filling types and numbers), tell me and I'll run a post-processing step that coerces types and fills missing fields using the heuristics.
