#!/bin/bash

echo "================================================"
echo "Invoice Extraction Setup and Run Script"
echo "================================================"

# Configuration
PDF_FILE="/mnt/zone/B/train_validation_inference/A/pdfs/1_5N2387040#9000580721.pdf"
OUTPUT_DIR="/mnt/zone/B/train_validation_inference/A/outputs"
CHECKPOINT_PATH="/mnt/zone/B/train_validation_inference/A/checkpoint-800"

echo ""
echo "Step 1: Checking dependencies..."
echo "================================================"

# Check Python packages
python3 -c "import torch" 2>/dev/null && echo "✅ PyTorch installed" || echo "❌ PyTorch NOT installed - run: pip install torch"
python3 -c "import transformers" 2>/dev/null && echo "✅ Transformers installed" || echo "❌ Transformers NOT installed - run: pip install transformers==4.44.2"
python3 -c "import peft" 2>/dev/null && echo "✅ PEFT installed" || echo "❌ PEFT NOT installed - run: pip install peft"
python3 -c "import pdf2image" 2>/dev/null && echo "✅ pdf2image installed" || echo "❌ pdf2image NOT installed - run: pip install pdf2image"
python3 -c "from PIL import Image" 2>/dev/null && echo "✅ Pillow installed" || echo "❌ Pillow NOT installed - run: pip install pillow"

# Check poppler
which pdftoppm >/dev/null 2>&1 && echo "✅ poppler-utils installed" || echo "❌ poppler-utils NOT installed - run: sudo apt-get install poppler-utils"

echo ""
echo "Step 2: Checking model checkpoint..."
echo "================================================"

if [ -d "$CHECKPOINT_PATH" ]; then
    echo "✅ Checkpoint-800 found at: $CHECKPOINT_PATH"
else
    echo "❌ Checkpoint-800 NOT FOUND"
    echo ""
    echo "YOU NEED TO DOWNLOAD THE MODEL FIRST:"
    echo "1. Go to: https://drive.google.com/drive/folders/13chxoj6QlbZIiFk85rRYInSVKQ7aQ0j8"
    echo "2. Download the 'checkpoint-800' folder"
    echo "3. Place it at: $CHECKPOINT_PATH"
    echo ""
    echo "OR if you have it in Google Drive, you can copy it:"
    echo "   - Mount your Google Drive"
    echo "   - Copy from: /path/to/drive/MyDrive/donut_peft_lora_output/checkpoint-800"
    echo "   - To: $CHECKPOINT_PATH"
    echo ""
    exit 1
fi

echo ""
echo "Step 3: Checking PDF file..."
echo "================================================"

if [ -f "$PDF_FILE" ]; then
    echo "✅ PDF file found: $PDF_FILE"
else
    echo "❌ PDF file NOT FOUND: $PDF_FILE"
    exit 1
fi

echo ""
echo "Step 4: Running inference..."
echo "================================================"

# Update the inference.py to use the local checkpoint path
cd /mnt/zone/B/train_validation_inference/A

# Run inference
python3 inference.py "$PDF_FILE" "$OUTPUT_DIR"

echo ""
echo "================================================"
echo "Done! Check output in: $OUTPUT_DIR"
echo "================================================"
