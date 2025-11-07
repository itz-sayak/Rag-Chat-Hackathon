#!/bin/bash

# =============================================================================
# Script to Download Checkpoint-800 Model and Run Inference
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CHECKPOINT_DIR="/mnt/zone/B/train_validation_inference/A/checkpoint-800"
PDF_FILE="/mnt/zone/B/train_validation_inference/A/pdfs/1_5N2387040#9000580721.pdf"
OUTPUT_DIR="/mnt/zone/B/train_validation_inference/A/outputs"
INFERENCE_SCRIPT="/mnt/zone/B/train_validation_inference/A/inference.py"

# Google Drive folder ID (from the URL: https://drive.google.com/drive/folders/13chxoj6QlbZIiFk85rRYInSVKQ7aQ0j8)
GDRIVE_FOLDER_ID="13chxoj6QlbZIiFk85rRYInSVKQ7aQ0j8"

echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  Checkpoint-800 Download and Inference Script${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""

# Function to print status
print_status() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if checkpoint already exists
if [ -d "$CHECKPOINT_DIR" ] && [ -f "$CHECKPOINT_DIR/adapter_config.json" ]; then
    print_success "Checkpoint-800 already exists at $CHECKPOINT_DIR"
    echo ""
    read -p "Do you want to re-download it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Skipping download, using existing checkpoint..."
    else
        print_status "Removing existing checkpoint..."
        rm -rf "$CHECKPOINT_DIR"
    fi
fi

# Download checkpoint if not exists
if [ ! -d "$CHECKPOINT_DIR" ] || [ ! -f "$CHECKPOINT_DIR/adapter_config.json" ]; then
    print_status "Checkpoint not found. Downloading from Google Drive..."
    echo ""
    
    # Check if gdown is installed
    if ! command -v gdown &> /dev/null; then
        print_status "Installing gdown (Google Drive downloader)..."
        pip install gdown -q
        print_success "gdown installed"
    fi
    
    # Create parent directory
    mkdir -p "$(dirname "$CHECKPOINT_DIR")"
    
    # Download the folder from Google Drive
    print_status "Downloading checkpoint-800 folder..."
    print_status "This may take several minutes depending on your internet speed..."
    echo ""
    
    # Try to download the entire folder
    cd "$(dirname "$CHECKPOINT_DIR")"
    
    # Method 1: Try gdown folder download
    print_status "Attempting to download with gdown..."
    if gdown --folder "$GDRIVE_FOLDER_ID" -O checkpoint-800 --remaining-ok 2>/dev/null; then
        print_success "Download completed with gdown!"
    else
        print_error "gdown folder download failed."
        echo ""
        echo -e "${YELLOW}================================================================${NC}"
        echo -e "${YELLOW}  MANUAL DOWNLOAD REQUIRED${NC}"
        echo -e "${YELLOW}================================================================${NC}"
        echo ""
        echo "Please download the checkpoint-800 model manually:"
        echo ""
        echo "1. Visit: https://drive.google.com/drive/folders/13chxoj6QlbZIiFk85rRYInSVKQ7aQ0j8"
        echo "2. Download the entire 'checkpoint-800' folder"
        echo "3. Extract/Move it to: $CHECKPOINT_DIR"
        echo ""
        echo "The folder should contain:"
        echo "  - adapter_config.json"
        echo "  - adapter_model.bin"
        echo "  - (and other model files)"
        echo ""
        echo "After downloading, run this script again."
        echo ""
        exit 1
    fi
fi

# Verify checkpoint files exist
echo ""
print_status "Verifying checkpoint files..."

required_files=("adapter_config.json" "adapter_model.bin")
all_found=true

for file in "${required_files[@]}"; do
    if [ -f "$CHECKPOINT_DIR/$file" ]; then
        print_success "Found: $file"
    else
        print_error "Missing: $file"
        all_found=false
    fi
done

if [ "$all_found" = false ]; then
    print_error "Some required files are missing!"
    echo ""
    echo "Please ensure the checkpoint-800 folder contains all necessary files."
    echo "You may need to download it manually from:"
    echo "https://drive.google.com/drive/folders/13chxoj6QlbZIiFk85rRYInSVKQ7aQ0j8"
    exit 1
fi

print_success "All checkpoint files verified!"
echo ""

# Verify PDF file exists
print_status "Checking PDF file..."
if [ ! -f "$PDF_FILE" ]; then
    print_error "PDF file not found: $PDF_FILE"
    exit 1
fi
print_success "PDF file found: $PDF_FILE"
echo ""

# Run inference
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  Running Inference${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""

print_status "Processing invoice with Donut model..."
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the inference script
cd "$(dirname "$INFERENCE_SCRIPT")"

if python inference.py "$PDF_FILE" "$OUTPUT_DIR"; then
    echo ""
    echo -e "${GREEN}================================================================${NC}"
    echo -e "${GREEN}  Inference Completed Successfully!${NC}"
    echo -e "${GREEN}================================================================${NC}"
    echo ""
    
    # Find the output JSON file
    OUTPUT_JSON="$OUTPUT_DIR/$(basename "${PDF_FILE%.*}")_extracted.json"
    
    if [ -f "$OUTPUT_JSON" ]; then
        print_success "Output saved to: $OUTPUT_JSON"
        echo ""
        echo -e "${YELLOW}Extracted Data:${NC}"
        echo -e "${YELLOW}================================================================${NC}"
        cat "$OUTPUT_JSON" | python -m json.tool 2>/dev/null || cat "$OUTPUT_JSON"
        echo ""
    fi
else
    echo ""
    print_error "Inference failed! Check the error messages above."
    exit 1
fi

echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  All Done!${NC}"
echo -e "${GREEN}================================================================${NC}"
