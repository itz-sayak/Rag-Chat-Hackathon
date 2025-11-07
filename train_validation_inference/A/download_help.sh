#!/bin/bash

# =============================================================================
# Alternative: Download checkpoint-800 using direct links or manual steps
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

CHECKPOINT_DIR="/mnt/zone/B/train_validation_inference/A/checkpoint-800"

echo -e "${YELLOW}================================================================${NC}"
echo -e "${YELLOW}  Checkpoint-800 Download Helper${NC}"
echo -e "${YELLOW}================================================================${NC}"
echo ""
echo "The automatic download failed. Here are your options:"
echo ""

echo -e "${GREEN}OPTION 1: Manual Download (RECOMMENDED)${NC}"
echo "================================================"
echo ""
echo "1. Open this link in your browser:"
echo "   https://drive.google.com/drive/folders/13chxoj6QlbZIiFk85rRYInSVKQ7aQ0j8"
echo ""
echo "2. Sign in to Google if needed"
echo ""
echo "3. Download the 'checkpoint-800' folder:"
echo "   - Click on the folder"
echo "   - Press Ctrl+A to select all files"
echo "   - Right-click and select 'Download'"
echo "   - Or click the download button at the top"
echo ""
echo "4. Extract the downloaded files if they're in a zip:"
echo "   unzip ~/Downloads/checkpoint-800.zip -d /mnt/zone/B/train_validation_inference/A/"
echo ""
echo "5. Or move the folder:"
echo "   mv ~/Downloads/checkpoint-800 /mnt/zone/B/train_validation_inference/A/"
echo ""

echo -e "${GREEN}OPTION 2: If you already have the checkpoint elsewhere${NC}"
echo "================================================"
echo ""
echo "If you've already downloaded checkpoint-800 to another location:"
echo ""
echo "   # Move it"
echo "   mv /path/to/your/checkpoint-800 $CHECKPOINT_DIR"
echo ""
echo "   # Or create a symbolic link"
echo "   ln -s /path/to/your/checkpoint-800 $CHECKPOINT_DIR"
echo ""

echo -e "${GREEN}OPTION 3: Use gcloud CLI (if you have access)${NC}"
echo "================================================"
echo ""
echo "If you have the checkpoint in Google Cloud Storage:"
echo ""
echo "   gsutil -m cp -r gs://your-bucket/checkpoint-800 $CHECKPOINT_DIR"
echo ""

echo -e "${GREEN}OPTION 4: Try individual file download with gdown${NC}"
echo "================================================"
echo ""
echo "If you have direct file IDs for the files in checkpoint-800,"
echo "you can download them individually:"
echo ""
echo "   gdown <file_id> -O $CHECKPOINT_DIR/adapter_config.json"
echo "   gdown <file_id> -O $CHECKPOINT_DIR/adapter_model.bin"
echo ""

echo ""
echo -e "${YELLOW}================================================================${NC}"
echo -e "${YELLOW}After downloading, verify with:${NC}"
echo -e "${YELLOW}================================================================${NC}"
echo ""
echo "   ls -la $CHECKPOINT_DIR"
echo ""
echo "You should see:"
echo "   - adapter_config.json"
echo "   - adapter_model.bin"
echo "   - (possibly other files)"
echo ""
echo -e "${GREEN}Then run:${NC}"
echo "   cd /mnt/zone/B/train_validation_inference/A"
echo "   python inference.py pdfs/1_5N2387040#9000580721.pdf ./outputs"
echo ""
