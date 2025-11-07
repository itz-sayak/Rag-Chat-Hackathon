#!/bin/bash

echo "================================================"
echo "Installing Dependencies for Invoice Extraction"
echo "================================================"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 not found. Please install Python3 and pip3 first."
    exit 1
fi

echo ""
echo "Installing Python packages..."
echo "This may take several minutes..."
echo ""

# Install required packages
pip3 install --user torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install --user transformers==4.44.2
pip3 install --user datasets accelerate peft
pip3 install --user pdf2image pillow
pip3 install --user sentencepiece

echo ""
echo "Installing system dependencies..."
echo ""

# Check if apt-get is available (for poppler-utils)
if command -v apt-get &> /dev/null; then
    echo "Installing poppler-utils..."
    sudo apt-get update
    sudo apt-get install -y poppler-utils
else
    echo "⚠️  apt-get not found. Please install poppler-utils manually:"
    echo "   - Ubuntu/Debian: sudo apt-get install poppler-utils"
    echo "   - Fedora: sudo dnf install poppler-utils"
    echo "   - macOS: brew install poppler"
fi

echo ""
echo "================================================"
echo "✅ Installation complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Download checkpoint-800 from:"
echo "   https://drive.google.com/drive/folders/13chxoj6QlbZIiFk85rRYInSVKQ7aQ0j8"
echo ""
echo "2. Place it at:"
echo "   /mnt/zone/B/train_validation_inference/A/checkpoint-800"
echo ""
echo "3. Run inference:"
echo "   cd /mnt/zone/B/train_validation_inference/A"
echo "   python3 inference.py pdfs/1_5N2387040#9000580721.pdf ./outputs"
echo ""
