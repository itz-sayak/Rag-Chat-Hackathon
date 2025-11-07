#!/usr/bin/env python3
"""
Run inference after checkpoint is downloaded
This script assumes checkpoint-800 is already in place
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# Configuration
CHECKPOINT_DIR = Path("/mnt/zone/B/train_validation_inference/A/checkpoint-800")
PDF_FILE = Path("/mnt/zone/B/train_validation_inference/A/pdfs/1_5N2387040#9000580721.pdf")
OUTPUT_DIR = Path("/mnt/zone/B/train_validation_inference/A/outputs")

def print_colored(text, color=""):
    colors = {"green": "\033[0;32m", "red": "\033[0;31m", "yellow": "\033[1;33m", "nc": "\033[0m"}
    print(f"{colors.get(color, '')}{text}{colors['nc']}")

def check_checkpoint():
    """Verify checkpoint exists"""
    print("\n" + "="*60)
    print("Checking Checkpoint...")
    print("="*60)
    
    if not CHECKPOINT_DIR.exists():
        print_colored(f"✗ Checkpoint directory not found: {CHECKPOINT_DIR}", "red")
        return False
    
    required_files = {
        "adapter_config.json": "Adapter configuration",
        "adapter_model.bin": "Model weights"
    }
    
    all_good = True
    for file, desc in required_files.items():
        file_path = CHECKPOINT_DIR / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024*1024)
            print_colored(f"✓ {desc}: {file} ({size_mb:.2f} MB)", "green")
        else:
            print_colored(f"✗ Missing: {file}", "red")
            all_good = False
    
    return all_good

def run_inference():
    """Run inference on the PDF"""
    print("\n" + "="*60)
    print("Running Inference")
    print("="*60)
    
    if not PDF_FILE.exists():
        print_colored(f"✗ PDF not found: {PDF_FILE}", "red")
        return False
    
    print_colored(f"✓ PDF file: {PDF_FILE.name}", "green")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    inference_script = Path("/mnt/zone/B/train_validation_inference/A/inference.py")
    
    print(f"\nProcessing with Donut model...")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    try:
        # Run the inference script
        result = subprocess.run(
            [sys.executable, str(inference_script), str(PDF_FILE), str(OUTPUT_DIR)],
            cwd=str(inference_script.parent),
            check=True
        )
        
        # Check for output
        output_json = OUTPUT_DIR / f"{PDF_FILE.stem}_extracted.json"
        
        if output_json.exists():
            print("\n" + "="*60)
            print_colored("✓ Inference Complete!", "green")
            print("="*60)
            
            print(f"\nResults saved to: {output_json}")
            
            # Display the extracted data
            with open(output_json, 'r') as f:
                data = json.load(f)
            
            print("\n" + "-"*60)
            print("Extracted Data:")
            print("-"*60)
            print(json.dumps(data, indent=2, ensure_ascii=False))
            
            return True
        else:
            print_colored("✗ Output file not created", "red")
            return False
            
    except subprocess.CalledProcessError as e:
        print_colored(f"\n✗ Inference failed with exit code {e.returncode}", "red")
        return False
    except Exception as e:
        print_colored(f"\n✗ Error: {e}", "red")
        return False

def main():
    print("\n" + "="*60)
    print("Invoice Data Extraction")
    print("="*60)
    
    # Step 1: Check checkpoint
    if not check_checkpoint():
        print("\n" + "="*60)
        print_colored("Checkpoint-800 Not Found!", "yellow")
        print("="*60)
        print("\nPlease download it manually:")
        print("1. Visit: https://drive.google.com/drive/folders/13chxoj6QlbZIiFk85rRYInSVKQ7aQ0j8")
        print("2. Download the 'checkpoint-800' folder")
        print(f"3. Place it at: {CHECKPOINT_DIR}")
        print("\nThen run this script again.")
        return 1
    
    # Step 2: Run inference
    if run_inference():
        print("\n" + "="*60)
        print_colored("✓ Success!", "green")
        print("="*60)
        return 0
    else:
        print("\n" + "="*60)
        print_colored("✗ Failed", "red")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
