#!/usr/bin/env python3
"""
Complete script to download checkpoint-800 and run inference
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
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/13chxoj6QlbZIiFk85rRYInSVKQ7aQ0j8"

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def print_status(text, status="INFO"):
    colors = {
        "INFO": "\033[1;33m",
        "SUCCESS": "\033[0;32m",
        "ERROR": "\033[0;31m",
        "NC": "\033[0m"
    }
    print(f"{colors.get(status, '')}{text}{colors['NC']}")

def check_checkpoint():
    """Check if checkpoint exists and is valid"""
    if not CHECKPOINT_DIR.exists():
        return False
    
    required_files = ["adapter_config.json", "adapter_model.bin"]
    for file in required_files:
        if not (CHECKPOINT_DIR / file).exists():
            return False
    return True

def install_gdown():
    """Install gdown if not available"""
    try:
        import gdown
        return True
    except ImportError:
        print_status("Installing gdown...", "INFO")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
            return True
        except:
            return False

def download_checkpoint():
    """Attempt to download checkpoint using various methods"""
    print_header("Downloading Checkpoint-800")
    
    # Create parent directory
    CHECKPOINT_DIR.parent.mkdir(parents=True, exist_ok=True)
    
    # Try method 1: gdown
    if install_gdown():
        print_status("Attempting download with gdown...", "INFO")
        try:
            import gdown
            # Try to download folder
            folder_id = "13chxoj6QlbZIiFk85rRYInSVKQ7aQ0j8"
            output_dir = str(CHECKPOINT_DIR)
            
            print_status("Downloading... This may take several minutes.", "INFO")
            gdown.download_folder(id=folder_id, output=output_dir, quiet=False, use_cookies=False)
            
            if check_checkpoint():
                print_status("Download successful!", "SUCCESS")
                return True
        except Exception as e:
            print_status(f"gdown download failed: {e}", "ERROR")
    
    return False

def show_manual_instructions():
    """Show manual download instructions"""
    print_header("Manual Download Required")
    
    print("Automatic download failed. Please download manually:\n")
    print("1. Visit this link in your browser:")
    print(f"   {GDRIVE_FOLDER_URL}\n")
    print("2. Sign in to Google if needed\n")
    print("3. Download the 'checkpoint-800' folder\n")
    print("4. Move/extract to:")
    print(f"   {CHECKPOINT_DIR}\n")
    print("5. Run this script again\n")
    
    print("Expected files in checkpoint-800:")
    print("  - adapter_config.json")
    print("  - adapter_model.bin")
    print("  - (possibly other config files)\n")

def run_inference():
    """Run the inference script"""
    print_header("Running Inference")
    
    # Check PDF exists
    if not PDF_FILE.exists():
        print_status(f"PDF file not found: {PDF_FILE}", "ERROR")
        return False
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    inference_script = CHECKPOINT_DIR.parent / "inference.py"
    
    print_status(f"Processing: {PDF_FILE.name}", "INFO")
    print_status(f"Output to: {OUTPUT_DIR}", "INFO")
    print()
    
    try:
        result = subprocess.run(
            [sys.executable, str(inference_script), str(PDF_FILE), str(OUTPUT_DIR)],
            check=True,
            cwd=str(CHECKPOINT_DIR.parent)
        )
        
        print_header("Inference Complete!")
        
        # Show output
        output_json = OUTPUT_DIR / f"{PDF_FILE.stem}_extracted.json"
        if output_json.exists():
            print_status(f"Results saved to: {output_json}", "SUCCESS")
            print("\nExtracted Data:")
            print("-" * 60)
            with open(output_json, 'r') as f:
                data = json.load(f)
                print(json.dumps(data, indent=2))
        
        return True
        
    except subprocess.CalledProcessError as e:
        print_status(f"Inference failed with error code {e.returncode}", "ERROR")
        return False
    except Exception as e:
        print_status(f"Error: {e}", "ERROR")
        return False

def main():
    print_header("Checkpoint-800 Download and Inference")
    
    # Check if checkpoint already exists
    if check_checkpoint():
        print_status("✓ Checkpoint-800 found and verified!", "SUCCESS")
        
        response = input("\nCheckpoint already exists. Skip download? (Y/n): ").strip().lower()
        if response in ['', 'y', 'yes']:
            print_status("Using existing checkpoint...", "INFO")
        else:
            print_status("Re-downloading checkpoint...", "INFO")
            if CHECKPOINT_DIR.exists():
                import shutil
                shutil.rmtree(CHECKPOINT_DIR)
            if not download_checkpoint():
                show_manual_instructions()
                return 1
    else:
        print_status("Checkpoint not found. Attempting download...", "INFO")
        if not download_checkpoint():
            show_manual_instructions()
            return 1
    
    # Verify checkpoint again
    if not check_checkpoint():
        print_status("Checkpoint verification failed!", "ERROR")
        print("\nPlease ensure checkpoint-800 is properly downloaded to:")
        print(f"  {CHECKPOINT_DIR}\n")
        return 1
    
    # Run inference
    if run_inference():
        print_header("All Done!")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
