"""
Quick test script to verify inference.py setup and configuration
Run this before processing actual invoices
"""

import os
import sys

def check_dependencies():
    """Check if all required packages are installed"""
    print("="*60)
    print("Checking Dependencies...")
    print("="*60)
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers',
        'peft': 'PEFT (Parameter-Efficient Fine-Tuning)',
        'pdf2image': 'PDF2Image',
        'PIL': 'Pillow (PIL)'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - NOT FOUND")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("pip install transformers peft pdf2image pillow torch")
        return False
    else:
        print("\n✅ All dependencies installed!")
        return True


def check_poppler():
    """Check if poppler-utils is installed (required for PDF processing)"""
    print("\n" + "="*60)
    print("Checking Poppler Utils (PDF Processing)...")
    print("="*60)
    
    result = os.system("which pdftoppm > /dev/null 2>&1")
    if result == 0:
        print("✅ Poppler-utils installed")
        return True
    else:
        print("❌ Poppler-utils NOT FOUND")
        print("\nInstall with:")
        print("apt-get install poppler-utils")
        return False


def check_cuda():
    """Check CUDA availability"""
    print("\n" + "="*60)
    print("Checking CUDA...")
    print("="*60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU")
            print("   (This will be slower but still work)")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False


def check_model_checkpoint():
    """Check if model checkpoint exists"""
    print("\n" + "="*60)
    print("Checking Model Checkpoint...")
    print("="*60)
    
    # Default path from inference.py
    checkpoint_path = "/content/drive/MyDrive/donut_peft_lora_output/checkpoint-800"
    
    if os.path.exists(checkpoint_path):
        print(f"✅ Checkpoint-800 found at:")
        print(f"   {checkpoint_path}")
        
        # Check for required files
        required_files = ['adapter_config.json', 'adapter_model.bin']
        all_found = True
        
        for file in required_files:
            file_path = os.path.join(checkpoint_path, file)
            if os.path.exists(file_path):
                print(f"   ✅ {file}")
            else:
                print(f"   ❌ {file} - NOT FOUND")
                all_found = False
        
        return all_found
    else:
        print(f"❌ Checkpoint-800 NOT FOUND at:")
        print(f"   {checkpoint_path}")
        print("\nMake sure to:")
        print("1. Mount Google Drive")
        print("2. Download checkpoint-800 from:")
        print("   https://drive.google.com/drive/folders/13chxoj6QlbZIiFk85rRYInSVKQ7aQ0j8")
        print("3. Place it at the correct path")
        return False


def check_google_drive():
    """Check if Google Drive is mounted (for Colab)"""
    print("\n" + "="*60)
    print("Checking Google Drive...")
    print("="*60)
    
    if os.path.exists("/content/drive"):
        if os.path.exists("/content/drive/MyDrive"):
            print("✅ Google Drive mounted")
            return True
        else:
            print("⚠️  /content/drive exists but MyDrive not found")
            print("   Run: from google.colab import drive; drive.mount('/content/drive')")
            return False
    else:
        print("⚠️  Not running in Google Colab or Drive not mounted")
        print("   If using Colab, run: from google.colab import drive; drive.mount('/content/drive')")
        return False


def test_inference_import():
    """Test if inference.py can be imported"""
    print("\n" + "="*60)
    print("Testing inference.py Import...")
    print("="*60)
    
    try:
        # Try to import the functions
        from inference import load_model, process_single_invoice
        print("✅ Successfully imported inference functions")
        return True
    except ImportError as e:
        print(f"❌ Failed to import inference.py")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Import succeeded but error occurred: {e}")
        return False


def print_summary(results):
    """Print final summary"""
    print("\n" + "="*60)
    print("SETUP CHECK SUMMARY")
    print("="*60)
    
    checks = [
        ("Dependencies", results['dependencies']),
        ("Poppler Utils", results['poppler']),
        ("CUDA", results['cuda']),
        ("Google Drive", results['drive']),
        ("Model Checkpoint", results['checkpoint']),
        ("Inference Import", results['import'])
    ]
    
    all_critical_passed = (
        results['dependencies'] and 
        results['poppler'] and 
        results['checkpoint'] and
        results['import']
    )
    
    print("\nStatus:")
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check_name:20s}: {status}")
    
    print("\n" + "="*60)
    if all_critical_passed:
        print("✅ ALL CRITICAL CHECKS PASSED!")
        print("You're ready to run inference.py")
        print("\nUsage:")
        print("  python inference.py /path/to/invoice.pdf")
    else:
        print("❌ SOME CHECKS FAILED")
        print("Please fix the issues above before running inference.py")
    print("="*60)


def main():
    """Run all checks"""
    print("\n" + "🔍 INFERENCE.PY SETUP VERIFICATION")
    print("="*60)
    
    results = {
        'dependencies': check_dependencies(),
        'poppler': check_poppler(),
        'cuda': check_cuda(),
        'drive': check_google_drive(),
        'checkpoint': check_model_checkpoint(),
        'import': test_inference_import()
    }
    
    print_summary(results)


if __name__ == "__main__":
    main()
