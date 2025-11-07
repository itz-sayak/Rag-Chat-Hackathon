"""
Example usage of the inference.py script for extracting data from a single PDF invoice.
"""

from inference import load_model, process_single_invoice, BASE_MODEL_NAME, ADAPTER_PATH, DEVICE

# Example 1: Basic usage with default settings
def example_basic():
    """Process a single invoice with default settings"""
    
    # Load the model (do this once)
    print("Loading model...")
    model, processor = load_model(BASE_MODEL_NAME, ADAPTER_PATH, DEVICE)
    
    # Process a single file
    file_path = "/content/invoice.pdf"
    result = process_single_invoice(
        file_path=file_path,
        model=model,
        processor=processor,
        output_dir="./outputs"
    )
    
    if result:
        print("\nExtracted Data:")
        print(result)
        return result
    else:
        print("Failed to extract data")
        return None


# Example 2: Process multiple files one by one (not batch)
def example_multiple_files():
    """Process multiple files sequentially"""
    
    # Load the model once
    model, processor = load_model(BASE_MODEL_NAME, ADAPTER_PATH, DEVICE)
    
    # List of files to process
    files_to_process = [
        "/content/invoice1.pdf",
        "/content/invoice2.pdf",
        "/content/invoice3.pdf"
    ]
    
    results = {}
    
    for file_path in files_to_process:
        print(f"\n{'='*60}")
        print(f"Processing: {file_path}")
        print(f"{'='*60}")
        
        result = process_single_invoice(
            file_path=file_path,
            model=model,
            processor=processor,
            output_dir="./outputs"
        )
        
        if result:
            results[file_path] = result
    
    return results


# Example 3: Custom max length and output directory
def example_custom_settings():
    """Process with custom settings"""
    
    model, processor = load_model(BASE_MODEL_NAME, ADAPTER_PATH, DEVICE)
    
    result = process_single_invoice(
        file_path="/content/long_invoice.pdf",
        model=model,
        processor=processor,
        output_dir="/content/drive/MyDrive/invoice_results",
        max_target_len=1024  # Longer max length for complex invoices
    )
    
    return result


# Example 4: Process and access specific fields
def example_access_fields():
    """Process invoice and access specific fields"""
    
    model, processor = load_model(BASE_MODEL_NAME, ADAPTER_PATH, DEVICE)
    
    result = process_single_invoice(
        file_path="/content/invoice.pdf",
        model=model,
        processor=processor,
        output_dir="./outputs"
    )
    
    if result:
        # Access fields from the result
        if "document_pages" in result:
            # Multi-page document
            for page in result["document_pages"]:
                page_num = page.get("page_number", "N/A")
                print(f"\n--- Page {page_num} ---")
                print(f"Invoice Number: {page.get('invoice_number', 'N/A')}")
                print(f"Total Amount: {page.get('total', 'N/A')}")
        else:
            # Single-page document
            print(f"Invoice Number: {result.get('invoice_number', 'N/A')}")
            print(f"Total Amount: {result.get('total', 'N/A')}")
            print(f"Vendor: {result.get('vendor', 'N/A')}")
    
    return result


if __name__ == "__main__":
    # Run the basic example
    print("Running basic example...")
    example_basic()
