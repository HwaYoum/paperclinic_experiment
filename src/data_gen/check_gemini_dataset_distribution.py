import json
import os
from collections import Counter
from pathlib import Path

def main():
    # Define paths
    base_dir = Path(__file__).parent
    # Based on the user request, the json file is likely in data/ relative to this script
    json_path = base_dir / "data" / "paperclinic_generated_dataset_gemini.json"
    # The papers directory
    papers_dir = base_dir / "data" / "papers" / "gemini"

    print(f"Checking JSON file: {json_path}")
    print(f"Checking Papers dir: {papers_dir}")

    # Check if files exist
    if not json_path.exists():
        print(f"Error: JSON file not found at {json_path}")
        # We proceed only to list papers if JSON is missing, but cannot compare
        return

    if not papers_dir.exists():
        print(f"Error: Papers directory not found at {papers_dir}")
        return

    # Load JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            # User said "json의 data의 filename"
            if isinstance(content, dict) and "data" in content:
                data = content["data"]
            else:
                # Fallback if the root is the list
                data = content
                
            if not isinstance(data, list):
                print("Error: 'data' field is not a list or JSON root is not a list.")
                return
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    # Get list of PDF files in the directory
    pdf_files = [f.name for f in papers_dir.glob("*.pdf")]
    
    if not pdf_files:
        print("No PDF files found in the papers directory.")
        return

    print(f"Found {len(pdf_files)} PDF files in directory.")

    # Count occurrences in JSON
    # We look for 'filename' key
    json_filenames = []
    for item in data:
        if isinstance(item, dict) and "filename" in item:
            json_filenames.append(item["filename"])
    
    counts = Counter(json_filenames)

    # Check counts
    print(f"\n{'Filename':<40} | {'Count':<5} | {'Status'}")
    print("-" * 75)
    
    # We want to check all files in the directory, and also any extra in the JSON
    all_files = set(pdf_files) | set(counts.keys())
    sorted_files = sorted(list(all_files))

    match_count = 0
    mismatch_count = 0

    for filename in sorted_files:
        count = counts.get(filename, 0)
        
        status_parts = []
        if count == 65:
            status_parts.append("OK")
            match_count += 1
        else:
            status_parts.append("MISMATCH")
            mismatch_count += 1
            
        # Additional context
        if filename not in pdf_files:
             status_parts.append("(In JSON only)")
        elif filename not in counts:
             status_parts.append("(In Dir only)")
             
        status = " ".join(status_parts)
        print(f"{filename:<40} | {count:<5} | {status}")

    print("-" * 75)
    print(f"Summary: {match_count} files correct (65 count), {mismatch_count} mismatches.")

if __name__ == "__main__":
    main()
