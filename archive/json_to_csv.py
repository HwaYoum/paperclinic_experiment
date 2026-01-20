import json
import glob
import os
import pandas as pd

def convert_latest_consolidated_json_to_csv():
    # 1. Find all JSON files in the outputs directory
    output_dir = "outputs"
    # Match patterns starting with "consolidated_"
    json_files = glob.glob(os.path.join(output_dir, "consolidated_aes_dataset_*.json"))
    
    if not json_files:
        print(f"No consolidated JSON files found in '{output_dir}/'.")
        return

    # 2. Sort by modification time to get the latest one
    latest_file = max(json_files, key=os.path.getmtime)
    print(f"--- Reading latest results from: {latest_file} ---")

    with open(latest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 3. Process data into a flat list of dictionaries
    flattened_data = []
    
    # Iterate through papers dictionary
    papers_dict = data.get("papers", {})
    
    for paper_title, paper_data in papers_dict.items():
        filename = paper_data.get("meta", {}).get("filename")
        # Context text is now at the paper level, but usually too long for CSV cells.
        # We can include a snippet or omit it. Let's omit full context for clean CSV.
        
        results = paper_data.get("generated_results", [])
        for entry in results:
            target_labels = entry.get("target_labels", {})
            
            row = {
                "id": entry.get("id"),
                "paper_title": paper_title,
                "filename": filename,
                "model_name": entry.get("model_name"),
                "target_level": entry.get("target_level"),
                
                # Flatten Scores
                "score_content": target_labels.get("1. 내용 이해 및 요약"),
                "score_persuasiveness": target_labels.get("2. 설득력"),
                "score_critical_thinking": target_labels.get("3. 비판적 사고 및 학술적 맥락 파악"),
                "score_organization": target_labels.get("4. 구조 및 조직"),
                "score_expression": target_labels.get("5. 표현"),
                "score_formatting": target_labels.get("6. 형식"),
                
                # Output
                "generated_essay": entry.get("generated_essay")
            }
            flattened_data.append(row)

    # 4. Save to CSV
    base_name = os.path.splitext(os.path.basename(latest_file))[0]
    csv_filename = os.path.join(output_dir, f"{base_name}.csv")

    df = pd.DataFrame(flattened_data)
    
    # Sort for better readability (Paper -> Model -> Level)
    if not df.empty:
        df = df.sort_values(by=["paper_title", "model_name", "target_level"])

    df.to_csv(csv_filename, index=False, encoding="utf-8-sig")

    print(f"Successfully converted to CSV: {csv_filename}")
    print(f"Total rows: {len(df)}")

if __name__ == "__main__":
    convert_latest_consolidated_json_to_csv()