import json
import glob
import os
import csv
import pandas as pd

def convert_latest_json_to_csv():
    # 1. Find all JSON files in the outputs directory
    output_dir = "outputs"
    json_files = glob.glob(os.path.join(output_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in '{output_dir}/'.")
        return

    # 2. Sort by modification time to get the latest one
    latest_file = max(json_files, key=os.path.getmtime)
    print(f"--- Reading latest results from: {latest_file} ---")

    with open(latest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 3. Process data into a flat list of dictionaries
    flattened_data = []
    for entry in data.get("data", []):
        row = {
            "id": entry.get("id"),
            "paper_title": entry.get("meta", {}).get("paper_title"),
            "filename": entry.get("meta", {}).get("filename"),
            # Flatten Target Labels
            "score_content": entry.get("generation_config", {}).get("target_labels", {}).get("content"),
            "score_persuasiveness": entry.get("generation_config", {}).get("target_labels", {}).get("persuasiveness"),
            "score_critical_thinking": entry.get("generation_config", {}).get("target_labels", {}).get("critical_thinking"),
            "score_organization": entry.get("generation_config", {}).get("target_labels", {}).get("organization"),
            "score_expression": entry.get("generation_config", {}).get("target_labels", {}).get("expression"),
            "score_formatting": entry.get("generation_config", {}).get("target_labels", {}).get("formatting"),
            # Output
            "generated_essay": entry.get("output", {}).get("generated_essay")
        }
        flattened_data.append(row)

    # 4. Save to CSV
    # Create a CSV filename based on the JSON filename
    base_name = os.path.splitext(os.path.basename(latest_file))[0]
    csv_filename = os.path.join(output_dir, f"{base_name}.csv")

    df = pd.DataFrame(flattened_data)
    df.to_csv(csv_filename, index=False, encoding="utf-8-sig") # utf-8-sig for Excel compatibility

    print(f"Successfully converted to CSV: {csv_filename}")
    print(f"Total rows: {len(df)}")

if __name__ == "__main__":
    convert_latest_json_to_csv()
