import json
import os

def convert_json_to_jsonl(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    print(f"Reading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    data_list = json_data.get("data", [])
    print(f"Found {len(data_list)} items. Converting...")

    with open(output_file, "w", encoding="utf-8") as f:
        for item in data_list:
            # Parse the inner JSON string in 'output'
            try:
                scores = json.loads(item.get("output", "{{}}"))
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse output string for an item. Skipping scores.")
                scores = {}

            # Construct the new format
            new_item = {
                "question": item.get("question", ""),
                "essay": item.get("input", ""),
                "organization": scores.get("2. 구성", 0.0),
                "language": scores.get("3. 언어", 0.0),
                "content": scores.get("1. 내용", 0.0)
            }

            # Write as a single line in JSONL
            f.write(json.dumps(new_item, ensure_ascii=False) + "\n")

    print(f"Conversion complete. Saved to {output_file}")

if __name__ == "__main__":
    input_path = "data/paperclinic_generated_dataset_gpt.json"
    output_path = "data/paperclinic_generated_dataset_gpt.jsonl"
    
    convert_json_to_jsonl(input_path, output_path)
