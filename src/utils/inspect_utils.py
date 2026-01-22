import os
import json
from collections import Counter, defaultdict

# ---------------------------------------------------------
# 1. Count Data Files (from count_data.py)
# ---------------------------------------------------------
def count_data_files():
    print("\n=== Counting Data Files ===")
    def count_files(root_dir):
        file_count = 0
        dir_counts = {}
        for root, dirs, files in os.walk(root_dir):
            files = [f for f in files if not f.startswith('.')]
            count = len(files)
            if count > 0:
                rel_path = os.path.relpath(root, root_dir)
                dir_counts[rel_path] = count
                file_count += count
        return file_count, dir_counts

    base_path = "27.주제별 글쓰기 평가 데이터/3.개방데이터/1.데이터"
    training_path = os.path.join(base_path, "Training/02.라벨링데이터")
    validation_path = os.path.join(base_path, "Validation/02.라벨링데이터")

    if not os.path.exists(base_path):
        print(f"Path not found: {base_path}")
        return

    train_count, train_details = count_files(training_path)
    val_count, val_details = count_files(validation_path)
    total_count = train_count + val_count

    if total_count > 0:
        train_ratio = (train_count / total_count) * 100
        val_ratio = (val_count / total_count) * 100
    else:
        train_ratio = 0
        val_ratio = 0

    print(f"Training data count: {train_count:,}")
    print(f"Validation data count: {val_count:,}")
    print(f"Total data count: {total_count:,}")
    print(f"Ratio (Train:Val): {train_ratio:.2f}% : {val_ratio:.2f}%")


# ---------------------------------------------------------
# 2. Analyze Scores (from analyze_scores.py)
# ---------------------------------------------------------
def analyze_raw_scores():
    print("\n=== Analyzing Raw Scores (Training Data) ===")
    
    def get_average_score(json_data):
        scores = []
        try:
            holistic_scores = json_data.get('score', {}).get('personal', {}).get('holistic', {}).get('score', [])
            if isinstance(holistic_scores, list):
                scores.extend(holistic_scores)
        except: pass
            
        try:
            analytic = json_data.get('score', {}).get('personal', {}).get('analytic', {})
            for category in analytic.values():
                cat_scores = category.get('score', [])
                if isinstance(cat_scores, list):
                    scores.extend(cat_scores)
        except: pass
        
        scores = [s for s in scores if isinstance(s, (int, float))]
        if not scores: return None
        return sum(scores) / len(scores)

    training_path = "27.주제별 글쓰기 평가 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터"
    if not os.path.exists(training_path):
        print("Training path not found.")
        return

    count_5_0 = 0
    count_4_5 = 0
    count_4_0 = 0
    total_files = 0

    for root, dirs, files in os.walk(training_path):
        for file in files:
            if file.endswith('.json'):
                total_files += 1
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        avg_score = get_average_score(data)
                        if avg_score is not None:
                            if avg_score >= 5.0: count_5_0 += 1
                            if avg_score >= 4.5: count_4_5 += 1
                            if avg_score >= 4.0: count_4_0 += 1
                except: pass

    print(f"Total files processed: {total_files}")
    print(f"Average score >= 5.0: {count_5_0}")
    print(f"Average score >= 4.5: {count_4_5}")
    print(f"Average score >= 4.0: {count_4_0}")


# ---------------------------------------------------------
# 3. Inspect Converted Dataset (Merged Logic)
# ---------------------------------------------------------
def inspect_converted_dataset(jsonl_path="data/converted_dataset.jsonl"):
    print(f"\n=== Inspecting Converted Dataset: {jsonl_path} ===")
    
    if not os.path.exists(jsonl_path):
        print(f"File not found: {jsonl_path}")
        return

    total = 0
    holistic_counts = Counter()
    unique_prompts = set()
    
    # For high score analysis
    high_score_4_5 = 0
    high_score_4_7 = 0
    high_score_4_8 = 0
    high_score_5_0 = 0
    essays_by_prompt = defaultdict(list)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            data = json.loads(line)
            
            # Holistic Distribution
            h_score = data.get("holistic", 0)
            holistic_counts[h_score] += 1
            
            # Unique Prompts
            prompt = data.get("prompt", "")
            if prompt: unique_prompts.add(prompt)
            
            # High Score Check (Avg of 3 components)
            c = data.get("content", 0)
            o = data.get("organization", 0)
            e = data.get("expression", 0)
            avg_score = (c + o + e) / 3.0
            
            if avg_score >= 4.5:
                high_score_4_5 += 1
                essays_by_prompt[prompt].append(data)
            if avg_score >= 4.7:
                high_score_4_7 += 1
            if avg_score >= 4.8:
                high_score_4_8 += 1
            if avg_score >= 5.0:
                high_score_5_0 += 1

    print(f"Total entries: {total}")
    
    print("\n[Holistic Score Distribution]")
    for score in sorted(holistic_counts.keys(), reverse=True):
        print(f"  {score}: {holistic_counts[score]}")

    print(f"\n[Unique Prompts]: {len(unique_prompts)}")
    
    print(f"\n[High Score Analysis (Avg of Content, Org, Expr)]")
    print(f"Avg >= 4.5: {high_score_4_5}")
    print(f"Avg >= 4.7: {high_score_4_7}")
    print(f"Avg >= 4.8: {high_score_4_8}")
    print(f"Avg == 5.0: {high_score_5_0}")
    print("-" * 40)
    print("Distribution of Avg >= 4.5 per Prompt:")
    for prompt, essays in sorted(essays_by_prompt.items()):
        print(f"Prompt: {prompt}")
        print(f"Count: {len(essays)}")
    print("-" * 40)


# ---------------------------------------------------------
# 4. Validate Generated JSONL (from inspect_data.py)
# ---------------------------------------------------------
def validate_generated_jsonl(file_path="data/train.jsonl"):
    print(f"\n=== Validating Generated JSONL: {file_path} ===")
    
    if not os.path.exists(file_path):
        print("File not found.")
        return

    total_lines = 0
    valid_json_lines = 0
    parse_errors = 0
    structure_errors = 0
    output_parse_errors = 0
    
    score_distribution = Counter()
    missing_criteria_counts = Counter()
    
    # Expected keys in the top-level JSON
    required_keys = {"instruction", "input", "output"}
    
    # Expected keys in the 'output' JSON (The rubric criteria)
    expected_criteria = [
        "1. 내용", 
        "2. 구성", 
        "3. 언어", 
        "총점"
    ]

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            line = line.strip()
            if not line:
                continue

            # 1. Validate Line JSON
            try:
                data = json.loads(line)
                valid_json_lines += 1
            except json.JSONDecodeError:
                print(f"[Line {line_num}] Invalid JSON formatting.")
                parse_errors += 1
                continue

            # 2. Validate Top-level Keys
            if not required_keys.issubset(data.keys()):
                missing = required_keys - set(data.keys())
                print(f"[Line {line_num}] Missing top-level keys: {missing}")
                structure_errors += 1
                continue

            # 3. Validate Output Field (Should be a JSON string)
            output_raw = data['output']
            output_data = {}
            
            try:
                if isinstance(output_raw, str):
                    output_data = json.loads(output_raw)
                elif isinstance(output_raw, dict):
                    output_data = output_raw
                else:
                    print(f"[Line {line_num}] 'output' field is neither string nor dict.")
                    output_parse_errors += 1
                    continue
            except json.JSONDecodeError:
                print(f"[Line {line_num}] Could not parse 'output' string as JSON.")
                output_parse_errors += 1
                continue

            # 4. Validate Score Structure
            current_keys = set(output_data.keys())
            missing_criteria = []
            
            for criteria in expected_criteria:
                if criteria not in current_keys:
                    missing_criteria.append(criteria)
            
            if missing_criteria:
                if missing_criteria_counts[tuple(missing_criteria)] < 3:
                    print(f"[Line {line_num}] Missing criteria in output: {missing_criteria}")
                missing_criteria_counts[tuple(missing_criteria)] += 1
            
            # 5. Collect Stats (Total Score)
            if "총점" in output_data:
                try:
                    score = float(output_data["총점"])
                    score_distribution[score] += 1
                except ValueError:
                    print(f"[Line {line_num}] '총점' is not a number: {output_data['총점']}")

    print("\n" + "="*40)
    print("VALIDATION SUMMARY")
    print("="*40)
    print(f"Total Lines Processed: {total_lines}")
    print(f"Valid JSON Lines: {valid_json_lines}")
    print(f"Top-level Structure Errors: {structure_errors}")
    print(f"Output Field Parse Errors: {output_parse_errors}")
    print(f"Criteria Missing Count: {sum(missing_criteria_counts.values())}")
    
    if missing_criteria_counts:
        print("\nMost Common Missing Criteria Combinations:")
        for k, v in missing_criteria_counts.most_common(5):
            print(f"  {k}: {v} times")

    print("\nScore Distribution (Top 10):")
    for score, count in score_distribution.most_common(10):
        print(f"  Score {score}: {count} entries")
    print("="*40)


def main():
    # count_data_files()
    # analyze_raw_scores()
    inspect_converted_dataset()
    # validate_generated_jsonl()

if __name__ == "__main__":
    main()
