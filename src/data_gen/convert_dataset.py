import os
import json

def custom_round(val):
    """
    Rounds a float to the nearest integer.
    Standard round half up strategy (e.g., 3.5 -> 4, 3.4 -> 3).
    """
    return int(val + 0.5)

def calculate_average(scores):
    """Calculates the average of a list of numbers."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract Prompt and Text
        prompt = data.get('essay_question', {}).get('prompt', "")
        text = data.get('essay_answer', {}).get('text', "")
        
        personal_score = data.get('score', {}).get('personal', {})
        analytic = personal_score.get('analytic', {})
        
        # --- Calculate Scores ---
        
        # 1. Content (content_1 + content_2)
        # "content1+2 평균해서 소숫점 첫재짜리에서 반올림" -> Integer
        c1 = analytic.get('content_1', {}).get('score', [])
        c2 = analytic.get('content_2', {}).get('score', [])
        c_all = c1 + c2
        content_final = custom_round(calculate_average(c_all))
        
        # 2. Organization (organization_1 + organization_2)
        # "organization 1+2 평균해서 소굿점자리 반올림" -> Integer
        o1 = analytic.get('organization_1', {}).get('score', [])
        o2 = analytic.get('organization_2', {}).get('score', [])
        o_all = o1 + o2
        organization_final = custom_round(calculate_average(o_all))
        
        # 3. Expression (expression_1 + expression_2)
        # "expression 1+2해서 소수점 자리 반올림" -> Integer
        e1 = analytic.get('expression_1', {}).get('score', [])
        e2 = analytic.get('expression_2', {}).get('score', [])
        e_all = e1 + e2
        expression_final = custom_round(calculate_average(e_all))
        
        # 4. Holistic
        # "holitic score는 그냥 평균" -> Float
        holistic_raw = personal_score.get('holistic', {}).get('score', [])
        holistic_final = calculate_average(holistic_raw)
        
        return {
            "prompt": prompt,
            "text": text,
            "content": content_final,
            "organization": organization_final,
            "expression": expression_final,
            "holistic": holistic_final
        }
        
    except Exception as e:
        # print(f"Skipping {file_path}: {e}")
        return None

def convert_directory(input_dir, output_file):
    print(f"Converting files in {input_dir}...")
    if not os.path.exists(input_dir):
        print(f"Warning: Directory not found: {input_dir}")
        return

    count = 0
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    result = process_file(file_path)
                    
                    if result:
                        json_line = json.dumps(result, ensure_ascii=False)
                        outfile.write(json_line + "\n")
                        count += 1
                        
                        if count % 1000 == 0:
                            print(f"  Processed {count} files...")

    print(f"Successfully processed {count} files.")
    print(f"Saved to {output_file}")

def main():
    base_path = "27.주제별 글쓰기 평가 데이터/3.개방데이터/1.데이터"
    
    # 1. Convert Training Data
    train_dir = os.path.join(base_path, "Training/02.라벨링데이터")
    train_output = "data/converted_train.jsonl"
    convert_directory(train_dir, train_output)
    
    # 2. Convert Validation Data
    val_dir = os.path.join(base_path, "Validation/02.라벨링데이터")
    val_output = "data/converted_val.jsonl"
    convert_directory(val_dir, val_output)

if __name__ == "__main__":
    main()
