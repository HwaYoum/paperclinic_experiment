import json
import random
import time
import os
import re
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.augmentation_utils import KoreanNoiseInjector

# Load environment variables and configure Gemini
load_dotenv()
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GENAI_API_KEY:
    print("Warning: GEMINI_API_KEY not found.")
else:
    genai.configure(api_key=GENAI_API_KEY)

# Reuse existing classes and functions
noise_injector = KoreanNoiseInjector()

# Re-define Rubric (Must match generate_data.py)
RUBRIC = {
    "1. 내용": {
        1: "논문의 핵심 내용을 파악하지 못하고, 요약이 왜곡되거나 무관한 내용 중심으로 이루어져 과제 목적을 전혀 충족하지 못한다.",
        2: "논문의 주요 개념·결과를 충분히 식별하지 못하고, 요약 과정에서 핵심 정보를 누락·오해하여 원문과 불일치하는 내용이 다수 발생한다.",
        3: "핵심 내용을 부분적으로만 파악하고, 일부 중요한 요소를 누락하거나 모호하게 표현한다. 요약이 가능하지만 불완전하거나 균형이 부족하다.",
        4: "주요 내용을 대체로 정확하게 이해하고 핵심 정보를 적절히 선택해 요약한다. 일부 세부 정보는 단순화될 수 있으나 전체 요지 전달에는 문제가 없다.",
        5: "논문의 핵심 요소(연구 목적·개념·방법·결과·의의)를 정확히 식별하고, 중요 정보를 선별하며, 불필요한 내용을 배제하고, 원문 의미를 왜곡 없이 재구성하여 완성도 높은 요약을 제시한다."
    },
    "2. 구성": {
        1: "구조가 거의 없거나 완전히 무너져 있으며, 내용이 무작위로 배열된다. 조직화된 글로 보기 어렵다.",
        2: "구조 요소가 형식적으로만 존재하며 정보가 부적절한 순서로 제시된다. 단락 간 연결이 부족해 글의 흐름을 따라가기 어렵다.",
        3: "기본 구조는 있으나 정보 배열이 불균형하거나 일부 전개가 단절적이다. 흐름은 이해 가능하나 완성도가 낮다.",
        4: "구조를 안정적으로 유지하고 단락을 대체로 논리적 순서로 배열한다. 일부 연결이 약간 어색할 수 있으나 전체 흐름은 자연스럽다.",
        5: "도입–전개–결론 구조를 명확히 구성하고, 정보를 논문 흐름에 따라 논리적으로 배열하며, 단락 간 관계를 부드럽게 연결한다. 전환 표현을 적절히 사용해 글 전체가 매우 일관적이다."
    },
    "3. 언어": {
        1: "문장 구조에 심각한 오류가 있으며, 어휘 사용이 부적절해 의미 파악이 어렵다. 표현이 비일관·비논리적이다.",
        2: "문장이 어색하거나 불완전해 의미 전달이 자주 흐려진다. 어휘 선택이 부정확하며 학술적 글쓰기 스타일과 부적합한 표현이 많다.",
        3: "문장 구성은 기본적으로 이해 가능하나 모호하거나 단순한 표현이 반복된다. 학술적 문체가 부분적으로 흔들린다.",
        4: "문장을 대체로 명확히 작성하고 어휘 사용이 대부분 적절하다. 표현은 자연스럽지만 일부 문장에서 경미한 어색함이 있을 수 있다.",
        5: "문장을 정확히 구성하고 다양한 구조를 자연스럽게 활용하며, 학술적 어조를 일관되게 유지한다. 어휘를 정밀하게 선택해 의미를 선명하게 전달한다."
    }
}

def clean_text(text: str) -> str:
    """Removes special characters except periods, commas, and quotes."""
    # Keep Korean, English, numbers, whitespace, and . , ' "
    text = re.sub(r'[^가-힣a-zA-Z0-9\s.,\'"]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_gemini_response(prompt: str, model_name: str = "gemini-2.5-flash") -> str:
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return ""

def evaluate_essay(text: str, target_trait: str) -> Dict[str, float]:
    """
    Evaluates only a specific trait of the essay to ensure focused scoring.
    target_trait: "1. 내용", "2. 구성", or "3. 언어"
    """
    # Only provide the rubric for the relevant trait
    trait_rubric = {target_trait: RUBRIC[target_trait]}
    rubric_text = json.dumps(trait_rubric, ensure_ascii=False, indent=2)
    
    prompt = f"""
    당신은 엄격한 학술 에세이 평가자입니다. 아래 [에세이]를 읽고, **오직 [{target_trait}] 항목에 대해서만** 제공된 [채점 기준]에 따라 공정하게 점수를 매기세요.
    다른 평가 요소(내용, 구성, 언어 중 해당하지 않는 것)는 무시하고, 지정된 항목의 기준에만 집중하여 채점하십시오.
    
    [채점 기준]:
    {rubric_text}
    
    [에세이]:
    {text}
    
    [출력 형식]:
    반드시 아래 JSON 형식으로만 출력하세요. 다른 말은 포함하지 마세요.
    {{
        "{target_trait}": 점수(1~5)
    }}
    """
    # Use gemini-2.5-flash-lite for cost-efficient and fast validation
    response_text = get_gemini_response(prompt, model_name="gemini-2.5-flash-lite")
    if not response_text:
        print("    [Evaluation Error] No response from Gemini.")
        return None
    try:
        # Clean up markdown code blocks if present
        clean_response = response_text.replace("```json", "").replace("```", "").strip()
        
        # Extract JSON from response (in case of extra text)
        match = re.search(r"\{.*\}", clean_response, re.DOTALL)
        if match:
            json_str = match.group(0)
            scores = json.loads(json_str)
            return {k: float(v) for k, v in scores.items()}
        else:
            print(f"    [Evaluation Error] JSON not found in response:\n{response_text[:200]}...")
    except Exception as e:
        print(f"    [Evaluation Error] Parse failed: {e}\nResponse: {response_text[:200]}...")
    return None

def split_sentences(text: str) -> List[str]:
    """Splits text into sentences using regex."""
    # Look behind for [.?!] and look ahead for whitespace or end of string
    chunks = re.split(r'(?<=[.?!])\s+', text)
    return [c.strip() for c in chunks if c.strip()]

def get_distractor_sentences() -> List[str]:
    """Loads out-of-domain sentences from refined_sentences.csv on Hugging Face."""
    sentences = []
    
    # Download from Hugging Face Hub
    try:
        from huggingface_hub import hf_hub_download
        csv_path = hf_hub_download(
            repo_id="SJunha/aes-dataset",
            filename="refined_sentences.csv",
            repo_type="dataset"
        )
        print(f"Loaded refined_sentences.csv from: {csv_path}")
    except Exception as e:
        print(f"Warning: Could not fetch refined_sentences.csv from HF ({e}). Using fallback sentences.")
        return ["무작위 문장입니다."] * 10
        
    try:
        import csv
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Assuming no header, or simple text lines. 
            # If there's a header or specific column, adjust accordingly.
            # Reading all rows as potential sentences.
            for row in reader:
                if row:
                    sentences.append(row[0].strip()) # Take the first column
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        
    if not sentences:
         return ["무작위 문장입니다."] * 10
         
    return sentences

def apply_case_noise(text: str, noise_type: str, target_score: float, distractor_pool: List[str]) -> Tuple[str, float]:
    """
    Implements CASE noise injection.
    """
    sentences = split_sentences(text)
    n_se = len(sentences)
    if n_se == 0: return text, 5.0

    # Calculate number of corrupted sentences needed
    n_sc = int(round(n_se * (5.0 - target_score) / 5.0))
    n_sc = min(n_sc, n_se) # Cap at max sentences
    
    if n_sc == 0:
        return text, 5.0

    noisy_sentences = sentences[:] # Copy

    if noise_type == "Content":
        # Substitute randomly-sampled sentences with out-of-domain sentences
        indices_to_replace = random.sample(range(n_se), n_sc)
        for idx in indices_to_replace:
            distractor = random.choice(distractor_pool)
            noisy_sentences[idx] = distractor

    elif noise_type == "Organization":
        # Swap two randomly-sampled sentences.
        for _ in range(n_sc):
            idx1, idx2 = random.sample(range(n_se), 2)
            noisy_sentences[idx1], noisy_sentences[idx2] = noisy_sentences[idx2], noisy_sentences[idx1]

    elif noise_type == "Language":
        # Substitute randomly-sampled sentences into ungrammatical/malformed sentences
        indices_to_corrupt = random.sample(range(n_se), n_sc)
        for idx in indices_to_corrupt:
            # Apply weighted noise to the selected sentence
            noisy_sentences[idx] = noise_injector.apply_weighted_noise(noisy_sentences[idx])
    
    return ' '.join(noisy_sentences), target_score 

def generate_validated_noisy_essay(
    base_text: str, 
    noise_type: str, 
    target_score: float, 
    distractor_pool: List[str], 
    max_retries: int = 3
) -> str:
    """
    Generates a noisy essay and verifies it using Gemini focused on the specific trait.
    """
    
    # Map noise type to rubric key
    trait_map = {
        "Language": "3. 언어",
        "Organization": "2. 구성",
        "Content": "1. 내용"
    }
    target_trait = trait_map[noise_type]
    
    # This parameter controls the intensity of noise in apply_case_noise.
    current_noise_param = target_score
    
    best_text = None
    min_diff = float('inf')
    
    for attempt in range(max_retries):
        # Apply Noise
        noisy_text, _ = apply_case_noise(base_text, noise_type, current_noise_param, distractor_pool)
        
        # Verify - ONLY for the target trait
        eval_scores = evaluate_essay(noisy_text, target_trait)
        if not eval_scores:
            continue
            
        actual_score = eval_scores.get(target_trait, 5.0) # Default to 5 if missing
        diff = actual_score - target_score # Signed difference
        abs_diff = abs(diff)
        
        print(f"      [Noise: {noise_type}] Target: {target_score} | Param: {current_noise_param:.1f} | Actual: {actual_score} (Diff: {diff})")
        
        # Success Condition: Tolerance +/- 1.0
        if abs_diff <= 1.0:
            return noisy_text
            
        # Keep track of best attempt
        if abs_diff < min_diff:
            min_diff = abs_diff
            best_text = noisy_text
        
        # Feedback Loop
        if diff > 0:
            current_noise_param -= 1.0
        else:
            current_noise_param += 1.0
            
        current_noise_param = max(0.0, min(5.0, current_noise_param))

    print(f"      [Warning] Could not match target score exactly. Best Diff: {min_diff}")
    return best_text if best_text else base_text

def load_gold_essays(limit=10):
    gold_essays = []
    
    # Fetch from HF
    try:
        from huggingface_hub import hf_hub_download
        dataset_path = hf_hub_download(
            repo_id="SJunha/aes-dataset",
            filename="train.jsonl",
            repo_type="dataset"
        )
    except Exception as e:
        print(f"Error fetching dataset from HF: {e}")
        return []

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            c = data.get("content", 0)
            o = data.get("organization", 0)
            e = data.get("expression", 0)
            avg = (c + o + e) / 3.0
            
            # Select perfect 5.0 essays
            if avg >= 5.0:
                gold_essays.append(data)
                
    # Shuffle and pick limit
    random.shuffle(gold_essays)
    
    if limit is None:
        return gold_essays
        
    return gold_essays[:limit]

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate augmented dataset from gold essays.")
    parser.add_argument("--limit", type=int, default=None, help="Number of gold essays to sample. If not set, uses all.")
    parser.add_argument("--aug_factor", type=int, default=1, help="Number of variations to generate per essay per noise type.")
    args_p = parser.parse_args()

    print(f"Loading Sample Gold Essays (Limit: {args_p.limit if args_p.limit else 'All'})...")
    gold_samples = load_gold_essays(args_p.limit)
    print(f"Loaded {len(gold_samples)} gold essays.")

    distractor_pool = get_distractor_sentences()
    output_dataset = []
    common_instruction = "다음 학술 에세이를 읽고, 평가 기준(내용, 구성, 언어)에 따라 채점한 뒤 결과를 JSON 형식으로 출력하세요."

    for i, item in enumerate(gold_samples):
        print(f"Processing Essay {i+1} / {len(gold_samples)}...")
        base_text = item['text']
        question = item['prompt']
        q_id = i + 1  # Arbitrary ID

        # 1. Gold Data (5.0)
        output_dataset.append({
            "instruction": common_instruction,
            "question": question,
            "input": clean_text(base_text),
            "output": json.dumps({"1. 내용": 5.0, "2. 구성": 5.0, "3. 언어": 5.0, "총점": 5.0}, ensure_ascii=False)
        })

        # 2. Noisy Variations
        target_scores = [4.0, 3.0, 2.0, 1.0]
        
        for score in target_scores:
            print(f"  Target Score: {score}")
            
            for f in range(args_p.aug_factor):
                if args_p.aug_factor > 1:
                    print(f"    - Augmentation {f+1}/{args_p.aug_factor}")
                
                # Language
                noisy_lang = generate_validated_noisy_essay(base_text, "Language", score, distractor_pool)
                output_dataset.append({
                    "instruction": common_instruction,
                    "question": question,
                    "input": clean_text(noisy_lang),
                    "output": json.dumps({"1. 내용": 5.0, "2. 구성": 5.0, "3. 언어": score, "총점": (10+score)/3}, ensure_ascii=False)
                })
                
                # Organization
                noisy_org = generate_validated_noisy_essay(base_text, "Organization", score, distractor_pool)
                output_dataset.append({
                    "instruction": common_instruction,
                    "question": question,
                    "input": clean_text(noisy_org),
                    "output": json.dumps({"1. 내용": 5.0, "2. 구성": score, "3. 언어": 5.0, "총점": (10+score)/3}, ensure_ascii=False)
                })
                
                # Content
                noisy_cont = generate_validated_noisy_essay(base_text, "Content", score, distractor_pool)
                output_dataset.append({
                    "instruction": common_instruction,
                    "question": question,
                    "input": clean_text(noisy_cont),
                    "output": json.dumps({"1. 내용": score, "2. 구성": 5.0, "3. 언어": 5.0, "총점": (10+score)/3}, ensure_ascii=False)
                })

    output_file = "data/train_gold_sample.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in output_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Done. Saved {len(output_dataset)} entries to {output_file}")

if __name__ == "__main__":
    main()
