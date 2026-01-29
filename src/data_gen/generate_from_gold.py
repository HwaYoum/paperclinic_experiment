import os
import json
import glob
import re
import random
import time
from typing import List, Dict, Tuple, Any
from dotenv import load_dotenv
import google.generativeai as genai
from datasets import Dataset
from huggingface_hub import login, hf_hub_download
import sys
import concurrent.futures
import threading
from datetime import datetime

# Ensure project root is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.augmentation_utils import KoreanNoiseInjector

# Load environment variables
# load_dotenv()
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GENAI_API_KEY:
    print("Warning: GEMINI_API_KEY not found.")
else:
    genai.configure(api_key=GENAI_API_KEY)

HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize Noise Injector
noise_injector = KoreanNoiseInjector()

# Initialize Lock for thread safety
stats_lock = threading.Lock()

# Rubric Definition
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
        1: "규범 오류가 매우 많아 글의 이해가 어렵고, 인용·참고문헌이 없거나 전혀 형식 미준수 상태이다. 과제 수행 형식을 거의 따르지 않는다.",
        2: "규범 오류가 빈번하며 인용·참고문헌이 불완전·누락 상태다. 분량·형식 요건을 충족하지 못한다.",
        3: "규범 오류가 다수 보이나 의미 전달에는 큰 지장을 주지 않는다. 인용·참고문헌에 불일치나 누락이 있다. 분량·형식 요건을 부분적으로 준수한다.",
        4: "대부분 정확히 사용하며 소수의 경미한 오류만 보인다. 인용·참고문헌 형식도 대체로 정확하다. 분량·형식 요건을 대체로 준수한다.",
        5: "맞춤법·띄어쓰기·문장부호를 정확히 적용하고 오류가 거의 없다. 인용·참고문헌을 요구 형식에 맞춰 작성하며, 분량 및 형식 요건을 모두 충족한다."
    }
}

# --- Utility Functions ---

def clean_text(text: str) -> str:
    """Removes special characters except periods, commas, and quotes."""
    text = re.sub(r'[^가-힣a-zA-Z0-9\s.,\'"]', '', text)
    return text

def get_gemini_response(prompt: str, model_name: str = "gemini-2.5-flash") -> str:
    max_retries = 5
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            # Check for rate limit or quota errors
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "resource" in error_str:
                sleep_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"API Rate limit hit. Retrying in {sleep_time:.2f}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(sleep_time)
            else:
                print(f"Gemini API Error: {e}")
                time.sleep(1)
    
    print("Max retries reached for Gemini API.")
    return ""

def split_sentences(text: str) -> List[str]:
    chunks = re.split(r'(?<=[.?!])\s+', text)
    return [c.strip() for c in chunks if c.strip()]

def get_distractor_sentences() -> List[str]: ## 필요시 경로 수정
    """Loads out-of-domain sentences from Hugging Face or fallback."""
    sentences = []
    try:
        from huggingface_hub import hf_hub_download
        csv_path = hf_hub_download(
            repo_id="SJunha/aes-dataset",
            filename="refined_sentences.csv",
            repo_type="dataset"
        )
        import csv
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row: sentences.append(row[1].strip()) 
        print(f"Loaded {len(sentences)} distractor sentences.")
    except Exception as e:
        print(f"Warning: Could not fetch refined_sentences.csv ({e}). Using fallback.")
        return ["이것은 무작위 문장입니다."] * 10
        
    return sentences if sentences else ["무작위 문장입니다."] * 10

def load_gold_essays(limit=None):
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

# --- Core Validation & Generation Logic ---

def evaluate_essay(text: str, target_trait: str, target_score: float) -> Dict[str, Any]: ## 에세이 평가
    rubric_text = RUBRIC[target_trait][int(target_score)]
    
    prompt = f"""
        당신은 엄격한 학술 에세이 평가 전문가입니다. 
        당신의 임무는 [에세이]가 주어진 [특정 등급 루브릭]에 얼마나 완벽하게 부합(Matching)하는지 '부합도'를 산출하는 것입니다.

        [지침]:
        1. 오직 제공된 [특정 등급 루브릭]의 내용만을 기준으로 판단하십시오.
        2. '부합도 점수'가 100점에 가까울수록 해당 루브릭의 설명과 에세이의 상태가 '완벽히 일치'함을 의미합니다.

        [특정 등급 루브릭]:
        {target_trait} {target_score}점 기준: {rubric_text}

        [에세이]:
        {text}

        [출력 형식]:
        반드시 아래 JSON 형식으로만 응답하십시오.다른 말은 포함하지 마세요.
        {{
            "reasoning": "에세이의 특징과 루브릭 기준을 대조한 상세 분석 (1~2문장)",
            "consistency_score": "루브릭 일치도 점수 (0.00~100.00, 소수점 둘째 자리)"
        }}
        """

    response_text = get_gemini_response(prompt, model_name="gemini-2.5-flash")
    if not response_text: 
        print("response_text is None")
        return None
    
    try:
        clean_response = response_text.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{{.*\}}", clean_response, re.DOTALL)
        if match:
            json_str = match.group(0)
            result = json.loads(json_str)
            if "consistency_score" in result:
                result["consistency_score"] = float(result["consistency_score"])
            return result
    except:
        pass
    
    print("try part error")
    return None

def apply_case_noise(text: str, noise_type: str, target_score: float, distractor_pool: List[str]) -> Tuple[str, float, Dict]:
    sentences = split_sentences(text)
    n_se = len(sentences)
    noise_log = {
        "noise_ratio": 0.0,
        "noise_content_indices": [],
        "noise_organization_swaps": [],
        "noise_language_details": []
    }

    if n_se == 0: return text, 5.0, noise_log

    n_ratio = (5.0 - target_score) / 5.0
    n_sc = int(round(n_se * n_ratio))
    n_sc = min(n_sc, n_se)
    
    noise_log["noise_ratio"] = n_ratio

    if n_sc == 0: return text, 5.0, noise_log

    noisy_sentences = sentences[:]

    if noise_type == "Content":
        indices_to_replace = random.sample(range(n_se), n_sc)
        for idx in indices_to_replace:
            noisy_sentences[idx] = random.choice(distractor_pool)
        noise_log["noise_content_indices"] = indices_to_replace

    elif noise_type == "Organization":
        swaps = []
        for _ in range(n_sc):
            idx1, idx2 = random.sample(range(n_se), 2)       
            noisy_sentences[idx1], noisy_sentences[idx2] = noisy_sentences[idx2], noisy_sentences[idx1]
            swaps.append((idx1, idx2))
        noise_log["noise_organization_swaps"] = swaps

    elif noise_type == "Language":
        indices_to_corrupt = random.sample(range(n_se), n_sc)
        details_list = []
        for idx in indices_to_corrupt:
            original = noisy_sentences[idx]
            # Returns (text, dict_of_errors)
            changed, noise_counts = noise_injector.apply_weighted_noise(original)
            noisy_sentences[idx] = changed
            
            details_list.append({
                "index": idx,
                "original": original,
                "injected_sentence": changed,
                "noise_types": noise_counts
            })
            
        noise_log["noise_language_details"] = details_list
    
    return ' '.join(noisy_sentences), target_score, noise_log

dict_cnt = {
    "Language": {1:0, 2:0, 3:0, 4:0},
    "Organization": {1:0, 2:0, 3:0, 4:0},
    "Content": {1:0, 2:0, 3:0, 4:0}
}
dict_consistency_average = {
    "Language": {1:0, 2:0, 3:0, 4:0},
    "Organization": {1:0, 2:0, 3:0, 4:0},
    "Content": {1:0, 2:0, 3:0, 4:0}
}

def generate_candidates(base_text: str, noise_type: str, target_score: float, distractor_pool: List[str], count: int = 5) -> List[Tuple[str, float, Dict]]:
    candidates = []
    noise_map = {
        4.0 : 4.0,
        3.0 : 3.0,
        2.0 : 2.0,
        1.0 : 1.0
    }
    fixed_noise_param = noise_map[target_score]
    
    for _ in range(count): 
        noisy_text, _, noise_log = apply_case_noise(base_text, noise_type, fixed_noise_param, distractor_pool)
        candidates.append((noisy_text, target_score, noise_log))
    
    return candidates

def evaluate_single_candidate(candidate_info: Dict[str, Any]) -> Dict[str, Any]:
    # candidate_info should contain: text, noise_type, target_score, index, noise_log
    text = candidate_info['text']
    noise_type = candidate_info['noise_type']
    target_score = candidate_info['target_score']
    noise_log = candidate_info['noise_log']
    
    trait_map = {"Language": "3. 언어", "Organization": "2. 구성", "Content": "1. 내용"}
    target_trait = trait_map[noise_type]
    
    eval_result = evaluate_essay(text, target_trait, target_score)
    
    result_data = {
        "text": text,
        "noise_type": noise_type,
        "target_score": target_score,
        "consistency": 0.0,
        "reasoning": "",
        "valid": False,
        "noise_log": noise_log
    }
    
    if eval_result:
        result_data["consistency"] = eval_result.get("consistency_score", 0.0)
        result_data["reasoning"] = eval_result.get("reasoning", "")
        result_data["valid"] = True
    
    print(f"[{noise_type} {target_score}] Consistency {result_data['consistency']}")
    return result_data

# --- Main Generator ---

def main():
    start_time = time.time()
    
    import argparse
    parser = argparse.ArgumentParser(description="Generate augmented dataset from gold essays.")
    parser.add_argument("--limit", type=int, default=None, help="Number of gold essays to sample. If not set, uses all.")
    args_p = parser.parse_args()

    print(f"Loading Sample Gold Essays (Limit: {args_p.limit if args_p.limit else 'All'})...")
    gold_samples = load_gold_essays(args_p.limit)
    print(f"Loaded {len(gold_samples)} gold essays.")
    
    distractor_pool = get_distractor_sentences()
    
    # Metadata construction
    metadata = {
        "generator_model": "gemini-2.5-flash",
        "evaluator_model": "gemini-2.5-flash",
        "evaluation_method": "First, load gold baseline data (5 points) from HuggingFace. Subsequently, derive data for scores ranging from 4 down to 1 by injecting targeted noise mapped to each specific trait. During the generation phase, produce five candidates for each score level and employ an LLM to select the sample that demonstrates the highest alignment with the rubric descriptions.",
        "noise_method": "Content: Insertion of irrelevant sentences; Organization: Rearrangement of sentence order; Language: Induction of grammatical errors.",
        "types_of_language_errors": "spacing(WS), spelling(SPELL), josa(PART), ending(END), conjugation(CONJ), word order(WO)",
        "evaluation_prompt": """
당신은 엄격한 학술 에세이 평가 전문가입니다. 당신의 임무는 [에세이]가 주어진 [특정 등급 루브릭]에 얼마나 완벽하게 부합(Matching)하는지 '부합도'를 산출하는 것입니다. [지침]: 1. 오직 제공된 [특정 등급 루브릭]의 내용만을 기준으로 판단하십시오. 2. '부합도 점수'가 100점에 가까울수록 해당 루브릭의 설명과 에세이의 상태가 '완벽히 일치'함을 의미합니다. [특정 등급 루브릭]: {target_trait} {target_score}점 기준: {rubric_text} [에세이]: {text} [출력 형식]: 반드시 아래 JSON 형식으로만 응답하십시오.다른 말은 포함하지 마세요. { \"reasoning\": \"에세이의 특징과 루브릭 기준을 대조한 상세 분석 (1~2문장)\", \"consistency_score\": \"루브릭 일치도 점수 (0.00~100.00, 소수점 둘째 자리)\" } """,
        "rubric": RUBRIC
    }
    
    output_file = "data/paperclinic_generated_from_gold.json"
    output_datas = []
    
    if not os.path.exists(output_file):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"metadata": metadata, "data": []}, f, ensure_ascii=False, indent=2)
    
    common_instruction = "다음 학술 에세이를 읽고, 평가 기준(내용, 구성, 언어)에 따라 채점한 뒤 결과를 JSON 형식으로 출력하세요."

    for i, item in enumerate(gold_samples):
        print(f"\n=== Processing Essay {i+1} / {len(gold_samples)} ===")
        
        # Gold Data from HF
        base_text = item['text']
        question = item['prompt']
        filename = item.get('filename', f"gold_sample_{i+1}")

        base_text = clean_text(base_text)

        gold_entry = {
            "instruction": common_instruction,
            "filename": filename,
            "question": question,
            "input": base_text,
            "output": json.dumps({"1. 내용": 5.0, "2. 구성": 5.0, "3. 언어": 5.0, "총점": 5.0}, ensure_ascii=False),
            "timestamp": datetime.now().isoformat(),
            "is_original": True,
            "noise_ratio": 0.0,
            "noise_content_indices": [],
            "noise_organization_swaps": [],
            "noise_language_details": []
        }
        output_datas.append(gold_entry)
        
        # 2. Generate Noisy Variations
        noise_types = ["Organization", "Language", "Content"]
        target_scores = [4.0, 3.0, 2.0, 1.0]
        
        all_tasks = []
        for n_type in noise_types:
            for score in target_scores:
                all_tasks.append((n_type, score))
        
        chunk_size = 1
        
        for i in range(0, len(all_tasks), chunk_size):
            tasks_chunk = all_tasks[i : i + chunk_size]
            
            candidates_to_evaluate = []
            
            for n_type, score in tasks_chunk:
                generated_candidates = generate_candidates(base_text, n_type, score, distractor_pool, count=5)
                for text, _, noise_log in generated_candidates:
                    candidates_to_evaluate.append({
                        "text": text,
                        "noise_type": n_type,
                        "target_score": score,
                        "noise_log": noise_log
                    })
            
            print(f"    Processing batch {i//chunk_size + 1}: {len(candidates_to_evaluate)} evaluations...")
            
            results = []
            # Use max_workers=10 for parallelism as requested (same as generate_dataset_v2.py)
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(evaluate_single_candidate, c) for c in candidates_to_evaluate]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        res = future.result()
                        results.append(res)
                    except Exception as e:
                        print(f"Error in parallel execution: {e}")
            
            grouped_results = {}
            for task in tasks_chunk:
                grouped_results[task] = []
            
            for res in results:
                key = (res['noise_type'], res['target_score'])
                if key in grouped_results:
                    grouped_results[key].append(res)
            
            for n_type, score in tasks_chunk:
                task_candidates = grouped_results[(n_type, score)]
                
                if not task_candidates:
                    print(f"      No results for {n_type} {score}")
                    continue
                    
                task_candidates.sort(key=lambda x: x['consistency'], reverse=True)
                best = task_candidates[0]
                
                best_text = best['text']
                best_reasoning = best['reasoning']
                best_consistency = best['consistency']
                best_log = best['noise_log']
                
                with stats_lock:
                    dict_consistency_average[n_type][int(score)] += best_consistency
                    dict_cnt[n_type][int(score)] += 1
                
                print(f"      [{n_type} {score}] Best Consistency: {best_consistency}")
                
                scores = {"1. 내용": 5.0, "2. 구성": 5.0, "3. 언어": 5.0}
                if n_type == "Language": scores["3. 언어"] = float(score)
                elif n_type == "Organization": scores["2. 구성"] = float(score)
                elif n_type == "Content": scores["1. 내용"] = float(score)
                
                scores["총점"] = round(sum(scores.values()) / 3.0, 2)
                
                new_item = {
                    "instruction": common_instruction,
                    "filename": filename,
                    "question": question,
                    "input": clean_text(best_text),
                    "output": json.dumps(scores, ensure_ascii=False),
                    "reasoning": best_reasoning,
                    "consistency_score": best_consistency,
                    "timestamp": datetime.now().isoformat(),
                    "is_original": False,
                    "noise_ratio": best_log["noise_ratio"],
                    "noise_content_indices": best_log["noise_content_indices"],
                    "noise_organization_swaps": best_log["noise_organization_swaps"],
                    "noise_language_details": best_log["noise_language_details"]
                }
                output_datas.append(new_item)

            # Incremental Save after each batch
            try:
                current_full_data = {
                    "metadata": metadata,
                    "data": output_datas
                }
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(current_full_data, f, ensure_ascii=False, indent=2)
                print(f"      [Auto-Save] Updated {output_file} ({len(output_datas)} samples).")
            except Exception as e:
                print(f"      [Auto-Save Failed] {e}")

    print(f"\nFinal Save: {len(output_datas)} samples to {output_file}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

    # 통계 출력 (Pandas DataFrame 사용)
    import pandas as pd
    stats_data = {
        "Score": [1, 2, 3, 4],
        "Count (Language)": [dict_cnt['Language'][i] for i in range(1, 5)],
        "Count (Organization)": [dict_cnt['Organization'][i] for i in range(1, 5)],
        "Count (Content)": [dict_cnt['Content'][i] for i in range(1, 5)],
        "Avg Consistency (Language)": [dict_consistency_average['Language'][i]/dict_cnt['Language'][i] if dict_cnt['Language'][i] > 0 else 0 for i in range(1, 5)],
        "Avg Consistency (Organization)": [dict_consistency_average['Organization'][i]/dict_cnt['Organization'][i] if dict_cnt['Organization'][i] > 0 else 0 for i in range(1, 5)],
        "Avg Consistency (Content)": [dict_consistency_average['Content'][i]/dict_cnt['Content'][i] if dict_cnt['Content'][i] > 0 else 0 for i in range(1, 5)]
    }
    df_stats = pd.DataFrame(stats_data)
    print("\n" + "="*50)
    print("Generation Statistics")
    print("-" * 50)
    print(df_stats.to_string(index=False))
    print("="*50 + "\n")

    # Upload to Hugging Face (Commented Out)
    # hf_token = os.getenv("HF_TOKEN")
    # if hf_token:
    #     print("Logging in to Hugging Face...")
    #     # login(token=hf_token)
    #     
    #     print("Uploading to Hugging Face Hub (SJunha/aes-dataset)...")
    #     try:
    #         ds = Dataset.from_list(output_datas) # Note: structure changed, might need adaptation if uncommented
    #         ds.push_to_hub("SJunha/aes-dataset", config_name="gold_augmented", split="train")
    #         print("Successfully uploaded to SJunha/aes-dataset (config: gold_augmented)!")
    #     except Exception as e:
    #         print(f"Upload failed: {e}")
    # else:
    #     print("HF_TOKEN not found. Skipping upload to Hugging Face.")

if __name__ == "__main__":
    main()