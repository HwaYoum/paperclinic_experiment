import os
import json
import glob
import re
import random
import time
from typing import List, Dict, Tuple
from pypdf import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
from datasets import Dataset
from huggingface_hub import login
import sys

# Ensure project root is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.augmentation_utils import KoreanNoiseInjector

# Load environment variables
load_dotenv()
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GENAI_API_KEY:
    print("Warning: GEMINI_API_KEY not found.")
else:
    genai.configure(api_key=GENAI_API_KEY)

HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize Noise Injector
noise_injector = KoreanNoiseInjector()

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
    # "2. 구성": {
    #     1: "구조가 거의 없거나 완전히 무너져 있으며, 내용이 무작위로 배열된다. 조직화된 글로 보기 어렵다.",
    #     2: "구조가 형식적으로만 존재하며 정보가 부적절한 순서로 제시된다. 단락 간 연결이 부족해 글의 흐름을 따라가기 어렵다.",
    #     3: "기본 구조는 있으나 정보 배열이 불균형하거나 일부 전개가 단절적이다. 흐름은 이해 가능하나 완성도가 낮다.",
    #     4: "구조를 안정적으로 유지하고 단락을 대체로 논리적 순서로 배열한다. 일부 연결이 약간 어색할 수 있으나 전체 흐름은 자연스럽다.",
    #     5: "도입–전개–결론 구조를 명확히 구성하고, 정보를 논문 흐름에 따라 논리적으로 배열하며, 단락 간 관계를 부드럽게 연결한다. 전환 표현을 적절히 사용해 글 전체가 매우 일관적이다."
    # },
    "3. 언어": {
        1: "규범 오류가 매우 많아 글의 이해가 어렵고, 인용·참고문헌이 없거나 전혀 형식 미준수 상태이다. 과제 수행 형식을 거의 따르지 않는다.",
        2: "규범 오류가 빈번하며 인용·참고문헌이 불완전·누락 상태다. 분량·형식 요건을 충족하지 못한다.",
        3: "규범 오류가 다수 보이나 의미 전달에는 큰 지장을 주지 않는다. 인용·참고문헌에 불일치나 누락이 있다. 분량·형식 요건을 부분적으로 준수한다.",
        4: "대부분 정확히 사용하며 소수의 경미한 오류만 보인다. 인용·참고문헌 형식도 대체로 정확하다. 분량·형식 요건을 대체로 준수한다.",
        5: "맞춤법·띄어쓰기·문장부호를 정확히 적용하고 오류가 거의 없다. 인용·참고문헌을 요구 형식에 맞춰 작성하며, 분량 및 형식 요건을 모두 충족한다."
    }
}

ESSAY_QUESTIONS = [
    "이 논문에서 주장하는 핵심 contribution을 논문이 발표된 시기를 기준으로 problem statement와 related work에 대한 분석이 녹아들도록 깔끔하게 작성하세요.",
    "이 논문에서 제시하는 핵심 개념 중 하나를 선택하여 설명하고, 그 개념이 논문의 전체적인 맥락에서 어떤 역할을 하는지 설명하세요.",
    "이 논문에서 제시하는 방법론의 주요 단계를 설명하고, 각 단계가 왜 필요한지 논리적으로 설명하세요.",
    "이 논문의 실험 결과 중 하나를 선택하여 설명하고, 그 결과가 논문의 주장을 어떻게 뒷받침하는지 분석하세요.",
    "이 논문의 한계점이나 향후 연구 방향을 논문 내용을 바탕으로 분석하고 설명하세요."
]

# --- Utility Functions ---

def clean_text(text: str) -> str:
    """Removes special characters except periods, commas, and quotes."""
    text = re.sub(r'[^가-힣a-zA-Z0-9\s.,\'"]', '', text)
    return text

def extract_pdf_context(pdf_path: str, max_pages: int = 5) -> str: ## 실제로 text를 확인하면 성능 향상될 수 있다. 일단 넘어가 / 다시 생각해보니까 직접 사용하는 것은 생성된 만점 대답
    """Extracts text from the first few pages of a PDF."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        num_pages = min(max_pages, len(reader.pages))
        for i in range(num_pages):
            page_text = reader.pages[i].extract_text()
            if page_text:
                text += page_text + "\n\n"
        return clean_text(text)
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def get_gemini_response(prompt: str, model_name: str = "gemini-2.5-flash") -> str:
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API Error: {e}")
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
                if row: sentences.append(row[1].strip()) # 문장 수가 줄던 이유는 csv파일에서 0번열을 가져왔기 때문이다.
        print(f"Loaded {len(sentences)} distractor sentences.")
    except Exception as e:
        print(f"Warning: Could not fetch refined_sentences.csv ({e}). Using fallback.")
        return ["이것은 무작위 문장입니다."] * 10
        
    return sentences if sentences else ["무작위 문장입니다."] * 10

# --- Core Validation & Generation Logic ---

def evaluate_essay(text: str, target_trait: str) -> Dict[str, float]: ## 에세이 평가 / 이거 설마 정수가 아니라 실수로 나오나?
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
        \"{target_trait}\": 점수(1~5)
    }}
    """

    response_text = get_gemini_response(prompt, model_name="gemini-2.5-flash-lite")
    if not response_text: 
        print("response_text is None")
        return None
    
    try:
        clean_response = response_text.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{.*\}", clean_response, re.DOTALL)
        if match:
            json_str = match.group(0)
            scores = json.loads(json_str)
            return {k: float(v) for k, v in scores.items()}
    except:
        pass
    
    print("try part error")
    return None

def apply_case_noise(text: str, noise_type: str, target_score: float, distractor_pool: List[str]) -> Tuple[str, float]:
    sentences = split_sentences(text)
    n_se = len(sentences)
    if n_se == 0: return text, 5.0

    n_sc = int(round(n_se * (5.0 - target_score) / 5.0))
    n_sc = min(n_sc, n_se)
    
    if n_sc == 0: return text, 5.0

    noisy_sentences = sentences[:]

    if noise_type == "Content":
        indices_to_replace = random.sample(range(n_se), n_sc)
        for idx in indices_to_replace:
            noisy_sentences[idx] = random.choice(distractor_pool)

    elif noise_type == "Organization":
        for _ in range(n_sc):
            idx1, idx2 = random.sample(range(n_se), 2)       
            noisy_sentences[idx1], noisy_sentences[idx2] = noisy_sentences[idx2], noisy_sentences[idx1]
        # indices = random.sample(range(n_se), n_sc)
        # selected_sentences = [noisy_sentences[i] for i in indices]
        # random.shuffle(selected_sentences)
        # for i, idx in enumerate(indices):
        #     noisy_sentences[idx] = selected_sentences[i]

    elif noise_type == "Language":
        indices_to_corrupt = random.sample(range(n_se), n_sc)
        for idx in indices_to_corrupt:
            noisy_sentences[idx] = noise_injector.apply_weighted_noise(noisy_sentences[idx])
    
    return ' '.join(noisy_sentences), target_score 

dict_cnt = {
    "Language": {1:0, 2:0, 3:0, 4:0},
    "Organization": {1:0, 2:0, 3:0, 4:0},
    "Content": {1:0, 2:0, 3:0, 4:0}
}
dict_noise_average = {
    "Language": {1:0, 2:0, 3:0, 4:0},
    "Organization": {1:0, 2:0, 3:0, 4:0},
    "Content": {1:0, 2:0, 3:0, 4:0}
}

def generate_validated_noisy_essay( ## 데이터 생성 및 검증 과정
    base_text: str, 
    noise_type: str, 
    target_score: float, 
    distractor_pool: List[str], 
    max_retries: int = 10
) -> str:
    trait_map = {"Language": "3. 언어", "Organization": "2. 구성", "Content": "1. 내용"}
    target_trait = trait_map[noise_type]
    current_noise_param = target_score
    
    best_text = None
    min_diff = float('inf')
    best_noise_param = 0
    last_noisy_text = None
    


    global dict_cnt
    global dict_noise_average
    # print(f"    [Debug] Generating {noise_type} Target: {target_score}")

    for attempt in range(max_retries):
        noisy_text, _ = apply_case_noise(base_text, noise_type, current_noise_param, distractor_pool)
        last_noisy_text = noisy_text
        
        # If noise generation failed to change text (e.g. short text), try forcing param
        # 마지막 noisy_text는 평가하지 않고 반복문 탈출
        if noisy_text == base_text and attempt < max_retries - 1:
             current_noise_param -= 1.0 # Aggressively increase noise
             continue
        # 평가한 점수
        eval_scores = evaluate_essay(noisy_text, target_trait)
        if not eval_scores: 
            # If evaluation fails, we still consider this text as a candidate if we have nothing else
            continue
            
        actual_score = eval_scores.get(target_trait, 5.0)
        print(f"목표 점수: {target_score}, 노이즈 점수: {current_noise_param}, 평가 점수: {actual_score}")
        diff = actual_score - target_score
        
        abs_diff = abs(diff)
        
        # print(f"      Attempt {attempt+1}: Param={current_noise_param:.1f} -> Actual={actual_score} (Diff={diff})")
        
        
        if abs_diff <= 0.7: # Relaxed tolerance / LLM 평가가 정수로 잘 나온다면 0과 0.7은 같은 설정.
            print("      => Best Case")
            dict_noise_average[noise_type][target_score] += current_noise_param
            return noisy_text
            
        if abs_diff < min_diff: 
            min_diff = abs_diff
            best_text = noisy_text
            best_noise_param = current_noise_param
        
        # Feedback Loop
        ## 점수 차이만큼 피드백 주지 않고 1점씩 바꾸는 방법 / 논의의 여지가 있다.
        if diff > 0: # Score too high (too good), need MORE noise (lower param score)
            current_noise_param -= 0.5
        else: # Score too low (too bad), need LESS noise (higher param score)
            current_noise_param += 0.5
            
        ## 왜 루브릭 기반인데 실수로 코드가 작성되어 있냐고!!!
        ## 0.5 -> 1.0
        ## 5.0 -> 4.0 , 5.0이면 4점 데이터를 생성할 때, 노이즈 주입 횟수를 손해봄
        current_noise_param = max(0.5, min(4.5, current_noise_param))

    # Fallback: Return best found, or the last noisy one, or re-generate blind noise
    if best_text:
        dict_cnt[noise_type][target_score] +=1
        dict_noise_average[noise_type][target_score] += best_noise_param
        print(f"      => Returning best match (Diff: {min_diff})")
        return best_text
    
    if last_noisy_text and last_noisy_text != base_text:
        print("      => Validation failed, returning last noisy version.")
        return last_noisy_text
        
    # Final fallback: Force blind noise
    print("      => Validation failed, forcing blind noise.")
    blind_noise, _ = apply_case_noise(base_text, noise_type, target_score, distractor_pool)
    return blind_noise

# --- Main Generator ---

def main():
    start_time = time.time()
    papers_dir = "data/papers"
    pdf_files = glob.glob(os.path.join(papers_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in '{papers_dir}/'. Please add papers.")
        return

    print(f"Found {len(pdf_files)} papers. Starting generation...")
    distractor_pool = get_distractor_sentences()
    
    output_dataset = []
    common_instruction = "다음 학술 에세이를 읽고, 평가 기준(내용, 구성, 언어)에 따라 채점한 뒤 결과를 JSON 형식으로 출력하세요." #이거 장식용인데 음..

    for pdf_path in pdf_files:
        paper_title = os.path.basename(pdf_path)
        print(f"\n=== Processing {paper_title} ===")
        
        context = extract_pdf_context(pdf_path)
        if len(context) < 500:
            print("  Skipping: Text too short.")
            continue
            
        for q_idx, question in enumerate(ESSAY_QUESTIONS):
            print(f"  Question {q_idx+1}: {question[:30]}...")
            
            # 1. Generate Gold Answer (Score 5)
            prompt_gold = f"""
            당신은 해당 분야의 전문가입니다. 아래 논문의 내용을 바탕으로, 질문에 대해 학술적 글쓰기 기준(내용, 구성, 언어)에서 만점(5점)을 받을 수 있는 완벽한 에세이를 작성하세요.
            
            [논문 텍스트]:
            {context}
            
            [질문]:
            {question}
            
            [조건]:
            - 한국어로 작성할 것.
            - 논문의 핵심 요소(연구 목적·개념·방법·결과·의의)를 정확히 식별하고, 중요 정보를 선별하며, 불필요한 내용을 배제하고, 원문 의미를 왜곡 없이 재구성하여 완성도 높은 요약을 제시한다.
            - 도입–전개–결론 구조를 명확히 구성하고, 정보를 논문 흐름에 따라 논리적으로 배열하며, 단락 간 관계를 부드럽게 연결한다. 전환 표현을 적절히 사용해 글 전체가 매우 일관적이다.
            - 문장을 정확히 구성하고 다양한 구조를 자연스럽게 활용하며, 학술적 어조를 일관되게 유지한다. 어휘를 정밀하게 선택해 의미를 선명하게 전달한다.
            - 10~15 문장 내외.
            """
            
            gold_text = get_gemini_response(prompt_gold, model_name="gemini-2.5-flash")
            if not gold_text:
                print("    Failed to generate gold answer.")
                continue
                
            gold_text = clean_text(gold_text)

            # Add Gold Sample
            output_dataset.append({
                "instruction": common_instruction,
                "filename": os.path.basename(pdf_path),
                "question": question,
                "input": gold_text,
                "output": json.dumps({"1. 내용": 5.0, "2. 구성": 5.0, "3. 언어": 5.0, "총점": 5.0}, ensure_ascii=False)
            })
            
            # 2. Generate Noisy Variations (Scores 1-4)
            target_scores = [4.0, 3.0, 2.0, 1.0]
            noise_types = ["Language", "Organization", "Content"]
            
            for n_type in noise_types:
                for score in target_scores:
                    # print(f"    Generating {n_type} score {score}...")
                    noisy_text = generate_validated_noisy_essay(gold_text, n_type, score, distractor_pool)
                    
                    scores = {"1. 내용": 5.0, "2. 구성": 5.0, "3. 언어": 5.0}
                    if n_type == "Language": scores["3. 언어"] = float(score)
                    elif n_type == "Organization": scores["2. 구성"] = float(score)
                    elif n_type == "Content": scores["1. 내용"] = float(score)
                    
                    scores["총점"] = sum(scores.values()) / 3.0
                    
                    output_dataset.append({
                        "instruction": common_instruction,
                        "filename": os.path.basename(pdf_path),
                        "question": question,
                        "input": clean_text(noisy_text),
                        "output": json.dumps(scores, ensure_ascii=False)
                    })

    # Save Locally
    output_file = "data/paperclinic_generated_dataset.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in output_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(output_dataset)} samples to {output_file}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

    # 통계 출력 (Pandas DataFrame 사용)
    import pandas as pd
    stats_data = {
        "Score": [1, 2, 3, 4],
        "Mismatch Count (Language)": [dict_cnt['Language'][i] for i in range(1, 5)],
        "Mismatch Count (Organization)": [dict_cnt['Organization'][i] for i in range(1, 5)],
        "Mismatch Count (Content)": [dict_cnt['Content'][i] for i in range(1, 5)],
        "Avg Noise (Language)": [dict_noise_average['Language'][i]/5*len(pdf_files) for i in range(1, 5)],
        "Avg Noise (Organization)": [dict_noise_average['Organization'][i]/5*len(pdf_files) for i in range(1, 5)],
        "Avg Noise (Content)": [dict_noise_average['Content'][i]/5*len(pdf_files) for i in range(1, 5)]
    }
    df_stats = pd.DataFrame(stats_data)
    print("\n" + "="*50)
    print("Generation Statistics")
    print("-" * 50)
    print(df_stats.to_string(index=False))
    print("="*50 + "\n")

    # # Upload to HF
    # if HF_TOKEN:
    #     print("Uploading to Hugging Face (SJunha/aes-dataset)...")
    #     try:
    #         ds = Dataset.from_list(output_dataset)
    #         ds.push_to_hub("SJunha/aes-dataset", config_name="paperclinic_generated_dataset", split="train")
    #         print("Upload Successful! Config: paperclinic_generated_dataset")
    #     except Exception as e:
    #         print(f"Upload failed: {e}")
    # else:
    #     print("HF_TOKEN missing. Skipping upload.")

if __name__ == "__main__":
    main()
