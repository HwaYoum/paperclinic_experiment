import os
import json
import random
import re
import glob
import time
from typing import List, Dict, Tuple
from pypdf import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
from augmentation_utils import KoreanNoiseInjector

# Load environment variables
load_dotenv()

# Configure Gemini
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GENAI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in environment variables.")
else:
    genai.configure(api_key=GENAI_API_KEY)

# Initialize Noise Injector (Loads Okt once)
noise_injector = KoreanNoiseInjector()

# ---------------------------------------------------------
# 1. Rubric Definition
# ---------------------------------------------------------
RUBRIC = {
    "1. 내용 이해 및 요약": {
        1: "논문의 핵심 내용을 파악하지 못하고, 요약이 왜곡되거나 무관한 내용 중심으로 이루어져 과제 목적을 전혀 충족하지 못한다.",
        2: "논문의 주요 개념·결과를 충분히 식별하지 못하고, 요약 과정에서 핵심 정보를 누락·오해하여 원문과 불일치하는 내용이 다수 발생한다.",
        3: "핵심 내용을 부분적으로만 파악하고, 일부 중요한 요소를 누락하거나 모호하게 표현한다. 요약이 가능하지만 불완전하거나 균형이 부족하다.",
        4: "주요 내용을 대체로 정확하게 이해하고 핵심 정보를 적절히 선택해 요약한다. 일부 세부 정보는 단순화될 수 있으나 전체 요지 전달에는 문제가 없다.",
        5: "논문의 핵심 요소(연구 목적·개념·방법·결과·의의)를 정확히 식별하고, 중요 정보를 선별하며, 불필요한 내용을 배제하고, 원문 의미를 왜곡 없이 재구성하여 완성도 높은 요약을 제시한다."
    },
    "2. 설득력": {
        1: "논증이 거의 존재하지 않거나, 주제에서 벗어나거나 무의미하며, 설득 목적이 불분명하다. 표현에 오류가 많고, 독자를 전혀 고려하지 않는다.",
        2: "논거는 있으나 약하거나 불충분하며, 주장과 근거 간 연결이 모호하다. 일부 논점은 타당하지만 독자 설득에는 실패한다. 감정적 표현 또는 논리 오류가 자주 나타난다.",
        3: "주장과 근거의 구조는 기본적으로 갖추었고 일부 설득 전략이 사용된다. 그러나 예시나 구체성 부족, 설득 목적 전달의 일관성에서 아쉬움이 있다.",
        4: "주장과 논거가 대체로 명확하게 연결되어 있으며, 수사적 장치나 독자 고려도 적절하다. 단점이 일부 있으나 설득 목적은 대체로 달성된다.",
        5: "주장의 타당성, 풍부한 근거, 구조적 논리성, 독자 인식, 수사 전략(예시, 비유, 반박 등)이 유기적으로 작동하여 독자가 자연스럽게 설득되도록 한다. 설득 목적이 매우 명확하고 효과적이다."
    },
    "3. 비판적 사고 및 학술적 맥락 파악": {
        1: "논문을 분석하지 못하고, 타당성·의의·한계를 전혀 판단하지 못하며, 무비판적으로 내용을 반복하거나 왜곡한다.",
        2: "분석·평가가 거의 이루어지지 않고 연구 내용을 나열하거나 피상적으로 반복한다. 학술적 맥락 이해가 부족하다.",
        3: "내용 분석이 표면적이며 결과를 단순 기술한다. 일부 평가·의미 언급은 있으나 일관성이 부족하다.",
        4: "핵심 요소를 적절히 분석하고 연구의 강점·한계를 식별한다. 결과의 의미를 설명하지만 분석의 깊이는 약간 제한된다.",
        5: "논문의 주장·방법·결과를 정확히 분석하고 타당성을 평가하며, 한계·전제·의의를 명확히 제시한다. 관련 이론·연구와 통합하거나 연결해 고차원적 해석을 제시한다."
    },
    "4. 구조 및 조직": {
        1: "구조가 거의 없거나 완전히 무너져 있으며, 내용이 무작위로 배열된다. 조직화된 글로 보기 어렵다.",
        2: "구조 요소가 형식적으로만 존재하며 정보가 부적절한 순서로 제시된다. 단락 간 연결이 부족해 글의 흐름을 따라가기 어렵다.",
        3: "기본 구조는 있으나 정보 배열이 불균형하거나 일부 전개가 단절적이다. 흐름은 이해 가능하나 완성도가 낮다.",
        4: "구조를 안정적으로 유지하고 단락을 대체로 논리적 순서로 배열한다. 일부 연결이 약간 어색할 수 있으나 전체 흐름은 자연스럽다.",
        5: "도입–전개–결론 구조를 명확히 구성하고, 정보를 논문 흐름에 따라 논리적으로 배열하며, 단락 간 관계를 부드럽게 연결한다. 전환 표현을 적절히 사용해 글 전체가 매우 일관적이다."
    },
    "5. 표현": {
        1: "문장 구조에 심각한 오류가 있으며, 어휘 사용이 부적절해 의미 파악이 어렵다. 표현이 비일관·비논리적이다.",
        2: "문장이 어색하거나 불완전해 의미 전달이 자주 흐려진다. 어휘 선택이 부정확하며 학술적 글쓰기 스타일과 부적합한 표현이 많다.",
        3: "문장 구성은 기본적으로 이해 가능하나 모호하거나 단순한 표현이 반복된다. 학술적 문체가 부분적으로 흔들린다.",
        4: "문장을 대체로 명확히 작성하고 어휘 사용이 대부분 적절하다. 표현은 자연스럽지만 일부 문장에서 경미한 어색함이 있을 수 있다.",
        5: "문장을 정확히 구성하고 다양한 구조를 자연스럽게 활용하며, 학술적 어조를 일관되게 유지한다. 어휘를 정밀하게 선택해 의미를 선명하게 전달한다."
    },
    "6. 형식": {
        1: "규범 오류가 매우 많아 글의 이해가 어렵고, 인용·참고문헌이 없거나 전혀 형식 미준수 상태이다. 과제 수행 형식을 거의 따르지 않는다.",
        2: "규범 오류가 빈번하며 인용·참고문헌이 불완전·누락 상태다. 분량·형식 요건을 충족하지 못한다.",
        3: "규범 오류가 다수 보이나 의미 전달에는 큰 지장을 주지 않는다. 인용·참고문헌에 불일치나 누락이 있다. 분량·형식 요건을 부분적으로 준수한다.",
        4: "대부분 정확히 사용하며 소수의 경미한 오류만 보인다. 인용·참고문헌 형식도 대체로 정확하다. 분량·형식 요건을 대체로 준수한다.",
        5: "맞춤법·띄어쓰기·문장부호를 정확히 적용하고 오류가 거의 없다. 인용·참고문헌을 요구 형식에 맞춰 작성하며, 분량 및 형식 요건을 모두 충족한다."
    }
}


# ---------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------

def extract_pdf_context(pdf_path: str) -> str:
    try:
        reader = PdfReader(pdf_path)
        text = ""
        # Read all pages of the PDF
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        return preprocess_text(text)
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def preprocess_text(text: str) -> str:
    # 1. Remove PDF specific artifacts (e.g. (cid:123))
    text = re.sub(r'\(cid:\d+\)', '', text)
    # 2. Replace multiple spaces/tabs with single space
    text = re.sub(r'[ \t]+', ' ', text)
    # 3. Remove non-printable characters
    text = "".join(ch for ch in text if ch.isprintable() or ch in '\n')
    # 4. Limit consecutive newlines to max 2
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def get_gemini_response(prompt: str, model_name: str = "gemini-2.5-flash") -> str:
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return ""

def split_sentences(text: str) -> List[str]:
    """Splits text into sentences using regex."""
    # Look behind for [.?!] and look ahead for whitespace or end of string
    chunks = re.split(r'(?<=[.?!])\s+', text)
    return [c.strip() for c in chunks if c.strip()]

def get_distractor_sentences(count: int = 20) -> List[str]:
    """Generates out-of-domain sentences for Content noise."""
    prompt = f"""
    한국어 학술 에세이에 사용될 법한 문장 {count}개를 작성하세요.
    단, 주제는 '심해 생물의 생태', '조선시대 도자기의 특징', '양자역학의 기초' 등 
    지금 분석 중인 논문과 전혀 상관없는 무작위 주제여야 합니다.
    각 문장은 독립적이어야 하며, 번호 없이 줄바꿈으로 구분해서 출력하세요.
    """
    response = get_gemini_response(prompt)
    if not response:
        # Fallback sentences
        return [
            "심해 생물은 고압 환경에 적응하기 위해 독특한 신체 구조를 발달시켰다.",
            "조선 백자의 순백색은 당시 유교 사회의 청렴 결백 사상을 반영한다.",
            "양자 중첩 현상은 미시 세계에서 입자가 동시에 여러 상태에 존재함을 의미한다.",
            "기후 변화는 전 지구적 해수면 상승을 초래하고 있다.",
            "인공지능의 윤리적 문제는 기술 발전 속도에 비해 논의가 부족한 실정이다."
        ] * (count // 5 + 1)
    
    return [line.strip() for line in response.split('\n') if line.strip()]

# ---------------------------------------------------------
# 3. Step 1: Gold Data Generation
# ---------------------------------------------------------

def generate_gold_essay(source_text: str, question: str) -> str:
    # Constructing the prompt with Rubric Level 5 descriptions
    rubric_instructions = ""
    for trait, criteria in RUBRIC.items():
        rubric_instructions += f"- {trait}: {criteria[5]}\n"

    prompt = f"""
    당신은 한국어 학술 작문 전문가입니다. 아래 [논문 텍스트]를 읽고, [질문]에 대한 학술 에세이를 작성하세요.
    다음의 [평가 기준(Level 5 - 만점)]을 완벽하게 충족해야 합니다.
    
    [논문 텍스트]: {source_text} 
    
    [질문]: {question}
    
    [평가 기준(Level 5 - 만점)]:
    {rubric_instructions}
    
    [추가 필수 조건]:
    1. 인용 시 반드시 '(홍길동, 2023)' 또는 '홍길동(2023)에 따르면' 형식을 엄격히 준수할 것. (형식 만점 기준의 핵심)
    2. 분량은 500자 내외.
    """
    return get_gemini_response(prompt)

# ---------------------------------------------------------
# 4. Step 2: CASE (Corruption-based Augmentation Strategy)
# ---------------------------------------------------------

def apply_case_noise(text: str, noise_type: str, target_score: float, distractor_pool: List[str]) -> Tuple[str, float]:
    """
    Implements CASE:
    n(Sc) = round( n(SE) * (5.0 - xi) / 5.0 )
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
        # using Weighted Random Mix of 6 error types.
        indices_to_corrupt = random.sample(range(n_se), n_sc)
        for idx in indices_to_corrupt:
            # Apply weighted noise to the selected sentence
            noisy_sentences[idx] = noise_injector.apply_weighted_noise(noisy_sentences[idx])
    
    return ' '.join(noisy_sentences), target_score 

# ---------------------------------------------------------
# 5. Main Pipeline
# ---------------------------------------------------------

def main():
    papers_dir = "archive/papers"
    pdf_files = glob.glob(os.path.join(papers_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in '{papers_dir}/'.")
        return

    output_file = "train.jsonl"
    dataset = []

    print(f"Found {len(pdf_files)} papers.")

    # 0. Generate Distractor Pool (Once)
    print("Generating Distractor Pool...")
    distractor_pool = get_distractor_sentences(count=30)
    print(f"Distractor Pool Size: {len(distractor_pool)}")

    total_papers = len(pdf_files)
    for i, pdf_path in enumerate(pdf_files):
        filename = os.path.basename(pdf_path)
        print(f"\n=== Processing Paper [{i+1}/{total_papers}]: {filename} ===")
        
        context_text = extract_pdf_context(pdf_path)
        if not context_text: continue
        
        # Define 5 distinct questions
        questions = [
            "이 논문에서 주장하는 핵심 contribution을 논문이 발표된 시기를 기준으로 problem statement와 related work에 대한 분석이 녹아들도록 깔끔하게 작성하세요.",
            "이 논문에서 제시하는 핵심 개념 중 하나를 선택하여 설명하고, 그 개념이 논문의 전체적인 맥락에서 어떤 역할을 하는지 설명하세요.",
            "이 논문에서 제시하는 방법론의 주요 단계를 설명하고, 각 단계가 왜 필요한지 논리적으로 설명하세요.",
            "이 논문의 실험 결과 중 하나를 선택하여 설명하고, 그 결과가 논문의 주장을 어떻게 뒷받침하는지 분석하세요.",
            "이 논문의 한계점이나 향후 연구 방향을 논문 내용을 바탕으로 분석하고 설명하세요."
        ]
        
        # 1. Generate Gold Essays (One for each question)
        print(f"  Generating {len(questions)} Gold Essays...")
        gold_essays = []
        for q_idx, q in enumerate(questions):
            print(f"    Processing Question {q_idx+1}...")
            essay = generate_gold_essay(context_text, q)
            if essay:
                gold_essays.append(essay)
                time.sleep(2) # Prevent Rate Limiting

        # 2. Process each Gold Essay
        for essay in gold_essays:
            # Case 0: Gold Data
            dataset.append({
                "instruction": "당신은 한국어 학술 에세이 채점관입니다. 다음 에세이를 읽고 [내용 이해 및 요약, 설득력, 비판적 사고, 구조 및 조직, 표현, 형식] 6가지 항목에 대해 5점 만점으로 채점하고, JSON 형식으로 출력하세요.",
                "input": essay,
                "output": json.dumps({
                    "1. 내용 이해 및 요약": 5.0, "2. 설득력": 5.0, "3. 비판적 사고 및 학술적 맥락 파악": 5.0, 
                    "4. 구조 및 조직": 5.0, "5. 표현": 5.0, "6. 형식": 5.0, 
                    "총점": 5.0
                }, ensure_ascii=False)
            })

            # Iterate through Target Scores
            target_scores = [4.0, 3.0, 2.0, 1.0]
            augmentation_factor = 5
            
            for score in target_scores:
                print(f"    Target Score: {score} (Generating {augmentation_factor} variations...)")
                
                for _ in range(augmentation_factor):
                    # Case A: Language Noise (Corrupts 'Format' & 'Expression')
                    noisy_text_lang, _ = apply_case_noise(essay, "Language", score, distractor_pool)
                    total_score_lang = (5.0 + 5.0 + score) / 3.0
                    
                    dataset.append({
                        "input": noisy_text_lang,
                        "output": json.dumps({
                            "1. 내용 이해 및 요약": 5.0, "2. 설득력": 5.0, "3. 비판적 사고 및 학술적 맥락 파악": 5.0, 
                            "4. 구조 및 조직": 5.0, "5. 표현": score, "6. 형식": score, 
                            "총점": total_score_lang
                        }, ensure_ascii=False)
                    })

                    # Case B: Organization Noise (Corrupts 'Structure')
                    noisy_text_org, _ = apply_case_noise(essay, "Organization", score, distractor_pool)
                    total_score_org = (5.0 + score + 5.0) / 3.0
                    
                    dataset.append({
                        "instruction": "당신은 한국어 학술 에세이 채점관입니다. 다음 에세이를 읽고 [내용 이해 및 요약, 설득력, 비판적 사고, 구조 및 조직, 표현, 형식] 6가지 항목에 대해 5점 만점으로 채점하고, JSON 형식으로 출력하세요.",
                        "input": noisy_text_org,
                        "output": json.dumps({
                            "1. 내용 이해 및 요약": 5.0, "2. 설득력": 5.0, "3. 비판적 사고 및 학술적 맥락 파악": 5.0, 
                            "4. 구조 및 조직": score, "5. 표현": 5.0, "6. 형식": 5.0, 
                            "총점": total_score_org
                        }, ensure_ascii=False)
                    })

                    # Case C: Content Noise (Corrupts 'Content' related traits)
                    noisy_text_cont, _ = apply_case_noise(essay, "Content", score, distractor_pool)
                    total_score_cont = (score + 5.0 + 5.0) / 3.0
                    
                    dataset.append({
                        "instruction": "당신은 한국어 학술 에세이 채점관입니다. 다음 에세이를 읽고 [내용 이해 및 요약, 설득력, 비판적 사고, 구조 및 조직, 표현, 형식] 6가지 항목에 대해 5점 만점으로 채점하고, JSON 형식으로 출력하세요.",
                        "input": noisy_text_cont,
                        "output": json.dumps({
                            "1. 내용 이해 및 요약": score, "2. 설득력": score, "3. 비판적 사고 및 학술적 맥락 파악": score, 
                            "4. 구조 및 조직": 5.0, "5. 표현": 5.0, "6. 형식": 5.0, 
                            "총점": total_score_cont
                        }, ensure_ascii=False)
                    })
                
                # Prevent Rate Limiting
                time.sleep(0.1)

    # Save to JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"\nSuccessfully generated {len(dataset)} examples in '{output_file}'.")

if __name__ == "__main__":
    main()
