import os
import json
import uuid
import random
import glob
from datetime import datetime
from typing import Dict, List, Any
from pypdf import PdfReader
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------
# 1. Configuration & Rubric Definition
# ---------------------------------------------------------

# Retrieve API Key from environment variable
API_KEY = os.getenv("GEMINI_API_KEY")
client = None
if API_KEY:
    client = genai.Client(api_key=API_KEY)

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
# 2. Dynamic Prompt Construction
# ---------------------------------------------------------

def construct_prompt_data(
    target_scores: Dict[str, int], 
    context_text: str, 
    question: str
) -> Dict[str, str]:
    
    system_message = (
        "You are an AI assistant tasked with generating synthetic student essays "
        "that simulate specific levels of writing proficiency based on a provided rubric."
    )
    
    user_instruction_header = (
        "Task: Write an academic essay based on the provided [Context Text] in response to the [Question].\n\n"
        "*** CRITICAL INSTRUCTION ***\n"
        "You must deliberately adjust the quality of your writing to match the specific "
        "rubric levels requested below. Do not just write a perfect essay. "
        "Mimic the flaws or strengths described for each trait.\n"
    )

    criteria_str = "\n[Target Assessment Criteria]\n"
    rubric_key_map = {
        "content": "1. 내용 이해 및 요약",
        "persuasiveness": "2. 설득력",
        "critical_thinking": "3. 비판적 사고 및 학술적 맥락 파악",
        "organization": "4. 구조 및 조직",
        "expression": "5. 표현",
        "formatting": "6. 형식"
    }

    for trait, score in target_scores.items():
        full_key = rubric_key_map.get(trait, trait)
        description = RUBRIC.get(full_key, {}).get(score, "Description not found.")
        criteria_str += f"- {full_key} (Level {score}): {description}\n"

    input_data_str = (
        f"\n[Context Text (Excerpt from Paper)]\n{context_text}\n\n"
        f"[Question]\n{question}\n\n"
        "Please generate the essay now."
    )

    return {
        "system_message": system_message,
        "user_message": user_instruction_header + criteria_str + input_data_str
    }

# ---------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------

def extract_pdf_context(pdf_path: str, max_pages: int = 2) -> str:
    try:
        reader = PdfReader(pdf_path)
        text = ""
        num_pages = min(max_pages, len(reader.pages))
        for i in range(num_pages):
            page_text = reader.pages[i].extract_text()
            if page_text:
                text += page_text + "\n\n"
        return text.strip()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def generate_random_scores() -> Dict[str, int]:
    traits = ["content", "persuasiveness", "critical_thinking", "organization", "expression", "formatting"]
    return {trait: random.randint(1, 5) for trait in traits}

# ---------------------------------------------------------
# 4. Generation Logic
# ---------------------------------------------------------

def generate_entry(
    paper_meta: Dict[str, Any],
    input_context: Dict[str, str],
    target_scores: Dict[str, int],
    model_id: str = "gemini-2.5-flash"
) -> Dict[str, Any]:
    
    prompts = construct_prompt_data(
        target_scores, 
        input_context["context_text"], 
        input_context["question"]
    )
    
    generated_text = "ERROR: Generation Failed"

    if client:
        try:
            # New SDK usage: client.models.generate_content
            response = client.models.generate_content(
                model=model_id,
                contents=prompts["user_message"],
                config=types.GenerateContentConfig(
                    system_instruction=prompts["system_message"],
                    temperature=0.7
                )
            )
            generated_text = response.text
        except Exception as e:
            generated_text = f"Error during generation: {str(e)}"
    else:
        generated_text = f"[MOCK] Essay for {paper_meta['paper_title']} with scores {target_scores}"

    return {
        "id": str(uuid.uuid4()),
        "meta": paper_meta,
        "input": input_context,
        "generation_config": {
            "model_name": model_id,
            "target_labels": target_scores
        },
        "prompt_constructed": prompts,
        "output": {
            "generated_essay": generated_text
        }
    }

# ---------------------------------------------------------
# 5. Main Execution
# ---------------------------------------------------------

def main():
    papers_dir = "papers"
    pdf_files = glob.glob(os.path.join(papers_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in '{papers_dir}/'.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_output = {
        "dataset_info": { 
            "version": "1.2", 
            "description": "Synthetic AES Dataset via google-genai",
            "generated_at": timestamp
        },
        "data": []
    }

    print(f"Found {len(pdf_files)} papers. Generating 20 samples each...")

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        paper_title = os.path.splitext(filename)[0].replace("_", " ").title()
        print(f"\nProcessing: {paper_title}")
        
        context_text = extract_pdf_context(pdf_path, max_pages=2)
        if not context_text: continue
            
        question = "Summarize the paper's core contributions and arguments based on the provided text."

        for i in range(20):
            target_scores = generate_random_scores()
            print(f"  - Sample {i+1}/20...")
            
            entry = generate_entry(
                paper_meta={"paper_title": paper_title, "filename": filename},
                input_context={"context_text": context_text, "question": question},
                target_scores=target_scores
            )
            dataset_output["data"].append(entry)

    output_filename = f"synthetic_aes_dataset_{timestamp}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(dataset_output, f, indent=2, ensure_ascii=False)
    
    print(f"\nDone. {len(dataset_output['data'])} total entries saved to '{output_filename}'.")

if __name__ == "__main__":
    main()
