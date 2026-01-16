# Academic Writing Assessment Dataset Generator (AES Experiment)

이 프로젝트는 Google Gemini API (`google-genai`)를 활용하여 학술적 글쓰기 평가(Automated Essay Scoring, AES) 연구를 위한 합성 데이터셋을 생성하는 실험 도구입니다.

논문 PDF의 내용을 바탕으로 다양한 수준(Rubric Level 1~5)의 에세이를 생성하여, AES 모델 학습 및 검증에 사용할 수 있는 데이터를 구축하는 것을 목표로 합니다.

## 1. 주요 기능

*   **PDF 텍스트 추출:** `pypdf`를 사용하여 논문 PDF의 초반부 내용을 자동으로 추출합니다.
*   **합성 데이터 생성:** Gemini 1.5 Pro/Flash 모델을 이용해 주어진 논문을 요약하는 에세이를 생성합니다.
*   **동적 프롬프트 (Dynamic Prompting):** 6가지 평가 항목(내용, 설득력, 비판적 사고, 구조, 표현, 형식)에 대해 무작위 점수(1~5점)를 부여하고, 각 점수에 해당하는 루브릭(Rubric) 정의를 프롬프트에 주입하여 의도적인 품질 저하/향상을 시뮬레이션합니다.
*   **JSON 데이터셋 구축:** 생성된 에세이와 목표 점수(Target Scores), 원본 텍스트 등을 구조화된 JSON 형식으로 저장합니다.

## 2. 데이터셋 구조 (Output Schema)

생성된 `.json` 파일은 다음과 같은 구조를 가집니다:

```json
{
  "dataset_info": {
    "version": "1.2",
    "description": "Synthetic AES Dataset via google-genai",
    "generated_at": "20260116_143005"
  },
  "data": [
    {
      "id": "uuid-string",
      "meta": {
        "paper_title": "Paper Title",
        "filename": "paper.pdf"
      },
      "input": {
        "context_text": "Extracted text from PDF...",
        "question": "Summarize the paper's core contributions..."
      },
      "generation_config": {
        "model_name": "gemini-1.5-pro",
        "target_labels": {
          "content": 4,
          "persuasiveness": 3,
          "critical_thinking": 2,
          "organization": 5,
          "expression": 4,
          "formatting": 3
        }
      },
      "output": {
        "generated_essay": "Generated essay content..."
      }
    }
  ]
}
```

## 3. 설치 및 실행 방법

### 요구 사항 (Requirements)
Python 3.10 이상이 권장됩니다.

```bash
pip install -r requirements.txt
```
*`requirements.txt`에는 `google-genai`, `pypdf`, `python-dotenv`가 포함되어 있습니다.*

### 환경 변수 설정 (.env)
프로젝트 루트에 `.env` 파일을 생성하고 Gemini API 키를 입력하세요.
```env
GEMINI_API_KEY=your_api_key_here
```

### 실행
`papers/` 폴더에 분석할 PDF 파일들을 넣은 후 스크립트를 실행합니다.
```bash
python generate_dataset.py
```
스크립트는 폴더 내의 모든 PDF를 순회하며, 각 논문당 20개의 에세이 변형(총 논문 수 × 20)을 생성하여 타임스탬프가 찍힌 JSON 파일로 저장합니다.

## 4. 평가 루브릭 (6-Trait Analytic Rubric)
이 실험은 다음 6가지 항목에 대한 한국어 루브릭을 기반으로 합니다:
1.  **내용 이해 및 요약 (Content)**
2.  **설득력 (Persuasiveness)**
3.  **비판적 사고 및 학술적 맥락 (Critical Thinking)**
4.  **구조 및 조직 (Organization)**
5.  **표현 (Expression)**
6.  **형식 (Formatting)**
