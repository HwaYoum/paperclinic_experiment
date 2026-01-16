# Academic Writing Assessment Dataset Generator (AES Experiment)

이 프로젝트는 Google Gemini API (`google-genai`)를 활용하여 학술적 글쓰기 평가(Automated Essay Scoring, AES) 연구를 위한 합성 데이터셋을 생성하는 실험 도구입니다.

논문 PDF의 내용을 바탕으로 다양한 수준(Rubric Level 1~5)의 에세이를 생성하여, AES 모델 학습 및 검증에 사용할 수 있는 데이터를 구축하는 것을 목표로 합니다.

## 1. 주요 기능

*   **PDF 텍스트 추출:** `pypdf`를 사용하여 논문 PDF의 초반 2페이지 내용을 자동으로 추출합니다.
*   **합성 데이터 생성:** Gemini API를 이용해 주어진 논문을 요약하는 에세이를 생성합니다. (기본 모델: `gemini-1.5-pro`)
*   **동적 프롬프트 (Dynamic Prompting):** 6가지 평가 항목에 대해 무작위 점수(1~5점)를 부여하고, 해당 루브릭 정의를 프롬프트에 주입하여 다양한 품질의 에세이를 시뮬레이션합니다.
*   **결과 관리:** 모든 결과물은 `outputs/` 폴더 내에 타임스탬프와 함께 저장되어 버전 관리가 용이합니다.
*   **포맷 변환:** 생성된 JSON 데이터를 분석하기 쉬운 CSV 형식으로 변환하는 도구를 제공합니다.

## 2. 파일 구조

*   `generate_dataset.py`: 메인 데이터 생성 스크립트.
*   `json_to_csv.py`: 최신 JSON 결과물을 CSV로 변환하는 스크립트.
*   `debug_gemini.py`: API 연결 및 환경 변수 설정을 점검하는 도구.
*   `papers/`: 분석할 논문 PDF 파일을 넣는 디렉토리.
*   `outputs/`: 생성된 JSON 및 CSV 파일이 저장되는 디렉토리 (Git 추적 제외).
*   `.env`: API Key 등 민감 정보를 보관하는 파일 (Git 추적 제외).

## 3. 설치 및 실행 방법

### 요구 사항 (Requirements)
Python 3.10 이상 권장.

```bash
pip install -r requirements.txt
```

### 환경 변수 설정 (.env)
프로젝트 루트에 `.env` 파일을 생성하고 다음과 같이 API 키를 입력합니다.
```env
GEMINI_API_KEY=your_actual_api_key_here
```

### 데이터 생성 실행
```bash
python generate_dataset.py
```
*   `papers/` 폴더의 각 PDF당 20개의 샘플을 생성합니다.
*   결과물 예시: `outputs/synthetic_aes_dataset_20260116_153000.json`

### CSV 변환
```bash
python json_to_csv.py
```
*   `outputs/` 폴더에서 가장 최근에 생성된 JSON 파일을 찾아 동일한 이름의 CSV로 변환합니다.

## 4. 데이터셋 구조 (Output Schema)

```json
{
  "dataset_info": {
    "version": "1.2",
    "generated_at": "20260116_153000",
    "description": "Synthetic AES Dataset via google-genai"
  },
  "data": [
    {
      "id": "uuid-string",
      "meta": { "paper_title": "...", "filename": "..." },
      "input": { "context_text": "...", "question": "..." },
      "generation_config": {
        "model_name": "gemini-1.5-pro",
        "target_labels": { "content": 3, "persuasiveness": 2, ... }
      },
      "output": { "generated_essay": "..." }
    }
  ]
}
```

## 5. 주의 사항 (Important)

*   **API 비용:** 대량의 데이터를 생성할 경우 Gemini API 호출 비용이 발생할 수 있습니다.
*   **저작권:** 생성된 데이터셋에는 논문 원문의 일부가 포함되어 있습니다. 저작권이 있는 논문을 사용한 경우, 데이터셋을 Public GitHub Repository 등에 **공개적으로 업로드하지 마십시오.** (기본적으로 `.gitignore`를 통해 차단되어 있습니다.)
*   **보안:** `.env` 파일이 외부에 노출되지 않도록 주의하십시오.