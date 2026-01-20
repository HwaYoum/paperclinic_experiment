# Academic Writing Assessment Dataset Generator & Fine-tuner (CASE Implementation)

이 프로젝트는 **CASE (Corruption-based Augmentation Strategy)** 방식을 활용하여 한국어 학술 글쓰기 평가(AES)를 위한 데이터를 생성하고, 이를 기반으로 **Qwen2.5-1.5B** 모델을 파인튜닝(Fine-tuning)하는 실험 프로젝트입니다.

완벽한 점수의 **Gold Standard 에세이**를 Gemini API로 생성한 후, 의도적인 **노이즈(Noise)**를 주입하여 점수대별(1점~4점) 데이터를 자동으로 구축합니다.

## 📂 프로젝트 구조

```bash
├── archive/
│   └── papers/          # [Input] 데이터 생성의 소스가 될 PDF 논문들을 위치시킵니다.
├── aes_finetuned/       # [Output] 학습된 LoRA 어댑터와 평가 결과가 저장됩니다.
├── generate_data.py     # [Step 1] 데이터 생성 스크립트 (PDF -> train.jsonl)
├── train_lora.py        # [Step 2] 모델 학습 및 평가 스크립트 (train.jsonl -> Fine-tuned Model)
├── requirements.txt     # 필요한 파이썬 라이브러리 목록
├── .env                 # API 키 설정 파일
└── README.md            # 프로젝트 설명서
```

## 🛠️ 설치 및 설정 (Setup)

### 1. 환경 설정
필요한 라이브러리를 설치합니다.
```bash
pip install -r requirements.txt
```
*주의: `KoNLPy` 실행을 위해 시스템에 Java(JDK)가 설치되어 있어야 합니다.*

### 2. API 키 설정
프로젝트 루트에 `.env` 파일을 생성하고 Google Gemini API 키를 입력합니다.
```bash
GEMINI_API_KEY=your_api_key_here
```

## 🚀 사용 방법 (Workflow)

### Step 1. 데이터셋 생성 (Data Generation)
`archive/papers/` 폴더에 분석하고 싶은 PDF 논문 파일들을 넣고 아래 명령어를 실행합니다.

```bash
python generate_data.py
```
**작동 원리:**
1.  **Gold Essay 생성**: Gemini가 PDF 내용을 바탕으로 6가지 평가 기준(내용, 설득력, 구조 등)에서 만점을 받는 에세이를 작성합니다.
2.  **CASE 노이즈 주입**: 만점 에세이에 3가지 유형의 노이즈를 주입하여 1~4점대 데이터를 생성합니다.
    *   **Content Noise**: 문장을 주제와 무관한 내용(Distractor)으로 교체
    *   **Organization Noise**: 문장 순서를 무작위로 뒤섞음
    *   **Language Noise**: 조사를 엉뚱하게 변경하여 문법 오류 유발
3.  **결과**: `train.jsonl` 파일이 생성됩니다.

### Step 2. 모델 학습 및 평가 (Train & Evaluate)
생성된 `train.jsonl` 데이터를 사용하여 모델을 학습시킵니다.

```bash
python train_lora.py
```
**주요 기능:**
*   **Model**: `Qwen/Qwen2.5-1.5B-Instruct` (4-bit Quantization)
*   **Method**: LoRA (Low-Rank Adaptation)
*   **Evaluation**: 학습 후 테스트 셋에 대해 **QWK (Quadratic Weighted Kappa)** 및 **MSE** 지표로 성능을 측정합니다.
*   **Output**: `aes_finetuned/` 폴더에 학습된 모델, 손실 그래프(`training_loss.png`), 성능 비교 그래프(`qwk_comparison.png`)가 저장됩니다.

## 📊 평가 기준 (Rubric)

데이터셋은 다음 6가지 항목에 대해 5점 척도로 평가됩니다.

1.  **내용 이해 및 요약**: 논문 핵심 내용 파악 정도
2.  **설득력**: 주장의 논리적 연결 및 근거
3.  **비판적 사고**: 연구의 의의 및 한계 분석
4.  **구조 및 조직**: 글의 흐름과 문단 구성
5.  **표현**: 어휘 선택 및 문장 구조의 정확성
6.  **형식**: 인용 양식 및 맞춤법 준수

## 📈 결과 확인

학습이 완료되면 `aes_finetuned/` 폴더에서 다음 파일들을 확인할 수 있습니다.
*   `statistical_results.json`: Base 모델 vs Fine-tuned 모델의 상세 점수 비교
*   `qwk_comparison.png`: 채점 일치도(QWK) 시각화 그래프
*   `metrics_comparison.png`: MSE 및 QWK 비교 그래프

---
*이 프로젝트는 생성형 AI를 활용한 합성 데이터(Synthetic Data) 구축 방법론을 연구하기 위해 제작되었습니다.*
