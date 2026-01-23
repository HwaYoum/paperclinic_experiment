# 📝 한국어 학술 에세이 자동 채점 (AES) 프로젝트 가이드

본 문서는 프로젝트의 전체 워크플로우(데이터 생성, 증강, 모델 학습)를 설명합니다.
---

## 📚 1. 프로젝트 개요 (Overview)
이 프로젝트는 **한국어 학술 에세이**를 **내용(Content), 구성(Organization), 언어(Language)** 세 가지 기준으로 자동 채점하는 AI 모델을 개발하는 것입니다.

핵심 전략은 **"검증된 노이즈 주입(Validated Noise Injection)"**입니다.
1.  **High Quality Data**: AI Hub 원천 데이터 또는 논문 기반의 완벽한(5점) 에세이를 확보합니다.
2.  **Noise Injection**: 각 평가 기준에 맞춰 의도적으로 오류(노이즈)를 주입하여 1~4점대 데이터를 합성합니다.
3.  **Validation**: 생성된 데이터가 목표 점수에 부합하는지 LLM(Gemini)으로 검증합니다.
4.  **Fine-tuning**: 확보된 데이터를 모두 결합하여 SLM(Solar-10.7B 등)을 미세 조정합니다.

---

## 📂 2. 프로젝트 구조 (Structure)

```bash
.
├── config/
│   └── config.yaml          # [설정] 모델, LoRA 파라미터, 학습 하이퍼파라미터 정의
├── data/
│   └── papers/              # [소스] 합성 데이터 생성용 PDF 논문 저장소
├── scripts/
│   ├── setup_env.sh         # [실행] 개발 환경 자동 설정 스크립트
│   └── run_training.sh      # [실행] 모델 학습 시작 스크립트
├── src/
│   ├── data_gen/            # [데이터] 데이터 생성 및 변환 모듈
│   │   ├── generate_dataset.py     # (New) 논문 PDF 기반 데이터 생성기
│   │   ├── generate_from_gold.py   # (Aug) 기존 골드 에세이 기반 증강기
│   │   └── convert_dataset.py      # (Raw) AI Hub 데이터 포맷 변환기
│   ├── train/               # [학습] 모델 학습 모듈
│   │   └── train_lora.py           # LoRA Fine-tuning 실행 코드
│   └── utils/               # [도구] 유틸리티
│       ├── augmentation_utils.py   # 한국어 노이즈 주입 클래스 (KoreanNoiseInjector)
│       └── upload_to_hf.py         # Hugging Face 업로드 스크립트
└── aes_finetuned/           # [출력] 학습 완료된 모델 저장 경로
```

---

## 🛠️ 3. 데이터 파이프라인 (Data Pipeline)

이 프로젝트는 세 가지 소스의 데이터를 결합하여 학습합니다.

### A. AI Hub 원천 데이터 (Original)
기존에 구축된 한국어 에세이 평가 데이터셋입니다.
- **처리 파일**: `src/data_gen/convert_dataset.py`
- **역할**: JSON 파일을 모델 학습용 JSONL 포맷으로 변환합니다.

### B. 논문 기반 합성 데이터 (Generated)
새로운 학술 논문(PDF)을 읽고, 해당 논문에 대한 질문과 답변을 생성합니다.
- **실행 파일**: `src/data_gen/generate_dataset.py`
- **프로세스**:
    1.  `data/papers/*.pdf`에서 텍스트 추출.
    2.  Gemini가 "5점 만점 답변" 생성 (Gold Essay).
    3.  노이즈를 주입하여 1~4점 답변 생성.
    4.  Hugging Face `SJunha/aes-dataset` (config: `paperclinic_generated_dataset`) 업로드.

### C. 골드 데이터 증강 (Augmented)
기존 데이터 중 5점(만점)인 에세이만을 활용하여 데이터를 불립니다.
- **실행 파일**: `src/data_gen/generate_from_gold.py`
- **프로세스**: 기존 5점 에세이에 노이즈를 주입하여 점수별 데이터를 추가 확보합니다.
- **저장소**: Hugging Face `SJunha/aes-dataset` (config: `gold_augmented`).

---

## 🧩 4. 노이즈 주입 메커니즘 (Noise Injection)

데이터 생성 시, 점수를 깎기 위해 다음과 같은 구체적인 노이즈를 주입합니다.

| 평가 항목 | 주입 방식 (Method) | 구현 상세 |
| :--- | :--- | :--- |
| **1. 내용 (Content)** | **문장 교체**<br>(Replacement) | 문맥과 무관한 문장(Distractor)으로 원본 문장을 1:1 교체합니다.<br>점수가 낮을수록 교체 비율이 높아집니다.<br>|
| **2. 구성 (Organization)** | **순서 섞기**<br>(Reordering) | 문장의 순서를 무작위로 섞어 논리적 흐름을 파괴합니다.<br>점수가 낮을수록 섞이는 횟수가 증가합니다. |
| **3. 언어 (Language)** | **언어적 오류 주입**<br>(Linguistic Error) | `src/utils/augmentation_utils.py`의 `KoreanNoiseInjector` 사용:<br>- **조사/어미 오류**: '은/는' 오용, 격식체 파괴<br>- **맞춤법/띄어쓰기**: 자모 분리, 띄어쓰기 무시<br>- **문장 호응**: 주술 불일치 유도 |

---

## 🚀 5. 실행 가이드 (How to Run)

서버(Ubuntu) 환경에서 처음부터 학습까지 진행하는 순서입니다.

### 1단계: 환경 설정
필요한 라이브러리를 설치하고 가상환경을 구성합니다.
```bash
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

### 2단계: 데이터 준비 (선택 사항)
이미 Hugging Face에 데이터가 업로드되어 있다면 이 단계는 건너뛰어도 됩니다. 직접 데이터를 새로 생성하려면 아래 명령어를 실행하세요.
```bash
# 논문 기반 데이터 생성 및 업로드
python src/data_gen/generate_dataset.py
```

### 3단계: 모델 학습 및 실험 (Training & Experiments)
4가지 실험 모드를 지원하여 데이터 구성에 따른 성능 변화를 비교할 수 있습니다.

| 모드 (Mode) | 설명 (Description) | 학습 데이터 구성 |
| :--- | :--- | :--- |
| **zero_shot** | 학습 없이 기본 모델(Base Model) 성능 평가 | (학습 없음) |
| **original** | 원천 데이터만 사용하여 미세 조정 | AI Hub Original |
| **augmented** | 원천 데이터 + 골드 증강 데이터 사용 | AI Hub Original + Gold Augmented |
| **full** (기본값) | 모든 데이터(논문 생성 데이터 포함) 사용 | AI Hub Original + Gold Augmented + Paper Clinic Generated |

스크립트 실행 시 모드를 인자로 전달하세요.

```bash
# 1. Zero-shot 평가 (학습 X)
./scripts/run_training.sh zero_shot

# 2. Original 데이터만 학습
./scripts/run_training.sh original

# 3. Augmented 데이터까지 학습
./scripts/run_training.sh augmented

# 4. 모든 데이터 학습 (Full) - 기본값
./scripts/run_training.sh full
```

각 실험의 결과(모델 및 평가 리포트)는 `aes_finetuned_{mode}/` 디렉토리에 개별적으로 저장되므로, 실험 결과를 쉽게 비교할 수 있습니다.

---

## 📊 6. 평가 (Evaluation)

학습 중 및 학습 후 **QWK (Quadratic Weighted Kappa)** 점수로 모델 성능을 모니터링합니다. QWK는 사람이 채점한 점수와 모델 예측 점수의 일치도를 보여주는 지표입니다. (1.0에 가까울수록 일치)

---

## ⚙️ 요구 사항 (Requirements)
- **OS**: Linux (Ubuntu 권장) 또는 macOS
- **Python**: 3.10 이상
- **GPU**: VRAM 24GB 이상 권장 (LoRA 학습 시)
- **API Keys**: `.env` 파일에 `GEMINI_API_KEY`, `HF_TOKEN` 설정 필요.