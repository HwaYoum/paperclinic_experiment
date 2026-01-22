# 📝 한국어 학술 에세이 자동 평가 (AES) 모델 파인튜닝 프로젝트

본 프로젝트는 **SOLAR-10.7B-Instruct** 모델을 기반으로, 한국어 학술 에세이의 **내용, 구성, 언어** 및 **총점**을 자동으로 평가하는 모델을 구축하기 위한 파인튜닝(LoRA) 파이프라인입니다.

AI Hub의 '주제별 글쓰기 평가 데이터'를 활용하며, 데이터 전처리, 노이즈 주입을 통한 증강(Augmentation), 그리고 Hugging Face Hub를 연동한 학습 및 평가 프로세스를 포함합니다.

---

## 📂 프로젝트 구조 (Project Structure)

```bash
.
├── config/
│   └── config.yaml          # 모델, LoRA, 학습 하이퍼파라미터 설정
├── data/                    # (로컬) 데이터 저장소 (생성된 jsonl, csv 등)
├── scripts/                 # 실행용 쉘 스크립트
│   ├── run_training.sh      # [핵심] 학습 실행 스크립트
│   └── setup_env.sh         # 가상환경 및 의존성 설치 스크립트
├── src/                     # 소스 코드
│   ├── data_gen/            # 데이터 전처리 및 증강
│   │   ├── convert_dataset.py     # 원본 JSON -> 학습용 JSONL 변환
│   │   └── generate_from_gold.py  # 만점 에세이 기반 노이즈 주입/증강
│   ├── train/               # 학습 메인 코드
│   │   └── train_lora.py          # LoRA 파인튜닝 및 QWK 평가
│   └── utils/               # 유틸리티
│       ├── augmentation_utils.py  # 한국어 노이즈 주입기 (Josa, 맞춤법 등)
│       ├── upload_to_hf.py        # Hugging Face 데이터 업로드
│       └── inspect_utils.py       # 데이터 검증 툴
└── aes_finetuned/           # 학습 완료된 모델 및 결과 저장소 (자동 생성)
```

---

## 🚀 Quick Start (Ubuntu Server)

서버 환경에서 바로 학습을 시작하는 방법입니다. 데이터는 Hugging Face Hub에서 자동으로 다운로드되므로 별도의 데이터 복사가 필요 없습니다.

### 1. 환경 설정 (최초 1회)
```bash
# 실행 권한 부여
chmod +x scripts/*.sh

# 가상환경 생성 및 필수 라이브러리 설치
./scripts/setup_env.sh

# 가상환경 활성화
source venv/bin/activate
```

### 2. 학습 실행
데이터셋 저장소 ID(예: `SJunha/aes-dataset`)를 인자로 주어 실행합니다.
```bash
# 사용법: ./scripts/run_training.sh <HF_DATASET_ID>
./scripts/run_training.sh SJunha/aes-dataset
```

학습이 완료되면 `aes_finetuned/` 폴더에 모델 가중치(Adapter)와 평가 결과(`metrics_comparison.png`, `statistical_results.json`)가 저장됩니다.

---

## 🛠️ 데이터 파이프라인 (Data Pipeline)

이 프로젝트는 로컬 데이터를 가공하여 Hugging Face에 업로드하고, 서버에서 이를 받아 학습하는 구조를 권장합니다.

### 1. 데이터 변환 (Raw JSON -> JSONL)
AI Hub 원본 데이터를 학습 가능한 포맷으로 변환하고, Train/Validation 셋으로 분리합니다.
```bash
python src/data_gen/convert_dataset.py
# 결과물: data/converted_train.jsonl, data/converted_val.jsonl
```

### 2. 데이터 증강 (Data Augmentation)
만점(5.0) 에세이에 인위적인 노이즈(조사, 어미, 문장 순서 오류 등)를 주입하여, 낮은 점수대의 데이터를 생성하고 모델의 채점 기준을 견고하게 만듭니다.
```bash
# 기본: 모든 만점 에세이에 대해 각 항목별 1회씩 증강
python src/data_gen/generate_from_gold.py

# 옵션: 증강 배수(Factor) 조절 (예: 에세이당 3개씩 변이 생성)
python src/data_gen/generate_from_gold.py --aug_factor 3
```

### 3. Hugging Face 업로드
생성된 데이터(`train.jsonl`, `validation.jsonl`, `refined_sentences.csv`)를 저장소에 업로드합니다. 저장소가 없으면 자동 생성합니다.
```bash
# 공개 저장소로 생성 및 업로드 예시
python src/utils/upload_to_hf.py SJunha/aes-dataset
```

---

## 📊 평가 지표 (Evaluation)

학습 종료 후, **QWK (Quadratic Weighted Kappa)** 지표를 사용하여 모델 성능을 평가합니다.
- **QWK**: 두 평가자(사람 vs 모델) 간의 일치도를 측정하는 지표로, 점수 차이가 클수록 더 큰 페널티를 부여합니다. (AES 분야 표준)
- **비교**: Base 모델(SOLAR-10.7B)과 Fine-tuned 모델의 점수를 비교하여 `aes_finetuned/metrics_comparison.png`에 시각화합니다.

---

## ⚙️ 설정 변경 (Configuration)

학습 하이퍼파라미터(Epoch, Batch Size, Learning Rate 등)는 `config/config.yaml` 파일에서 수정할 수 있습니다.

```yaml
training:
  num_train_epochs: 1
  per_device_train_batch_size: 4
  learning_rate: 2e-4
  # ...
```

---

## 📝 관리자 참고 사항 (Notes for Maintainers)

*   **데이터셋 스키마**: `train_lora.py`는 `prompt`(주제), `text`(에세이), `content/organization/expression/holistic`(점수) 필드를 자동으로 인식하여 학습 프롬프트를 구성합니다.
*   **서버 의존성**: `train_lora.py`는 `data/` 폴더가 비어있어도 `args.dataset_name`이 제공되면 Hugging Face에서 데이터를 우선적으로 로드하므로 서버 배포에 용이합니다.
