# Academic Writing Assessment Experiment (CASE Implementation)

이 프로젝트는 **CASE (Corruption-based Augmentation Strategy)** 방식을 활용하여 한국어 학술 글쓰기 데이터셋을 구축하고, LLM을 파인튜닝하여 채점 성능을 평가하는 실험입니다.

---

## 🚀 빠른 시작 (Quick Start for Server)

본 프로젝트는 한국어 학술 논문의 질적 평가를 자동화하기 위해 설계되었습니다. CASE(Corruption-based Augmentation Strategy) 기법을 적용하여 데이터셋을 확장하고, 최신 LLM(SOLAR 등)을 파인튜닝하여 논리적 완결성과 문법적 정확도를 종합적으로 평가하는 모델을 구축합니다.

서버 환경(Ubuntu 등)에서 아래 명령어를 순서대로 실행하면 **환경 설정부터 모델 학습, 평가**까지 한 번에 진행됩니다.

### 1. 저장소 클론 (Clone Repository)
```bash
git clone <GITHUB_REPO_URL>
cd <REPO_DIRECTORY>
```

### 2. 학습 스크립트 실행 (Run Training)
Hugging Face에 업로드된 데이터셋을 자동으로 다운로드하여 학습을 시작합니다.
*(실행 중 Hugging Face 토큰 입력이 필요할 수 있습니다.)*

```bash
# 실행 권한 부여
chmod +x run_server.sh

# 스크립트 실행 (데이터셋 ID 전달)
./run_server.sh SJunha/aes_dataset
```

---

## 📊 결과 확인 (Results)
학습이 완료되면 `aes_finetuned/` 디렉토리에 다음 결과물이 저장됩니다.

1.  **`qwk_comparison.png`**: 모델의 채점 일치도(QWK 점수) 비교 그래프 (Base vs Fine-tuned)
2.  **`metrics_comparison.png`**: 오차(MSE) 및 QWK 성능 종합 비교 그래프
3.  **`statistical_results.json`**: 상세 평가 수치 및 로그
4.  **`adapter_model.safetensors`**: 학습된 LoRA 모델 가중치 파일 (Adapter Weights)

---

## 🤖 모델 학습 (Model Training: LoRA Fine-tuning)
생성된 데이터셋을 활용하여 LLM이 학술 에세이를 정교하게 채점할 수 있도록 파인튜닝을 진행합니다.

### 1. 베이스 모델 (Base Model)
*   **Model**: `upstage/SOLAR-10.7B-Instruct-v1.0`
*   **특징**: 한국어와 영어 모두에 뛰어난 성능을 보이는 SOLAR 모델을 활용하여 학술적 문맥 이해도를 극대화합니다.

### 2. 학습 방법 (Training Strategy)
*   **LoRA (Low-Rank Adaptation)**: 모델 전체를 학습시키는 대신 일부 파라미터(Adapter)만 학습시켜 효율적인 파인튜닝을 수행합니다.
*   **4-bit Quantization (QLoRA)**: BitsAndBytes를 사용하여 모델을 4비트로 양자화하여 VRAM 사용량을 최적화하고 학습 속도를 높입니다.
*   **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (모든 Linear 레이어 대상)

### 3. 주요 하이퍼파라미터 (Hyperparameters)
*   **Epochs**: 3
*   **Batch Size**: 16 (Gradient Accumulation Steps: 4, Effective Batch Size: 64)
*   **Learning Rate**: 2e-4
*   **Max Length**: 1024 tokens
*   **Optimizer**: `paged_adamw_8bit`

### 4. 평가 지표 (Evaluation Metrics)
학습된 모델의 성능은 Base 모델과 비교하여 다음 지표로 평가됩니다.
*   **QWK (Quadratic Weighted Kappa)**: 실제 점수와 모델 채점 점수 간의 일치도를 측정하는 지표 (1에 가까울수록 일치)
*   **MSE (Mean Squared Error)**: 점수 예측의 오차 제곱 평균 (0에 가까울수록 정확)

---

## 🛠 데이터 생성 로직 (Data Generation: CASE)
이 프로젝트는 **CASE (Corruption-based Augmentation Strategy)** 기법을 통해 고품질의 학습 데이터를 생성합니다. 
생성된 데이터는 6가지 평가 기준에 맞춰 정교하게 점수가 부여됩니다.

1.  **Gold Data (Score 5)**: 전문가 수준의 루브릭(Rubric)을 Prompt로 제공하여 Gemini-2.5-Flash 모델이 만점 기준의 학술 에세이를 생성합니다.
2.  **Corruption Strategy & Mapping**: 생성된 만점 에세이에 의도적인 노이즈를 주입하여, 각 노이즈 유형에 해당하는 평가 항목의 점수를 감점시킵니다.

| 노이즈 유형 (Noise Type) | 적용 방식 (Technique) | 영향 받는 평가 기준 (Impacted Criteria) |
| :--- | :--- | :--- |
| **Content Noise** | **Out-of-domain Sentence Injection**<br>논문 주제와 전혀 무관한 문장(Distractor)을 무작위로 삽입하여 글의 맥락과 논리적 흐름을 해침 | • 1. 내용 이해 및 요약<br>• 2. 설득력<br>• 3. 비판적 사고 및 학술적 맥락 파악 |
| **Organization Noise** | **Sentence Swapping**<br>문장 간의 순서를 무작위로 뒤섞어 글의 구조적 일관성과 전개 논리를 파괴함 | • 4. 구조 및 조직 |
| **Language Noise** | **Grammar & Format Injection (with Konlpy)**<br>• **Grammar**: 조사(Josa), 어미(Ending), 용언 활용(Conjugation) 오류 주입<br>• **Format**: 맞춤법(Spelling), 띄어쓰기(Spacing) 오류 주입 | • 5. 표현<br>• 6. 형식 |

3.  **Automatic Labeling**: 주입된 노이즈의 비율에 따라 산술적으로 계산된 점수(1~4점)를 정답 라벨(Ground Truth)로 부여하여 대규모 데이터셋을 확보합니다.

