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

## 🛠 데이터 생성 로직 (Data Generation: CASE)
이 프로젝트는 **CASE (Corruption-based Augmentation Strategy)** 기법을 통해 고품질의 학습 데이터를 생성합니다.

1.  **Gold Data (Score 5)**: 전문가 수준의 루브릭(Rubric)을 Prompt로 제공하여 Gemini-2.5-Flash 모델이 만점 기준의 학술 에세이를 생성합니다.
2.  **Corruption Strategy**: 생성된 만점 에세이에 의도적인 노이즈를 주입하여 다양한 점수대(1~4점)의 데이터를 구축합니다.
    *   **Content Noise**: 논문과 무관한 문장(Out-of-domain)을 삽입하여 내용 이해 및 비판적 사고 점수를 하락시킵니다.
    *   **Organization Noise**: 문장의 순서를 무작위로 교체(Swap)하여 구조 및 조직 점수를 하락시킵니다.
    *   **Language Noise**: 형태소 분석기(Konlpy)를 활용해 조사를 임의로 변경하여 표현 및 형식 점수를 하락시킵니다.
3.  **Automatic Labeling**: 주입된 노이즈의 양에 따라 산술적으로 계산된 점수를 정답 라벨(Ground Truth)로 부여하여 대규모 데이터셋을 확보합니다.

