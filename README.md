# Academic Writing Assessment Dataset Generator (CASE Implementation)

이 프로젝트는 **Gold Data(만점 에세이)** 를 생성한 후, **CASE(Corruption-based Augmentation Strategy)** 방식을 응용하여 의도적인 **노이즈(Noise)** 를 주입함으로써 다양한 품질의 학술적 글쓰기(AES) 데이터를 구축하는 실험 도구입니다.

Gemini API를 활용해 완벽한 에세이를 생성하고, 코드를 통해 문장 단위의 기계적 결함을 정교하게 주입하여 모델이 점수별 차이를 학습할 수 있도록 설계되었습니다.

## 1. 주요 접근 방식 (Methodology)

이 도구는 **만점 답안(Gold Standard)** 을 먼저 생성하고, 목표 점수(Target Score)에 비례하여 **망가뜨릴 문장의 비율**을 계산하는 방식을 사용합니다.

### 1) CASE 기반 노이즈 비율 계산
점수가 낮을수록 더 많은 문장에 노이즈를 주입합니다.<br>
$$ n_{sc} = \text{round} \left( n_{se} \times \frac{5.0 - \text{score}}{5.0} \right) $$
*   $n_{se}$: 전체 문장 수
*   $n_{sc}$: 노이즈를 주입할 문장 수
*   예: 4점 목표 시 20%, 1점 목표 시 80%의 문장을 변형.

### 2) 노이즈 유형 (Noise Types)
| 유형 (Type) | 타겟 평가 항목 | 구현 방식 (Implementation) |
| :--- | :--- | :--- |
| **Content** | 1. 내용, 2. 설득력, 3. 비판적 사고 | 원본 문장을 주제와 무관한 **방해 문장(Distractor)**으로 **교체**합니다. (예: 논문 요약 중 갑자기 '심해 생물의 생태' 문장 등장) |
| **Organization** | 4. 구조 및 조직 | 문장들의 **순서(Position)**를 무작위로 **맞바꿉니다**. |
| **Language** | 6. 형식 (표현/문법) | `KoNLPy`를 활용하여 문장 내 **조사(은/는/이/가/을/를)**를 엉뚱한 것으로 **변경**합니다. |

## 2. 데이터 생성 파이프라인

1.  **Gold Data 생성**: PDF 논문(초반 3페이지)을 바탕으로 Gemini가 **6가지 평가 기준(5점 만점)**을 완벽히 충족하는 에세이를 작성합니다.
2.  **Distractor Pool 생성**: 논문 주제와 전혀 상관없는 무작위 문장 풀(Pool)을 생성합니다.
3.  **Noise Injection (변형 생성)**:
    *   각 Gold Essay에 대해 **4가지 목표 점수(4.0, 3.0, 2.0, 1.0)**를 설정합니다.
    *   각 목표 점수마다 3가지 노이즈 유형(Content, Organization, Language)을 각각 적용합니다.
    *   **Augmentation**: 데이터 다양성 확보를 위해 `augmentation_factor`(기본값 20)를 설정하여, 동일한 점수/유형 조합에 대해 서로 다른 무작위 노이즈가 주입된 데이터를 여러 개 생성합니다.
    *   **결과:** Gold Essay 1개당 약 241개의 데이터(원본 1 + [4개 점수 × 3개 유형 × 20개 변형])가 생성됩니다.

... (중략) ...

## 5. 데이터셋 구조 (Output Schema)

결과물은 LLM 파인튜닝(Instruction Tuning)에 적합한 `JSONL` 형식입니다. 총점은 노이즈 유형에 따라 다르게 계산됩니다.

```json
{
  "instruction": "당신은 한국어 학술 에세이 채점관입니다...",
  "input": "(노이즈가 주입된 에세이 텍스트)",
  "output": "{'1. 내용 이해 및 요약': 3.0, '2. 설득력': 3.0, ..., '6. 형식': 5.0, '총점': 3.5}"
}
```

*   **성능 평가 지표**: 이 프로젝트는 모델의 채점 정확도를 측정하기 위해 **QWK (Quadratic Weighted Kappa)**를 주 평가지표로 사용합니다. 이는 모델의 예측 점수와 실제 정답 점수 간의 일치도를 통계적으로 분석하며, 실제 AES(자동 에세이 채점) 연구에서 널리 쓰이는 표준 지표입니다.

... (중략) ...

**2. 모델 학습 (Train LoRA)**
데이터 생성이 완료되어 `train.jsonl` 파일이 준비되면 실행합니다. 가벼우면서도 강력한 `Qwen2.5-1.5B-Instruct` 모델을 사용하여 T4 GPU에서도 원활하게 학습 및 QWK 평가가 가능합니다.
```powershell
python train_lora.py
```