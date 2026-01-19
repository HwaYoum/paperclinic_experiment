# Academic Writing Assessment Dataset Generator (CASE Implementation)

이 프로젝트는 **Gold Data(만점 에세이)**를 생성한 후, **CASE(Corruption-based Augmentation Strategy)** 방식을 응용하여 의도적인 **노이즈(Noise)**를 주입함으로써 다양한 품질의 학술적 글쓰기(AES) 데이터를 구축하는 실험 도구입니다.

Gemini API를 활용해 완벽한 에세이를 생성하고, 코드를 통해 문장 단위의 기계적 결함을 정교하게 주입하여 모델이 점수별 차이를 학습할 수 있도록 설계되었습니다.

## 1. 주요 접근 방식 (Methodology)

이 도구는 **만점 답안(Gold Standard)**을 먼저 생성하고, 목표 점수(Target Score)에 비례하여 **망가뜨릴 문장의 비율**을 계산하는 방식을 사용합니다.

### 1) CASE 기반 노이즈 비율 계산
점수가 낮을수록 더 많은 문장에 노이즈를 주입합니다.
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
    *   **결과:** Gold Essay 1개당 13개의 데이터(원본 1 + 변형 12)가 생성됩니다.

## 3. 파일 구조

*   `generate_data.py`: 메인 데이터 생성 스크립트.
*   `archive/`: 이전 실험 코드 및 PDF 원본 저장소.
    *   `archive/papers/`: 분석 대상 논문 PDF 파일 위치.
*   `train.jsonl`: 생성된 학습 데이터셋 결과물.
*   `.env`: API Key 설정 파일.

## 4. 설치 및 실행 방법

### 요구 사항 (Requirements)
Python 3.10 이상 권장. `konlpy` 구동을 위해 JDK 설치가 필요할 수 있습니다.

```bash
pip install -r requirements.txt
```

### 환경 변수 설정 (.env)
프로젝트 루트에 `.env` 파일을 생성하고 Gemini API 키를 입력합니다.
```env
GEMINI_API_KEY=your_actual_api_key_here
```

### 데이터 생성 실행
```bash
python generate_data.py
```
*   `archive/papers/` 폴더 내의 모든 PDF를 읽어옵니다.
*   결과는 `train.jsonl` 파일에 저장됩니다.

## 5. 데이터셋 구조 (Output Schema)

결과물은 LLM 파인튜닝(Instruction Tuning)에 적합한 `JSONL` 형식입니다. 총점은 노이즈 유형에 따라 다르게 계산됩니다.

```json
{
  "instruction": "당신은 한국어 학술 에세이 채점관입니다...",
  "input": "(노이즈가 주입된 에세이 텍스트)",
  "output": "{'1. 내용 이해 및 요약': 3.0, '2. 설득력': 3.0, ..., '6. 형식': 5.0, '총점': 3.5}"
}
```

*   **Content Noise 적용 시:** 내용 관련 3개 항목 점수 하락 -> 총점 대폭 하락
*   **Organization/Language Noise 적용 시:** 해당 1개 항목 점수만 하락 -> 총점 소폭 하락

## 6. 주의 사항

*   **KoNLPy 의존성:** 실행 환경에 Java(JDK)가 설치되어 있어야 `konlpy`가 정상 작동합니다.
*   **데이터 다양성:** 하나의 논문에 대해 점수대별로 균일한 데이터를 생성하므로, 데이터 불균형 문제를 완화할 수 있습니다.