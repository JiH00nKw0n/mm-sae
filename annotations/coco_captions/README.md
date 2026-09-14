# 캡션의 객체 표현 추출

**캡션별 객체 표현만 추출하고, 코드가 원문에서 일치하는 모든 위치를 찾는다.** 출현 순번은 모델에 요구하지 않는다. 같은 COCO 캡션 16개로 GPT-5.4 mini, GPT-5.4와 GPT-5.1을 시험했고 모두 원문 표현 검사를 통과했다. 의미 판단 오류는 세 모델 모두 남았다. 전체 캡션 추출과 SAE 전체 실험은 시작하지 않았다.

입력은 공식 COCO2017 학습 캡션 591,753개와 검증 캡션 25,014개를 대상으로 준비한다. 현재 API 호출은 고정된 캡션 16개 시험까지만 실행했다. 각 캡션에는 대응 이미지의 COCO-Stuff 원본 픽셀 주석에 실제로 등장하는 범주 번호·이름을 제공한다. 중앙 자르기 전 원본 주석을 사용하며 이미지는 API에 보내지 않는다. 다른 캡션, SAE 활성값과 상관계수도 제공하지 않는다. 문장에 car가 있어도 이미지 후보에 car가 없으면 이번 추출 대상이 아니다.

## 현재 응답 형식

다음 클래스는 `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/src/mm_sae/data/caption_annotations.py`에 있다. API 호출에 `response_format=CaptionAnnotation`을 전달한다.

```python
from pydantic import BaseModel, ConfigDict


class ObjectSpans(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    concept_id: int
    spans: list[str] | None


class CaptionAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    objects: list[ObjectSpans]
```

`spans=["dog", "he"]`는 모델이 복사한 원문 표현이다. 같은 표현은 한 번만 반환한다. `spans=[]`는 해당 후보의 표현이 없다는 뜻이고 `spans=null`은 주어진 후보 중 어느 것인지 구별할 수 없다는 뜻이다. 모든 후보의 번호와 순서를 그대로 반환해야 한다. 정의하지 않은 필드와 암묵적인 자료형 변환은 허용하지 않는다.

코드는 반환한 문자열의 대소문자와 단어 경계가 맞는 모든 위치를 찾아 문자 범위를 저장한다. 시작은 포함하고 끝은 포함하지 않는다. 원문에 없는 표현과 잘못된 범주 목록은 실패로 처리한다. 같은 위치가 반복되면 한 번만 저장한다. **같은 문자열의 뜻이 위치마다 다른지는 구별하지 않는다.** 이는 모든 일치 위치를 가리는 이번 방식의 한계다. 원문 검사 통과가 의미상 올바른 범주 배정을 보장하지도 않는다.

저장한 결과의 각 줄에는 `split`, `caption_id`, `image_id`, `original`, `image_concept_ids`, `objects`가 들어간다. objects 안에는 코드가 계산한 `text`, `start`, `end`가 저장된다. 따라서 API 응답 형식을 줄여도 저장된 결과의 구조는 이전과 같다. 이 결과를 사람이 검수한 주석으로 표시하지 않는다.

실험 코드는 문자 범위를 토큰 위치로 변환해 해당 ID를 모델의 unknown token ID로 치환한다. 캡션의 문자열을 삭제하거나 문자 그대로 unknown token을 삽입하지 않는다. 나머지 ID, 입력 길이, attention mask와 원래 종료 위치를 유지한다. 서로 다른 범주의 위치가 겹치는 경우는 기존 가림 준비 코드에서 가림 불가로 처리한다. 현재 시험 결과는 전체 실험에 사용할 주석으로 등록하지 않았다.

## 16개 시험에서 확인한 품질과 비용

원문 표현 검사는 GPT-5.4 mini, GPT-5.4와 GPT-5.1 모두 16개를 통과했다. Codex가 이 형식의 호출 전에 작성한 임시 범주 기준 131개와는 각각 125개, 125개, 122개가 일치했다. 표현 있음 27개는 세 모델 모두 찾았지만 모호한 후보 4개는 세 모델 모두 틀렸다. 범주 정의나 처리 규칙을 더 확인해야 하는 5개 후보는 평가에서 제외하고 별도로 기록했다. 131개 대부분이 표현 없음이므로 95.4%와 93.1%를 전체 추출 정확도로 읽어서는 안 된다. 사람이 검수한 정답과 비교한 결과도 아니다.

16개 요청 비용은 GPT-5.4 mini가 $0.01089525, GPT-5.4가 $0.0363475, GPT-5.1이 $0.02313875였다. 일반 API 요금에 실제 사용량을 적용한 값이다. 같은 평균 길이라고 가정하면 전체 616,767개는 각각 약 $420, $1,401, $892다. 표본이 전체 평균을 대표하지 않고 재시도·세금·서버 비용도 빠져 있으므로 대략적인 예산으로만 사용한다. [mini 공식 요금](https://developers.openai.com/api/docs/models/gpt-5.4-mini), [GPT-5.4 공식 요금](https://developers.openai.com/api/docs/models/gpt-5.4), [GPT-5.1 공식 요금](https://developers.openai.com/api/docs/models/gpt-5.1)

최신 프롬프트, 실제 응답과 항목별 오류, 비용 계산을 `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions/pilot16/all-matches`에 저장했다. [상세 비교 문서](/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions/pilot16/all-matches/README.md)에는 16개 원문과 세 모델의 실제 추출 표현을 함께 적었다.

## 실행과 진행 상황 확인

다음 명령은 최신 프롬프트와 응답 형식으로 고정된 캡션 16개만 처리한다. 같은 작업 폴더에 이미 완료된 응답이 있으면 API를 다시 호출하지 않는다.

```bash
cd /Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae
uv sync --extra annotation --extra dev
.venv/bin/python scripts/annotate_captions.py \
  --input annotations/coco_captions/pilot16/inputs.jsonl \
  --output annotations/coco_captions/pilot16/all-matches/gpt54-none.jsonl \
  --work-dir runs/caption-annotation/pilot16-gpt54-none-all-matches \
  --spec annotations/coco_captions/pilot16/all-matches \
  --env-file /Users/jihoonkwon/Desktop/projects/.env \
  --model gpt-5.4-2026-03-05 \
  --reasoning-effort none \
  --max-completion-tokens 2048 \
  --concurrency 512
```

GPT-5.1 시험에는 `gpt-5.1-2025-11-13`을 사용하고 출력 파일과 작업 폴더의 gpt54를 gpt51로 바꾼다. mini 시험에는 `gpt-5.4-mini-2026-03-17`을 사용하고 출력 파일과 작업 폴더의 gpt54를 gpt54-mini로 바꾼다. 세 모델에 `reasoning_effort="none"`, `temperature=0`, `top_p=1`, `n=1`, `store=false`를 사용했다. API가 보고한 추론 토큰은 모두 0개다. 모델 버전과 요청 내용을 고정하고 결과를 저장하지만 재호출이 항상 같은 답을 낸다고 보장하지 않는다.

`AsyncOpenAI`와 `asyncio.Semaphore(512)`를 사용한다. 요청 하나에는 캡션 하나를 넣고 동시에 진행 중인 요청은 프로세스당 최대 512개로 제한한다. API 요청 제한에는 Retry-After와 대기 후 재시도를 적용한다. 최초 요청을 포함해 최대 5회이며 인증·결제 오류와 원문 검사 실패는 반복 재시도하지 않는다. SDK 자체 재시도는 끄고 코드에서만 관리한다.

작업 폴더의 progress.json에 완료·실패·재시도 수와 토큰 사용량, 관측 속도에 따른 예상 남은 시간을 기록한다. 10초마다 그리고 요청이 끝날 때 갱신한다. summary.json에는 실행 합계를, invocations.jsonl에는 실행별 이력을 저장한다. 재개 전 완료한 요청의 사용량을 이번 실행 합계에 중복 포함하지 않는다. SQLite에는 완료 결과와 원본 응답을 보존한다.

프롬프트, 응답 형식, 모델 설정이나 입력이 바뀌면 같은 작업 폴더를 재사용할 수 없다. 지정한 response_format.json과 코드의 Pydantic 형식이 다르면 API 요청 전에 중단한다. 출현 순번을 사용한 이전 시험을 그대로 재현하려면 그 형식을 사용한 Git 커밋 a7f9d9d의 코드를 사용해야 한다. 이전 결과와 프롬프트·형식은 그대로 보존했다.

## 파일 관리와 전체 실행 범위

현재 프롬프트는 `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions/prompt.txt`이고 응답 형식은 같은 폴더의 response_format.json에 있다. examples.jsonl은 형식을 설명하는 가상 예시이며 API에 보내지 않는다. prompt.proposed.txt에는 별도의 table 모호성 예시 제안이 있지만 아직 실행에 사용하지 않았다.

API 키는 `/Users/jihoonkwon/Desktop/projects/.env`에서 읽는다. `.gitignore`로 환경 파일, 개인키, 원본 API 응답, SQLite와 로그, 모델과 데이터 파일을 제외한다. 공개 COCO 캡션에서 추출한 주석, 프롬프트와 형식, 비용·품질 기록만 Git에 넣는다.

**사용자가 전체 실행을 명시적으로 허락한 뒤에만 전체 캡션 추출과 SAE 전체 실험을 진행한다.** 작은 시험이 통과해도 자동으로 전체를 시작하지 않는다. 전체 이미지별 후보 목록 생성과 결과 분할 저장은 아직 실행하지 않았다. 전체 결과는 학습·검증 분할별로 캡션 번호 순서대로 저장하고 파일당 최대 5,000줄 또는 16MiB를 넘기기 전에 나누도록 준비할 예정이다.
