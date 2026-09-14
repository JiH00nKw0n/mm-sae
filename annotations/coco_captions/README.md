# 캡션의 객체 표현 추출

**캡션 하나를 요청 하나로 보내며, `AsyncOpenAI`와 `asyncio.Semaphore(512)`로 동시에 처리한다.** 프롬프트에서 hot dog와 동물 dog를 구별하는 예시 문장만 삭제한 뒤 같은 실제 COCO 캡션 16개를 다시 시험했다. GPT-5.4 `none`은 원문 위치 검사를 16개 모두 통과했고 GPT-5.1 `none`은 12개를 통과했다. 두 모델 모두 의미 오류가 남았다. 전체 캡션 추출과 SAE 전체 실험은 시작하지 않았다.

최신 비교 결과와 실제 16개 응답은 [GPT-5.4와 GPT-5.1 비교](/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions/pilot16/without-hot-dog-example/README.md)에 정리했다. 두 모델의 실제 사용량으로 계산한 전체 616,767개 비용은 각각 약 $1,624와 $1,013이다. 16개 기능 시험을 전체로 환산한 추정이며 재시도·세금·서버 비용을 포함하지 않는다.

## 입력과 저장 자료

입력 대상은 공식 COCO2017 학습 캡션 591,753개와 검증 캡션 25,014개다. 각 캡션에는 대응 이미지의 COCO-Stuff 원본 픽셀 주석에 실제로 나타난 범주 번호와 이름만 제공한다. 중앙 자르기 전 원본 주석을 사용한다. 다른 캡션, 동의어 사전, SAE 활성값과 상관계수는 제공하지 않는다.

이 자료는 이미지에 있는 후보 중 무엇을 캡션이 지칭하는지 기록한다. 이미지 후보 없이 텍스트만 분류한 결과가 아니다. 문장에 `car`가 있어도 이미지 주석의 후보에 `car`가 없으면 이번 추출 대상에 들어가지 않는다. 같은 이미지의 모든 캡션은 같은 후보 목록을 받는다.

현재 파일들은 `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions`에 있다.

| 파일 | 기록하는 내용 |
| --- | --- |
| `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions/prompt.txt` | 지정한 예시 문장만 삭제한 현재 프롬프트다. GPT-5.4와 GPT-5.1 재시험에 사용했다. |
| `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions/prompt.proposed.txt` | 모호한 후보를 하나로 단정한 오류를 줄이기 위한 수정안이다. 승인과 시험을 거치지 않았으며 실행에 사용하지 않았다. |
| `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions/response_format.json` | Pydantic 클래스에서 생성한 엄격한 JSON 응답 형식이다. |
| `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions/examples.jsonl` | 형식을 설명하는 가상 예시 3개다. 실제 API 결과가 아니다. |
| `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions/pilot16/prompt.txt` | 예시 문장을 삭제하기 전 시험에 사용한 프롬프트를 보존했다. |
| `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions/pilot16/without-hot-dog-example/manifest.json` | 예시 문장을 삭제한 뒤 두 모델을 비교한 최신 설정과 비용을 기록한다. |
| `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions/pilot16/inputs.jsonl` | 모든 모델 시험에 공통으로 보낸 16개 캡션과 이미지 범주 목록을 담는다. |
| `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions/pilot16/gpt4o-accepted.jsonl` | GPT-4o의 원문 위치 검사 통과 결과 9개다. 의미 정답으로 승인한 자료가 아니다. |
| `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions/pilot16/gpt4o-rejected.jsonl` | GPT-4o의 실패 캡션 7개를 진단용으로 재요청한 응답과 실패 이유를 담는다. |
| `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions/pilot16/gpt5-low.jsonl` | GPT-5 `low`의 결과 16개다. 알려진 의미 오류를 수정하지 않고 보존했다. |
| `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions/pilot16/manifest.json` | 예시 문장을 삭제하기 전 요청 설정, 사용량, 입력과 출력의 파일 지문, 알려진 오류를 기록한다. |
| `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions/pilot16/README.md` | 실제 16개 결과와 검사에서 확인한 범위, 수정이 필요한 이유를 설명한다. |

## 응답을 정의한 클래스

`/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/src/mm_sae/data/caption_annotations.py`에 다음과 같이 선언했다. API 호출에 `response_format=CaptionAnnotation`을 전달한다.

```python
from pydantic import BaseModel, ConfigDict


class Span(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str
    occurrence: int


class ObjectSpans(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    concept_id: int
    spans: list[Span] | None


class CaptionAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    objects: list[ObjectSpans]
```

`text`는 원문 그대로의 객체 표현이다. `occurrence`는 그 문자열이 대소문자와 단어 경계를 유지하며 몇 번째로 나오는지 나타낸다. 1부터 시작하며 문장 속 단어 번호가 아니다. `A hot dog beside a dog.`에서 동물인 `dog`는 두 번째 출현이다.

`spans=[]`는 해당 후보의 긍정적 표현이 없다는 뜻이다. `spans=null`은 표현을 어느 후보에 배정해야 할지 구별할 수 없다는 뜻이다. 예를 들어 `table`과 `dining table`이 모두 후보인데 문장에 `table`만 나오면 두 후보 모두 `null`이어야 한다. `spans`에는 기본값이 없으므로 빈 목록이나 `null`인 경우에도 필드를 반드시 반환해야 한다. `extra="forbid"`는 정의하지 않은 필드를 금지하고, `strict=True`는 자료형의 암묵적 변환을 막는다.

모델은 가린 문장을 만들지 않는다. 원문 표현과 위치만 반환한다. 코드는 후보 번호와 순서가 입력과 정확히 같은지 확인하고, 원문에서 표현의 시작·끝 위치를 계산한다. 시작 위치는 포함하고 끝 위치는 포함하지 않는 Python 문자열 기준이다. 존재하지 않는 표현, 잘못된 출현 순번과 중복 위치는 실패로 기록한다. API 실패나 `null`을 객체 부재로 바꾸지 않는다. **이 검사는 원문과 형식을 검증할 뿐, 범주에 의미상 맞게 배정했는지를 보장하지 않는다.**

각 출력 줄에는 `split`, `caption_id`, `image_id`, `original`, `image_concept_ids`, `objects`가 들어간다. 모델 생성 주석을 `human_reviewed`로 기록하지 않는다. 실험 코드는 저장된 문자 범위를 토큰 위치로 바꾼 뒤 해당 토큰을 각각 unknown token으로 치환한다. 문자열 삭제나 생성형 문장 보완은 하지 않는다. 다른 토큰, 입력 길이, attention mask와 원래 종료 위치는 유지한다. 이 변경은 추출 프롬프트나 응답 형식을 바꾸지 않는다. 현재 16개 시험 결과는 의미 검토가 끝난 전체 실험 입력으로 등록하지 않았다.

## 실행과 재개

프로그램은 `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/scripts/annotate_captions.py`다. 다음 명령은 예시 문장을 삭제하기 전의 프롬프트로 저장된 입력 16개를 GPT-5 `low`로 처리한다. 동일한 작업 폴더에 이미 완료된 응답이 있으면 API를 다시 호출하지 않는다.

```bash
cd /Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae
uv sync --extra annotation --extra dev
.venv/bin/python scripts/annotate_captions.py \
  --input annotations/coco_captions/pilot16/inputs.jsonl \
  --output annotations/coco_captions/pilot16/gpt5-low.jsonl \
  --work-dir runs/caption-annotation/pilot16-gpt5-low \
  --env-file /Users/jihoonkwon/Desktop/projects/.env \
  --spec annotations/coco_captions/pilot16 \
  --model gpt-5-2025-08-07 \
  --reasoning-effort low \
  --max-completion-tokens 4096 \
  --concurrency 512
```

GPT-4o 시험에는 `gpt-4o-2024-11-20`, `temperature=0`, `top_p=1`, `seed=0`, `n=1`, 빈도·존재 페널티 0과 출력 상한 2,048을 사용했다. GPT-5 시험에는 `gpt-5-2025-08-07`, `reasoning_effort="low"`, `n=1`과 추론 토큰을 포함한 출력 상한 4,096을 사용했다. GPT-5는 `temperature`와 `top_p`를 지원하지 않아 전송하지 않았으며, 시드와 페널티도 전송하지 않았다. 두 모델 모두 `store=false`를 사용했다. 모델 버전과 요청 내용을 고정하고 실제 결과를 저장해 재사용하지만, 재호출이 항상 같은 결과를 낸다고 주장하지 않는다. [GPT-5 공식 정보](https://developers.openai.com/api/docs/models/gpt-5), [매개변수 지원 범위](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.2)

요청 하나에는 캡션 하나만 넣는다. 입력 16개 시험에서는 최대 16개가 동시에 실행됐다. 입력이 많으면 동시에 진행 중인 API 요청을 최대 512개로 제한한다. API 요청 제한 응답에는 `Retry-After`와 대기 후 재시도를 적용한다. 최초 요청을 포함해 최대 5회이며, 인증·결제 오류와 원문 검사 실패를 반복 재시도하지 않는다. SDK의 자체 재시도는 꺼서 중복 재시도를 막는다.

완료 응답과 원본 응답은 작업 폴더의 SQLite 파일에 저장한다. 프롬프트, 응답 형식, 모델 설정이나 입력이 바뀌면 같은 작업 폴더를 재사용할 수 없다. 작업 폴더의 `progress.json`은 완료 수, 실패 수, 재시도 수, 토큰 사용량, 관측 속도에 따른 예상 남은 시간을 10초마다 기록한다. 요청이 끝날 때도 갱신한다. 각 실행의 합계는 `summary.json`과 누적 `invocations.jsonl`에 남긴다. 재개 전 완료된 요청의 사용량은 이번 실행 합계에 중복 포함하지 않는다.

API 키는 `/Users/jihoonkwon/Desktop/projects/.env`에서 읽는다. 키, 원본 응답, SQLite 파일과 진행 로그는 Git에 포함하지 않는다. `.gitignore`에서 환경 파일, 개인키, 로그, 임시 파일, 데이터와 모델 파일을 제외한다.

## 전체 추출 전 남은 일

현재는 16개 시험과 수정안 검토 단계다. 모호성 오류를 해결하고 사용자가 명시적으로 허락한 뒤에만 전체 추출을 진행한다. 소규모 검사가 통과해도 전체 캡션 추출이나 SAE 전체 실험을 자동으로 시작하지 않는다. 전체 자료의 이미지별 후보 목록 생성과 결과 분할 저장은 아직 실행하지 않았다. 전체 결과는 학습·검증 분할별로 캡션 번호 순서대로 저장하고, 파일당 최대 5,000줄 또는 16MiB를 넘기기 전에 나누도록 준비할 예정이다. 실제 파일별 크기와 내용 지문을 기록하고 실패가 남으면 전체 완료로 표시하지 않는다.
