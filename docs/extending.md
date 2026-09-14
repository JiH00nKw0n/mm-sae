# 공용 코드와 새 실험

새 연구 질문은 기존 데이터, 모델, 학습 함수와 통계를 불러 사용한다. 실험별 표본 선택, 비교 조건, 결과 해석만 `/workspace/experiments/<name>` 아래에 둔다. `/workspace/src/mm_sae`에서 특정 연구 질문의 모듈을 가져오지 않는다.

명령행 실행기는 `experiment.name`으로 모듈을 찾는다. 아래 예시를 `/workspace/experiments/rq2/run.py`로 저장하고 같은 폴더에 `__init__.py`를 두면 새로운 실행 경로를 만들 수 있다. 예시는 자료 준비와 학습까지 담당하며 실제 RQ2 평가 내용은 추가해야 한다.

```python
from mm_sae.data.index import prepare
from mm_sae.features import embed
from mm_sae.models.encoder import make_encoder
from mm_sae.training import train, validate_training_arguments

STAGES = ["prepare", "embed", "train"]

def validate(config):
    validate_training_arguments(config)

def run(config, store, stage="all"):
    if stage != "all" and stage not in STAGES:
        raise ValueError(f"Unknown stage {stage}")
    encoder = None

    def get_encoder():
        nonlocal encoder
        if encoder is None:
            encoder = make_encoder(config.encoder)
        return encoder

    actions = {
        "prepare": lambda: prepare(config, get_encoder()),
        "embed": lambda: embed(config, get_encoder()),
        "train": lambda: train(config),
    }
    dependencies = {"prepare": [], "embed": ["prepare"], "train": ["embed"]}
    for name in STAGES if stage == "all" else [stage]:
        if store.done(name):
            continue
        store.require(*dependencies[name])
        actions[name]()
        store.complete(name)
```

설정에서는 `experiment.name`을 `rq2`로 바꾸고, `experiment.options`를 새 연구 질문의 설정으로 교체한다. RQ1에만 필요한 `RQ1Config`는 공용 설정의 일부가 아니다. 새 실험에서 필요하면 자신의 Pydantic 설정 클래스로 `experiment.options`를 검사한다.

**학습된 SAE를 재사용하려면 양쪽 모델 경로를 지정한다.** 다음은 전체 설정에서 변경할 부분이며 독립적으로 완성된 설정 파일은 아니다.

```yaml
output: /workspace/runs/rq2
cache: /workspace/cache
training:
  image_checkpoint: /workspace/runs/coco-rq1/models/image
  text_checkpoint: /workspace/runs/coco-rq1/models/text
  latent_size: 4096
  top_k: 8
experiment:
  name: rq2
  options: {}
```

양쪽 경로가 있으면 공용 `train` 단계는 학습을 수행하지 않고 모델을 불러와 입력 차원과 특징 수를 확인한다. 새 실행 폴더에도 모델을 저장하고 파일 지문을 기록한다. 한쪽 경로만 지정하면 설정 오류로 처리한다. 학습된 모델을 분석에 불러오는 `load_saes`는 기울기 계산을 끄고 평가 상태로 반환한다.

원본 CLIP 표현은 입력 이미지의 SHA-256 지문, 캡션 내용과 순서, 모델 버전, 전처리 설정이 같을 때 여러 실행에서 공유한다. SHA-256은 파일 내용의 변경을 확인하는 지문이다. 데이터가 연결된 폴더 경로나 연구별 객체 주석이 달라졌다는 이유만으로 원본 표현을 다시 계산하지 않는다. 여러 실행이 같은 표현 파일을 만들려고 하면 파일 잠금으로 쓰기 순서를 정한다. 같은 결과 폴더에 두 실행이 동시에 쓰는 작업은 거부한다. 읽기 전용 원본 자료를 사용하는 것을 전제로 하며 실행 중 데이터 파일을 바꾸지 않는다.

기본 `Encoder` 규약은 `dim`, `images`, `texts`, `visible_mask` 네 요소다. 이미지·텍스트 함수는 길이가 `dim`인 정규화된 벡터를 반환하고, `visible_mask`는 모델의 공간 전처리를 정수 주석에 적용한다. 다른 Hugging Face 모델을 추가할 때는 이 규약을 구현하고 설정 검증과 `make_encoder`에 등록한다. 실험의 상관계수나 짝짓기 코드를 수정할 필요는 없다.

현재 자료 준비 구현은 COCO와 검사 전용 인공 자료를 지원한다. 다른 데이터셋을 지원하려면 공용 `Index`가 읽는 이미지·캡션 행, 이미지 행 참조, 개념 존재 배열과 편집 정의를 생성하는 자료 준비 함수를 추가해야 한다. 아직 구현하지 않은 데이터셋이나 인코더를 설정 이름만 바꿔 지원한다고 가정하지 않는다.

학습 관련 새 옵션은 가능한 한 Hugging Face `TrainingArguments`로 전달한다. 직접 학습 반복문, 최적화기나 자료 묶음 생성기를 복제하지 않는다. 실행 폴더와 로깅 서비스, CPU 선택, 자료 열 보존, 메모리 고정 여부는 공용 실행기가 관리한다. 학습 외 실험의 중간 파일은 각 연구 질문의 결과 폴더에 두고 공용 모델 폴더를 변경하지 않는다.
