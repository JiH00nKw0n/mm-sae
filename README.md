# MM-SAE

이미지와 텍스트의 희소 오토인코더를 학습하고, 여러 연구 질문에서 학습된 특징을 재사용하는 실험 저장소다. 희소 오토인코더(SAE)는 입력 벡터를 소수의 활성 특징으로 표현하고 원래 벡터를 복원하는 모델이다. 공용 코드는 데이터, 모델, 학습, 특징 추출과 통계를 담당한다. 첫 번째 실험인 RQ1은 동시 활성화 상관계수가 객체 대응과 실제 짝짓기를 얼마나 잘 반영하는지 평가한다.

CLIP 이미지·텍스트 모델과 전처리는 Hugging Face `CLIPModel`과 `CLIPProcessor`를 사용한다. 학습과 자료 묶음 생성은 `Trainer`, `TrainingArguments`, `datasets.Dataset`에 맡긴다. SAE의 계산과 학습 후 디코더 정규화만 직접 구현했다. RQ1을 추가하기 위해 별도 학습기나 자료 로더를 만들지 않는다.

**실제 자료를 쓰는 소규모 실행부터 확인할 수 있다.** `smoke-real.yaml`은 COCO 학습 이미지 4장과 검증 이미지 2장, 각 이미지의 모든 캡션을 사용한다. 실제 CLIP을 실행하고 이미지 SAE와 텍스트 SAE를 각각 최대 2단계 학습한 뒤 객체 가림과 분석을 수행한다. 이 설정은 실행 경로를 확인하기 위한 것이며 연구 결과를 추정하기에는 너무 작다. 평가 가능한 오류 사례가 없으면 이를 결과에 명시한다.

저장소를 내려받고 CPU용 Docker 이미지를 만든 뒤 실행한다. Docker 컨테이너 내부의 프로젝트 경로는 항상 `/workspace`다.

```bash
git clone https://github.com/JiH00nKw0n/mm-sae.git
cd mm-sae
mkdir -p data cache runs
docker build --target cpu -t mm-sae:cpu .
docker run --rm --shm-size=2g \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/cache:/workspace/cache" \
  -v "$PWD/runs:/workspace/runs" \
  mm-sae:cpu --config /workspace/configs/smoke-real.yaml
```

처음에는 공개 CLIP 가중치와 COCO 주석을 내려받는다. 소규모 설정에서는 압축 파일의 필요한 구간만 읽어 선택된 이미지와 PNG 주석만 가져온다. 이미 내려받은 자료와 CLIP 가중치는 다음 실행에서도 재사용한다. 공개 자료를 받기 위한 Hugging Face 토큰은 필요하지 않다.

**전체 실행은 GPU용 이미지와 전체 설정을 사용한다.** `coco.yaml`에는 이미지 수 제한이나 학습 단계 제한이 없다. NVIDIA GPU를 Docker에 연결할 수 있는 서버에서 다음 명령을 실행한다. 이 명령은 전체 데이터를 준비하고 SAE를 실제로 학습하므로 소규모 점검용으로 실행하지 않는다.

```bash
docker build --target gpu -t mm-sae:gpu .
docker run --rm --gpus all --shm-size=8g \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/cache:/workspace/cache" \
  -v "$PWD/runs:/workspace/runs" \
  -v "$PWD/configs:/workspace/configs:ro" \
  mm-sae:gpu --config /workspace/configs/coco.yaml
```

전체 실행은 공식 COCO2017 학습 이미지 118,287장과 해당 이미지의 캡션 591,753개를 사용한다. 이미지 SAE는 이미지별 한 행을, 텍스트 SAE는 캡션별 한 행을 학습한다. 따라서 텍스트 학습 행 수와 갱신 횟수는 이미지보다 대략 다섯 배 많다. 실제 개수와 이미지당 캡션 수 분포는 `/workspace/runs/coco-rq1/dataset.json`과 `/workspace/runs/coco-rq1/training.json`에 남긴다. 이미지당 캡션이 정확히 다섯 개라고 가정하지 않는다.

전체 설정은 양쪽 각각 특징 4,096개, 활성 특징 최대 8개, 30회 전체 자료 학습, 묶음 크기 1,024를 사용한다. AdamW 학습률은 0.0005, 가중치 감쇠는 0.00001이다. 처음 5%의 갱신 동안 학습률을 올린 뒤 코사인 함수에 따라 줄인다. 세부 값은 `/workspace/configs/coco.yaml`의 `training.arguments`에 있으며 Hugging Face의 인자 이름을 그대로 사용한다. 필요한 값만 덮어쓸 수 있으며 생략한 값은 공용 기본값을 유지한다.

기존 연구의 COCO 설정은 Karpathy 분할과 캡션마다 반복된 이미지 행을 사용했다. 이 저장소의 기본값은 공식 COCO2017 분할과 이미지별 한 행이다. 기존 자료 가중 방식을 확인하려면 `training.sampling`을 `paired_repeat_image`로 바꾼다. 이 옵션만 바꿔도 Karpathy 분할로 바뀌는 것은 아니다. [구현 근거와 차이](https://github.com/JiH00nKw0n/mm-sae/blob/main/docs/provenance.md)에 비교 내용을 기록했다.

**새 연구 질문은 공용 코드를 가져와 독립된 실험으로 작성한다.** 컨테이너 내부 구조는 다음과 같다.

```text
/workspace/
  src/mm_sae/
    data/          COCO 자료 준비, 주석 읽기, 문장 편집
    models/        Hugging Face CLIP 연결, TopK SAE
    metrics/       상관계수, AUROC, 짝짓기, 활성값 변화
    config.py      공용 설정 검증
    cache.py       여러 실험이 공유하는 CLIP 표현의 저장 위치
    features.py    원본·가림 자료의 표현과 희소 활성값 추출
    training.py    Hugging Face Trainer 연결과 모델 저장
    io.py          중단 후 재개, 파일 검증, 동시 쓰기 제어
    progress.py    진행 상태 저장과 관측 속도에 따른 예상 남은 시간
    cli.py         설정에 지정한 실험 실행
  experiments/rq1/
    run.py         RQ1의 실행 순서
    config.py      RQ1에서만 쓰는 설정
    selection.py   원본·제거 AUROC로 대표 특징 선택
    correspondence.py  원본 점수와 실제 연결 평가
    interventions.py   객체 제거 후 점수와 연결 재계산
    plotting.py    세 본문 그림과 원자료 요약 저장
  configs/         전체 실행과 두 종류의 소규모 점검 설정
  tests/           통계 정의, 자료 대응, 재사용·재개 동작 검사
```

예를 들어 두 번째 연구 질문은 `/workspace/experiments/rq2/run.py`에 `run(config, store, stage)`를 작성하고 설정의 `experiment.name`을 `rq2`로 지정한다. 실험별 설정은 `experiment.options`로 받는다. 공용 코드에 RQ2 조건문을 추가할 필요가 없다. [확장 방법](https://github.com/JiH00nKw0n/mm-sae/blob/main/docs/extending.md)에 구현 규약과 학습된 SAE를 재사용하는 설정 예시를 적었다.

**RQ1은 한 번 학습한 모델을 고정하고 세 실험을 수행한다.** 객체별 대표 특징은 원본과 해당 객체를 제거한 입력의 활성값을 AUROC로 비교해 고른다. AUROC는 원본 점수가 제거본 점수보다 높을 확률을 순위로 측정한 값이며 동점은 절반으로 센다. 별도 임계값은 적용하지 않는다.

1. 서로 다른 객체의 모든 순서 있는 조합에서, 이미지 주석의 동시 등장 상관계수를 0.2 간격으로 나눈다. 각 구간에 해당하는 이미지·텍스트 대표 특징 상관계수의 평균과 표준편차를 그림으로 보여준다. 일부 조합을 추출하지 않는다.
2. 전체 활성 특징에서 이미지마다 최고점을 고르는 Greedy와 전체 점수 합을 최대화하는 일대일 헝가리안 연결을 각각 구한다. 그중 대표 특징의 실제 상대가 같은 객체인지, 다른 객체인지, 판정할 수 없는지 평가한다.
3. 원본 연결에서 확인된 서로 다른 객체의 오류를 대상으로 객체를 제거하고 상관행렬과 두 연결 방식을 다시 계산한다. 동시 등장 관계를 약화하는 제거를 같은 양의 무작위 제거와 비교한다. 올바른 연결 회복, 기존 올바른 연결의 손상, 꺼진 특징 수를 함께 기록한다.

정확한 표본 정의, AUROC 계산, 가릴 입력 수 결정식, 세 그림의 해석 범위는 [RQ1 실험 규약](https://github.com/JiH00nKw0n/mm-sae/blob/main/docs/rq1-protocol.md)에 설명했다. 기본 설정에서는 학습, 대표 선택, 상관 계산에 `train2017`을 사용한다. `val2017`에서는 선택한 특징 번호를 바꾸지 않고 제거 반응을 확인한다. 가림 과정에서 CLIP이나 SAE를 다시 학습하지 않는다.

**COCO-Stuff는 이미지의 픽셀 주석을 제공하지만 캡션의 객체 표현 주석은 제공하지 않는다.** PNG 정수값으로 영역을 읽으며, 캡션은 이미지 식별자로 공식 COCO 캡션과 연결한다. 이미지의 객체 표지를 캡션에 그대로 붙이지 않는다. 기본 캡션 표지는 범주명·동의어를 찾는 자동 규칙이다. 사람이 확인한 의미 정답으로 취급하면 안 된다. 텍스트는 객체 표현의 토큰을 기존 unknown token으로 바꾸며 문장 길이와 원래 종료 위치를 유지한다. 가림 위치 목록과 검토한 주석을 입력하는 방법은 [자료와 주석](https://github.com/JiH00nKw0n/mm-sae/blob/main/docs/data.md)에 적었다.

**설정과 저장 결과가 맞을 때 완료된 단계를 재사용한다.** 같은 명령을 다시 실행하면 완료한 단계를 건너뛴다. 중단된 표현 추출은 마지막으로 저장된 자료 묶음부터, 학습은 마지막 학습 상태 파일부터 재개한다. 코드나 설정을 바꾸면 `output`에 새 폴더를 지정해야 한다. 모델 가중치와 SAE 설정의 파일 지문을 대조해 분석 중 변경을 감지한다.

설정만 검사하려면 실행 명령 끝에 `validate`를 붙인다. 특정 단계를 실행하려면 `prepare`, `embed`, `train`, `select`, `correlate`, `experiment1`, `experiment2`, `experiment3`, `report` 중 하나를 붙인다. 앞 단계가 끝나지 않았으면 필요한 단계를 알리고 멈춘다.

결과는 `output` 아래에 저장한다. 주요 산출물은 원본 전체 상관행렬인 `panel.npz`, 대표 특징의 선택·검증 반응인 `representatives.csv`, 각 실험의 전체 CSV·JSON, 그리고 세 그림의 PNG·PDF다. 전체 설정의 그림 경로는 `/workspace/runs/coco-rq1/rq1/figures`다. 행렬에서 상관이 정의되지 않는 특징은 별도 표지로 보존한다.

**개발 점검은 인공 자료만으로도 실행할 수 있다.** 다음 명령은 Docker 없이 Python 3.11과 `uv`가 있는 환경에서 실행한다. `smoke.yaml`은 자동 검사 전용 인공 자료이고, 실제 COCO 확인에는 `smoke-real.yaml`을 사용한다.

```bash
uv sync --frozen --extra dev
uv run pytest -q
uv run pyright --pythonpath .venv/bin/python
uv run ruff check src experiments tests scripts
uv run ruff format --check src experiments tests scripts
uv run python -m mm_sae --config configs/smoke.yaml
```

의존성의 기준은 `pyproject.toml`과 `uv.lock`이다. Docker용 목록은 `uv run python scripts/export_runtime_lock.py`로 만든다. Torch의 CPU·CUDA 배포판은 Docker 빌드 대상에 따라 설치한다. [실행 확인 기록](https://github.com/JiH00nKw0n/mm-sae/blob/main/docs/validation.md)은 실제로 검사한 범위와 아직 실행하지 않은 범위를 구분한다.

서버 전체 실행에는 승인 검사를 켠 `/mnt/working/mm-sae/configs/elice.yaml`을 사용한다. [전체 실행 전 확인할 설정](https://github.com/JiH00nKw0n/mm-sae/blob/main/docs/elice-rq1-review.md)에 자료와 판정 기준, 그림 3개, 해석의 한계를 적었다. [서버 준비와 진행 조회](https://github.com/JiH00nKw0n/mm-sae/blob/main/docs/server.md)에 설치와 조회 명령을 적었다. 상태는 10초마다 저장하며 현재 작업의 예상 남은 시간을 관측 속도로 계산한다. 아직 수행하지 않은 단계의 소요 시간은 미산정으로 남긴다.
