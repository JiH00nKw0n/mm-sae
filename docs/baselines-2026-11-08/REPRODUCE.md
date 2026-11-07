### 저장된 결과로 표와 그림 다시 만들기

[측정값](measurements.json)이 저장소에 포함되어 있습니다. GPU와 원본 자료 없이 아래 명령으로 표와 그림을 다시 만들 수 있습니다. 측정 파일에 기록한 방법과 패널을 복원합니다.

```bash
git clone https://github.com/JiH00nKw0n/mm-sae.git
cd mm-sae
python3 -m venv .venv-report
.venv-report/bin/python -m pip install numpy matplotlib pyyaml
.venv-report/bin/python scripts/report_baselines.py \
  docs/baselines-2026-11-08/measurements.json docs/baselines-2026-11-08 \
  --config configs/report/coco_val2017_top16.yaml
```

### 같은 자료에서 대응 방법 다시 평가하기

Docker와 NVIDIA GPU를 준비합니다. [평가 설정](../../configs/coco_val2017_top16.yaml)의 상대 경로에 맞춰 [원본 이미지와 주석](../benchmarks.md)을 배치합니다. 호스트의 `/datasets/mm-sae`에는 원본 자료를, `/cache/mm-sae/cc3m`에는 기존 CC3M 임베딩 캐시를, `/checkpoints/cc3m/models`에는 이미지·텍스트 SAE 가중치를 둡니다. 기존 SAE와 캐시 식별값은 [기준 실험 기록](../baselines-2026-11-07/provenance.json)에, 추가 대응 행렬과 두 공유 SAE의 가중치 식별값은 [추가 실험 기록](extended-provenance.json)에 저장했습니다.

```bash
make build TARGET=gpu IMAGE=mm-sae:gpu
for method in hungarian pls cca sparse_pls sparse_cca cca_cov cca_cov64 sinkhorn procrustes; do
  make run TARGET=gpu IMAGE=mm-sae:gpu DATA=/datasets/mm-sae CACHE=/cache/mm-sae \
    RUNS=/results/mm-sae SAE=/checkpoints/cc3m/models \
    ARGS="run configs/coco_val2017_top16.yaml --mapping configs/mapping/${method}.yaml"
done

```

각 조합에 허용하는 SAE 활성값 개수는 평가 설정의 `concept_feature_budget`에서 지정합니다. 이 보고서에서는 16을 사용합니다. 검색과 개념 예측의 점수를 따로 계산하며, 개념 예측에서는 제한한 계수로 분류기를 새로 학습하고 검증 분할에서 대표 축을 선택합니다.

학습·평가 결과는 `/results/mm-sae/coco-val2017-top16/mappings/` 아래에 저장됩니다. 각 방법 폴더의 `fit.json`에 학습 자료와 가중치 식별값이, `evaluation` 아래에 개념별 시험 손실·검증 손실 행렬·검색 순위가 기록됩니다. COCO-Stuff와 Visual Genome의 범주별 결과는 같은 실행에서 해당 개념 열을 선택해 집계합니다.

### 어떤 설정을 바꾸면 되나

| 바꾸려는 내용 | 수정할 위치 |
|---|---|
| 평가에 포함할 이미지, 영역 문장의 주석 종류, 등장 이미지 수 | [평가 설정](../../configs/coco_val2017_top16.yaml)의 `benchmarks`에서 지정합니다. `min_test_images`는 양성 이미지 수와 해당 양성 주석이 없는 이미지 수에 각각 적용됩니다. `regions.annotated_only`는 지정한 주석 종류가 하나도 없는 문장을 제외합니다. |
| 조합 수, CCA 규제 값, 회전 여부 | [대응 방법 설정](../../configs/mapping/)에서 지정합니다. |
| 표의 물체·배경·관계 묶음과 그림에 사용할 지표 | [표와 그림 설정](../../configs/report/coco_val2017_top16.yaml)에서 지정합니다. 각 개념의 저장된 시험 손실과 조합별 중요도 행렬이 있으면 묶음을 바꿔 다시 집계할 수 있습니다. |

서버에서는 저장된 SAE 활성값의 이미지 ID와 주석의 이미지 ID를 대조하고 같은 입력을 세 방법에 사용했습니다. Visual Genome에서 동일한 문장은 한 번 인코딩한 뒤 모든 원래 영역과 주석을 복원했습니다. 분류기의 학습과 검증에서는 표본 점수 분포의 합을 GPU에서 수치적으로 계산하고, 최종 시험 손실은 선택한 조합의 원래 표본으로 직접 구했습니다. COCO-Stuff의 36개 조합·개념 분류기를 CPU 직접 계산과 대조했을 때 검증 손실의 최대 절대 차이는 0.000000002 미만이었습니다.

공유 SAE 두 조건은 자체 가중치를 학습합니다. 같은 CC3M CLIP 임베딩 캐시를 마운트하고 아래 명령으로 실행합니다. 각 설정에는 원 코드의 공유 SAE 폭, 활성 좌표 수, 보조 목적함수의 세기와 학습 횟수가 들어 있습니다. 기존 가중치를 재사용하려면 해당 조건에서 학습한 공유 SAE를 양쪽에 담은 `models` 폴더를 `SAE`로 넘깁니다.

```bash
for method in group_sparse iso_align; do
  make run TARGET=gpu IMAGE=mm-sae:gpu DATA=/datasets/mm-sae CACHE=/cache/mm-sae \
    RUNS=/results/mm-sae ARGS="run configs/${method}.yaml"
done
```

전체 비교표를 다시 만들 때는 세 실행 폴더의 공통 상위 폴더인 `/runs`를 보고서 입력으로 사용합니다. [그룹 희소성 설정](../../configs/group_sparse.yaml) · [Iso-Energy 설정](../../configs/iso_align.yaml)

```bash
docker run --rm --entrypoint python \
  -v "$PWD:/source:ro" -v /results/mm-sae:/runs:ro \
  -v "$PWD/docs/baselines-2026-11-08:/report-out" \
  mm-sae:gpu /source/scripts/report_baselines.py /runs /report-out \
  --config /source/configs/report/coco_val2017_top16.yaml
```
