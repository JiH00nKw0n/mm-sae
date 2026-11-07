# COCO val2017에서 비교한 세 SAE 대응 방법

텍스트의 물체+관계 범위에서 이미지·텍스트 평균 개념 예측 손실이 가장 낮은 방법은 **CCA (λ=1)** (0.07154)이고, 양방향 평균 R@5가 가장 높은 방법은 **PLS-SVD** (36.39%)입니다.

| 방법 | 이미지 평균 손실 | 텍스트 평균 손실 | 모달리티 평균 손실 | 양방향 평균 R@5 (%) |
| --- | --- | --- | --- | --- |
| Hungarian | 0.12320 | 0.02907 | 0.07613 | 12.17490 |
| PLS-SVD | 0.11882 | 0.02689 | 0.07285 | 36.39317 |
| CCA (λ=1) | 0.11596 | 0.02712 | 0.07154 | 35.94141 |

![개념 예측과 이미지·텍스트 검색](baseline-frontier.png)

가로축은 활성값 조합 하나로 개념의 유무를 예측한 손실입니다. 오른쪽으로 갈수록 손실이 낮습니다. 이미지 쪽은 COCO 물체 80개와 COCO-Stuff 전체 171개의 자료별 평균에 같은 비중을 두었습니다. COCO 물체 80개가 두 자료 모두에 들어갑니다. 텍스트 쪽은 각 패널에 표시한 Visual Genome 개념의 평균입니다. 가로축에서 이미지와 텍스트가 차지하는 비중은 각각 50%입니다.

세로축은 정답 이미지 또는 캡션을 상위 5개 안에서 찾은 질의의 비율(R@5)을 두 검색 방향에 대해 평균한 값입니다. 세 패널은 왼쪽부터 텍스트의 물체, 관계, 물체와 관계 전체를 평가합니다. 점 하나가 대응 방법 하나입니다. 점선은 측정한 방법 중 다른 방법이 두 지표에서 동시에 같거나 더 좋고 적어도 하나에서 앞서는 경우가 없는 점들을 연결합니다. [그림 좌표](figure-points.csv) · [전체 개념을 쓴 비교표](baseline.csv)

### 무엇을 학습하고 어디에서 평가했나

- **대응 행렬 학습.** 기존 CC3M 이미지·텍스트 SAE와 임베딩 캐시의 2,905,954쌍을 그대로 사용했습니다. 헝가리안은 특징들의 상관계수 합을 최대화하는 일대일 대응에서 상관계수가 큰 256쌍을 골랐습니다. PLS-SVD는 양쪽 조합 점수의 공분산 합이 커지는 256개 방향을 구했습니다. CCA는 각 모달리티 내부의 공분산을 반영한 제약으로 256개 방향을 구했고 규제 값은 1이었습니다. 세 방법 모두 회전과 계수 자르기를 적용하지 않았습니다.
- **개념 예측.** 대응 행렬을 고정하고 train2017을 이미지 단위로 학습과 검증에 나눴습니다. 조합 하나의 점수로 개념 유무를 예측하는 로지스틱 분류기를 학습하고, 검증 손실이 가장 작은 조합을 개념마다 하나씩 골랐습니다. 그 조합의 val2017 이진 교차엔트로피 손실을 개념별로 구해 같은 비중으로 평균했습니다. 이 손실은 예측 확률이 정답과 잘 맞을수록 낮습니다.
- **이미지 평가.** COCO val2017의 5,000장에서 개념이 주석되었는지 예측했습니다. COCO는 물체 80개를, COCO-Stuff는 물체 80개와 배경 91개를 평가합니다. COCO-Stuff의 전체 값은 171개 개념별 결과의 평균입니다. [COCO-Stuff 주석](https://github.com/nightrome/cocostuff#downloads)
- **텍스트 평가.** COCO val2017과 Visual Genome에 공통인 2,170장 가운데 물체나 관계가 하나 이상 주석된 2,147장의 영역 문장 74,895개를 사용했습니다. 문장에 연결된 주석이 있으면 해당 개념의 정답을 1로 두었습니다. 물체·관계 주석이 모두 비어 있는 문장은 제외했습니다. 하나 이상의 주석이 있는 문장에서도 일부 개념이 누락될 수 있으므로 결과는 공개 주석과의 일치도로 해석합니다. [Visual Genome 영역 그래프](https://visualgenome.org/api/v0/api_home)
- **텍스트의 등장 횟수.** val2017의 서로 다른 이미지 50장 이상에서 해당 주석이 있고, 다른 50장 이상에서 해당 주석이 없는 개념을 포함했습니다. 같은 이미지에서 여러 문장이 같은 개념을 언급해도 한 장으로 셌습니다. 학습 분할의 빈도는 이 선정에 사용하지 않았습니다. 선정된 물체 170개와 관계 42개 모두 학습과 검증에서 양성과 음성이 있었습니다. 두 종류를 합친 손실은 212개 개념별 손실의 평균입니다. [개념별 이미지 수와 문장 수](visual-genome-concepts.csv)
- **혼합도와 분산도.** 개념의 출현율만 예측하는 상수 분류기보다 각 조합의 검증 손실이 얼마나 줄었는지 구합니다. 양의 감소량이 한 조합에서 여러 개념으로 퍼질수록 혼합도가 높고, 한 개념에서 여러 조합으로 퍼질수록 분산도가 높습니다. 두 값 모두 각 표에 포함된 개념만 골라 정규화한 엔트로피로 다시 계산했습니다. 개념 예측 손실과 함께 읽어야 합니다.
- **검색.** COCO val2017 이미지 5,000장과 캡션 25,014개 전체를 후보로 두고 코사인 유사도로 순위를 매겼습니다. 이미지 질의는 그 이미지의 정답 캡션 중 하나가, 캡션 질의는 원래 이미지가 상위 1개·5개·10개 안에 드는지 각각 측정했습니다.

COCO test2017의 40,670장 중 Visual Genome에 같은 COCO 이미지 ID가 있는 경우는 0장이었습니다. COCO-Stuff의 공개 정답 마스크도 train2017과 val2017에만 있습니다. 모든 최종 점수는 val2017에서 계산했습니다. [이미지 ID 대조 결과](split-overlap.json) · [COCO 공식 배포 목록](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm)

## 범주별 개념 예측

| 자료 | 개념 범위 | 방법 | 개념 수 | 시험 손실 | 혼합도 | 분산도 |
| --- | --- | --- | --- | --- | --- | --- |
| COCO | 물체 | Hungarian | 80 | 0.10185 | 0.57557 | 0.56185 |
| COCO | 물체 | PLS-SVD | 80 | 0.09998 | 0.74652 | 0.83806 |
| COCO | 물체 | CCA (λ=1) | 80 | 0.09671 | 0.71757 | 0.81966 |
| COCO-Stuff | 물체 | Hungarian | 80 | 0.09984 | 0.57671 | 0.56348 |
| COCO-Stuff | 물체 | PLS-SVD | 80 | 0.09779 | 0.74403 | 0.83803 |
| COCO-Stuff | 물체 | CCA (λ=1) | 80 | 0.09461 | 0.71668 | 0.81984 |
| COCO-Stuff | 배경 | Hungarian | 91 | 0.18383 | 0.71781 | 0.66388 |
| COCO-Stuff | 배경 | PLS-SVD | 91 | 0.17269 | 0.74684 | 0.82283 |
| COCO-Stuff | 배경 | CCA (λ=1) | 91 | 0.17091 | 0.73092 | 0.81270 |
| COCO-Stuff | 물체+배경 | Hungarian | 171 | 0.14454 | 0.68711 | 0.61691 |
| COCO-Stuff | 물체+배경 | PLS-SVD | 171 | 0.13765 | 0.76451 | 0.82994 |
| COCO-Stuff | 물체+배경 | CCA (λ=1) | 171 | 0.13522 | 0.74265 | 0.81604 |
| Visual Genome | 물체 | Hungarian | 170 | 0.02499 | 0.47832 | 0.58562 |
| Visual Genome | 물체 | PLS-SVD | 170 | 0.02278 | 0.75179 | 0.84371 |
| Visual Genome | 물체 | CCA (λ=1) | 170 | 0.02283 | 0.73842 | 0.83520 |
| Visual Genome | 관계 | Hungarian | 42 | 0.04560 | 0.59329 | 0.66681 |
| Visual Genome | 관계 | PLS-SVD | 42 | 0.04356 | 0.60052 | 0.83107 |
| Visual Genome | 관계 | CCA (λ=1) | 42 | 0.04450 | 0.60133 | 0.82785 |
| Visual Genome | 물체+관계 | Hungarian | 212 | 0.02907 | 0.50107 | 0.60171 |
| Visual Genome | 물체+관계 | PLS-SVD | 212 | 0.02689 | 0.74146 | 0.84120 |
| Visual Genome | 물체+관계 | CCA (λ=1) | 212 | 0.02712 | 0.72806 | 0.83374 |

## 이미지·텍스트 검색

| 자료 | 방법 | 이미지 수 | 캡션 수 | 이미지 R@1 (%) | 이미지 R@5 (%) | 이미지 R@10 (%) | 텍스트 R@1 (%) | 텍스트 R@5 (%) | 텍스트 R@10 (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COCO val2017 | Hungarian | 5000 | 25014 | 5.36000 | 13.28000 | 19.60000 | 4.08200 | 11.07000 | 16.31900 |
| COCO val2017 | PLS-SVD | 5000 | 25014 | 20.50000 | 41.24000 | 51.72000 | 14.30800 | 31.54600 | 42.09600 |
| COCO val2017 | CCA (λ=1) | 5000 | 25014 | 20.66000 | 41.22000 | 52.18000 | 14.01200 | 30.66300 | 40.67700 |

## 재현 방법

### 저장된 결과로 표와 그림 다시 만들기

[측정값](measurements.json)이 저장소에 포함되어 있습니다. GPU와 원본 자료 없이 아래 명령으로 표와 그림을 다시 만들 수 있습니다.

```bash
git clone https://github.com/JiH00nKw0n/mm-sae.git
cd mm-sae
python3 -m venv .venv-report
.venv-report/bin/python -m pip install numpy matplotlib pyyaml
.venv-report/bin/python scripts/report_baselines.py \
  docs/baselines-2026-11-07/measurements.json docs/baselines-2026-11-07 \
  --config configs/report/coco_val2017.yaml
```

### 같은 자료에서 세 방법 다시 평가하기

Docker와 NVIDIA GPU를 준비합니다. [평가 설정](../../configs/coco_val2017.yaml)의 상대 경로에 맞춰 [원본 이미지와 주석](../benchmarks.md)을 배치합니다. 호스트의 `/datasets/mm-sae`에는 원본 자료를, `/cache/mm-sae/cc3m`에는 기존 CC3M 임베딩 캐시를, `/checkpoints/cc3m/models`에는 이미지·텍스트 SAE 가중치를 둡니다. 가중치와 캐시의 식별값은 [실행 기록](provenance.json)에 있습니다.

```bash
make build TARGET=gpu IMAGE=mm-sae:gpu
for method in hungarian pls cca; do
  make run TARGET=gpu IMAGE=mm-sae:gpu DATA=/datasets/mm-sae CACHE=/cache/mm-sae \
    RUNS=/results/mm-sae SAE=/checkpoints/cc3m/models \
    ARGS="run configs/coco_val2017.yaml --mapping configs/mapping/${method}.yaml"
done
docker run --rm --entrypoint python \
  -v "$PWD:/source:ro" -v /results/mm-sae:/runs:ro \
  -v "$PWD/docs/baselines-2026-11-07:/report-out" \
  mm-sae:gpu /source/scripts/report_baselines.py /runs/coco-val2017 /report-out \
  --config /source/configs/report/coco_val2017.yaml
```

학습·평가 결과는 `/results/mm-sae/coco-val2017/mappings/` 아래에 저장됩니다. 각 방법 폴더의 `fit.json`에 학습 자료와 가중치 식별값이, `evaluation` 아래에 개념별 시험 손실·검증 손실 행렬·검색 순위가 기록됩니다. COCO-Stuff와 Visual Genome의 범주별 결과는 같은 실행에서 해당 개념 열을 선택해 집계합니다.

### 어떤 설정을 바꾸면 되나

| 바꾸려는 내용 | 수정할 위치 |
|---|---|
| 평가에 포함할 이미지, 영역 문장의 주석 종류, 등장 이미지 수 | [평가 설정](../../configs/coco_val2017.yaml)의 `benchmarks`에서 지정합니다. `min_test_images`는 양성 이미지 수와 해당 양성 주석이 없는 이미지 수에 각각 적용됩니다. `regions.annotated_only`는 지정한 주석 종류가 하나도 없는 문장을 제외합니다. |
| 조합 수, CCA 규제 값, 회전 여부 | [대응 방법 설정](../../configs/mapping/)에서 지정합니다. |
| 표의 물체·배경·관계 묶음과 그림에 사용할 지표 | [표와 그림 설정](../../configs/report/coco_val2017.yaml)에서 지정합니다. 각 개념의 저장된 시험 손실과 조합별 중요도 행렬이 있으면 묶음을 바꿔 다시 집계할 수 있습니다. |

서버에서는 저장된 SAE 활성값의 이미지 ID와 주석의 이미지 ID를 대조하고 같은 입력을 세 방법에 사용했습니다. Visual Genome에서 동일한 문장은 한 번 인코딩한 뒤 모든 원래 영역과 주석을 복원했습니다. 분류기의 학습과 검증에서는 표본 점수 분포의 합을 GPU에서 수치적으로 계산하고, 최종 시험 손실은 선택한 조합의 원래 표본으로 직접 구했습니다. COCO-Stuff의 36개 조합·개념 분류기를 CPU 직접 계산과 대조했을 때 검증 손실의 최대 절대 차이는 0.000000002 미만이었습니다.
