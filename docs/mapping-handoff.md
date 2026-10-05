# COCO 대응 실험 실행 안내

최근 COCO2017 실험은 이 저장소의 `main` 브랜치에 있다. 이전 논문의 `cross_modal_feature_heterogeneity` 저장소에도 같은 이름의 비교 방법이 있지만, 그 저장소의 `post-rebuttal` 기본 검색 실험은 CC3M으로 SAE를 학습하고 COCO Karpathy test에서 평가한다. 두 실험을 같은 데이터 조건으로 부르면 안 된다.

| 비교할 내용 | 현재 저장소의 COCO2017 실험 | 이전 논문의 post-rebuttal 검색 실험 |
|---|---|---|
| 저장소와 브랜치 | `JiH00nKw0n/mm-sae`의 `main`을 사용한다. | `JiH00nKw0n/cross_modal_feature_heterogeneity`의 `post-rebuttal`을 사용한다. |
| SAE 학습 자료 | COCO2017 train 이미지 118,287장과 캡션 591,753개를 사용한다. | CC3M train을 사용한다. |
| SAE 설정 | 모달리티마다 특징 4,096개 중 최대 8개를 활성화하며 30회 학습한다. | 모달리티마다 특징 4,096개 중 최대 32개를 활성화하며 10회 학습한다. |
| 대응 학습 자료 | COCO2017 train 이미지를 학습 94,630장과 설정 선택 23,657장으로 나눈다. | CC3M train을 사용한다. CCA와 Procrustes에는 최대 300,000쌍을 사용한다. |
| 최종 검색 평가 | 공식 COCO2017 val 이미지 5,000장과 캡션 25,014개를 사용한다. | COCO Karpathy test 이미지 5,000장을 사용한다. |
| 검색 지표 | 이미지로 캡션을 찾는 방향과 캡션으로 이미지를 찾는 방향에서 Recall@1·5·10을 계산한다. | 동일한 두 방향에서 Recall@1·5·10을 계산한다. |

Recall@1·5·10은 검색 결과 상위 1·5·10개 안에 정답이 포함된 질의의 비율이다. 이미지 질의는 정답 캡션 중 하나라도 포함되면 성공으로 계산한다.

## 현재 COCO2017 실험을 실행한다

서버의 저장소 경로는 `/mnt/working/mm-sae`다. 다음 명령으로 새 코드를 받고, 필요한 저장 파일이 있는지 확인한다. `--check`는 실험을 실행하거나 결과 파일을 만들지 않는다.

```bash
cd /mnt/working/mm-sae
git pull --ff-only origin main
bash /mnt/working/mm-sae/scripts/run_coco_mapping.sh \
  /mnt/working/mm-sae/runs/coco-mapping-handoff --check
```

확인을 마친 뒤 같은 명령에서 `--check`를 빼면 실험을 실행한다.

```bash
bash /mnt/working/mm-sae/scripts/run_coco_mapping.sh \
  /mnt/working/mm-sae/runs/coco-mapping-handoff
```

이 명령은 저장된 활성값을 사용한다. CLIP과 SAE를 다시 학습하지 않으며, 이번 인수인계용 실행에서는 객체 제거 평가도 생략한다. 헝가리안, Greedy, 상위 특징 여러 개를 연결하는 방법, Sinkhorn, CCA, Procrustes와 기존 다대다 후보들을 모두 계산한다. 기본 비교 방법만 따로 실행하는 명령은 아니다.

첫 단계는 `/mnt/working/mm-sae/configs/mapping-server.yaml`의 설정을 사용하고, 두 번째 단계는 `/mnt/working/mm-sae/configs/mapping-ablation-server.yaml`의 설정을 사용한다. 실행 파일은 두 설정의 입력·출력 경로를 맞추고 첫 단계에 `skip_removal: true`를 적용한 설정 사본을 결과 폴더에 저장한다. 대응 학습과 검색 설정은 그대로 유지한다.

| 순서 | 실행하는 모듈 | 확인할 내용 |
|---|---|---|
| 첫 번째 단계 | `/mnt/working/mm-sae/experiments/mapping_suite/run.py`를 실행한다. | 대응 방법을 학습하고 설정을 선택한 뒤 검색을 평가한다. CCA와 Procrustes는 이 모듈에서 별도로 계산한다. |
| 두 번째 단계 | `/mnt/working/mm-sae/experiments/mapping_ablation/run.py`를 실행한다. | 학습한 대응을 고정하고 원래 활성값, 학습 평균을 뺀 활성값, 평균 제거와 표준편차 보정을 모두 적용한 활성값을 비교한다. 공통 공간과 특징 집합 비교도 포함한다. |

**모든 방법에 같은 입력 보정을 적용한 비교는 두 번째 보고서를 확인해야 한다.** 첫 번째 보고서에서는 헝가리안·Sinkhorn 등의 검색이 원래 활성값을 사용하고, CCA·Procrustes는 평균과 표준편차를 보정한 활성값을 사용한다. 두 번째 보고서의 `standardized` 조건은 헝가리안·Sinkhorn 등에도 같은 평균·표준편차 보정을 적용한다.

헝가리안·Sinkhorn 등은 이미지 특징 공간과 텍스트 특징 공간에서 각각 검색한다. 각 공간에서 두 검색 방향을 모두 계산하므로, 검색 공간과 검색 방향을 구분해서 비교한다. CCA와 Procrustes는 별도로 학습한 공통 공간에서 평가한다.

새 설정을 시험하려면 새 출력 폴더를 지정한다. 기존 출력 폴더는 입력·설정·계산 코드의 지문이 같을 때만 이어 실행할 수 있다. 과거 결과 폴더에 새 코드를 그대로 이어 실행하면 재사용 검사가 거절할 수 있다.

## 새 서버에는 저장된 활성값도 전달한다

**GitHub를 클론하는 것만으로는 이 실험을 실행할 수 없다.** 데이터, 모델 가중치, 활성값과 결과 폴더는 Git에서 제외한다. 기존 서버에는 `/mnt/working/mm-sae/runs/elice-rq1-lexicon2`에 필요한 파일이 있다. 다른 서버에서는 이 폴더의 아래 파일을 같은 구조로 복사해야 한다.

```text
/mnt/working/mm-sae/runs/elice-rq1-lexicon2/
  dataset.json
  models/frozen.json
  activations/train2017/image.npz
  activations/train2017/text.npz
  activations/val2017/image.npz
  activations/val2017/text.npz
  index/train2017/images.json
  index/train2017/concept_ids.json
  index/train2017/parents.npy
  index/train2017/presence.npy
  index/train2017/mentions.npy
  index/val2017/images.json
  index/val2017/concept_ids.json
  index/val2017/parents.npy
  index/val2017/presence.npy
  index/val2017/mentions.npy
```

이 실행에는 원본 이미지나 가림 결과가 필요하지 않다. 저장된 활성값이 없는 상태에서 SAE 학습부터 시작하는 명령은 아니며, 파일이 빠져 있으면 중단한다. 입력 경로가 다르면 `MM_SAE_SOURCE_RUN` 환경변수에 저장된 활성값 폴더의 절대 경로를 지정한다.

새 서버에 Python 환경도 없다면, `/mnt/working/mm-sae`에 저장소를 내려받은 뒤 다음 명령으로 설치한다. Python 3.11과 `uv`가 설치되어 있어야 한다.

```bash
cd /mnt/working/mm-sae
uv sync --frozen --extra dev
```

실행 파일은 기본적으로 `/mnt/working/mm-sae/.venv/bin/python`을 사용한다. 다른 Python 환경을 사용하려면 `MM_SAE_PYTHON` 환경변수에 실행 파일의 절대 경로를 지정한다. 기본 실험 설정은 CUDA GPU를 사용한다.

## 결과와 진행 상황을 확인한다

위 명령의 결과 파일은 다음 위치에 생긴다.

- `/mnt/working/mm-sae/runs/coco-mapping-handoff/suite/report.html`에는 첫 번째 대응 비교 결과가 나온다.
- `/mnt/working/mm-sae/runs/coco-mapping-handoff/ablation/report.html`에는 입력 보정과 공통 공간 비교 결과가 나온다.
- 각 결과 폴더의 `retrieval_summary.csv`에는 양방향 Recall@1·5·10이 나온다.
- 각 결과 폴더의 `progress.json`에는 현재 단계, 처리량과 해당 작업의 예상 남은 시간이 저장된다.

```bash
/mnt/working/mm-sae/.venv/bin/python -m mm_sae \
  --run-dir /mnt/working/mm-sae/runs/coco-mapping-handoff/suite status
/mnt/working/mm-sae/.venv/bin/python -m mm_sae \
  --run-dir /mnt/working/mm-sae/runs/coco-mapping-handoff/ablation status
```

## 이전 논문의 post-rebuttal 실험을 실행한다

이전 논문의 실행 파일은 [run_post_rebuttal.sh](https://github.com/JiH00nKw0n/cross_modal_feature_heterogeneity/blob/post-rebuttal/scripts/run_post_rebuttal.sh)이며, 검색 비교 구현은 [alignment_methods.py](https://github.com/JiH00nKw0n/cross_modal_feature_heterogeneity/blob/post-rebuttal/src/rebuttal/alignment_methods.py)에 있다. 다음은 저장소를 `/mnt/working/cross_modal_feature_heterogeneity`에 내려받은 경우다. 이 저장소의 README에 따라 환경과 데이터를 먼저 준비해야 한다.

```bash
cd /mnt/working/cross_modal_feature_heterogeneity
source /mnt/working/cross_modal_feature_heterogeneity/.venv/bin/activate
bash /mnt/working/cross_modal_feature_heterogeneity/scripts/run_post_rebuttal.sh all
```

임베딩과 SAE 학습 결과가 이미 준비돼 있으면 추가 분석 단계만 실행할 수 있다.

```bash
cd /mnt/working/cross_modal_feature_heterogeneity
.venv/bin/python run.py configs/post_rebuttal/clip_b32.yaml --stage rebuttal
```

`rebuttal` 단계에는 검색 비교 외의 분석도 들어 있다. 헝가리안·Sinkhorn·Procrustes·CCA 비교는 그중 `alignment_methods` 분석이며, 기본적으로 CC3M 학습 조건인 `cc3m_k32`에만 등록되어 있다. COCO를 학습하는 설정 파일도 있지만, 그것을 실행한다고 이 검색 비교가 COCO2017 조건으로 실행되는 것은 아니다.

이전 저장소에는 `/mnt/working/cross_modal_feature_heterogeneity/cache/clip_b32_cc3m`, `/mnt/working/cross_modal_feature_heterogeneity/cache/clip_b32_coco`, `/mnt/working/cross_modal_feature_heterogeneity/outputs/post_rebuttal/cc3m_clip_b32/seed0/separated/final`, `/mnt/working/cross_modal_feature_heterogeneity/outputs/post_rebuttal/cc3m_clip_b32/seed0/ours/panel.npz`가 필요하다. 캐시 형식이 현재 저장소와 다르므로 입력 경로만 바꿔 공유할 수는 없다.

이전 구현은 CCA·Procrustes의 공분산 학습에는 중심화를 적용하지만 검색 활성값에는 같은 평균·표준편차 보정을 적용하지 않는다. 이번 저장소의 동일 보정 비교와 별도 실험으로 취급한다. 이전 검색 결과는 `/mnt/working/cross_modal_feature_heterogeneity/outputs/post_rebuttal/rebuttal/cc3m_k32/alignment_methods.json`과 같은 경로의 `alignment_methods.md`에 저장된다.
