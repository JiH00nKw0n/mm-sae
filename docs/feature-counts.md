# 선택하는 특징 수를 늘린 분류기 비교

기존 이미지 표현 진단에서 분류기에 사용할 수 있는 SAE 좌표 수만 늘린다. 선택 기준별로 이미 계산한 범주별 특징 순위를 고정하고 상위 8개, 16개, 64개를 선택한다. 상위 좌표 집합은 서로 포함되는 관계다. 기존 5개, 전체 SAE 특징, 원본 임베딩의 분류기는 입력 파일의 해시와 분할·규제 설정이 일치할 때 재사용한다.

이미지 인코더와 SAE를 다시 학습하지 않는다. SAE의 전체 좌표는 4,096개이며 한 이미지에서 활성화되는 좌표는 계속 최대 8개다. 선택한 64개는 여러 이미지에 걸쳐 사용할 수 있는 고정 좌표의 목록을 뜻한다.

원본 이미지 94,630장에서 로지스틱 선형 분류기를 학습하고 23,657장에서 규제 강도를 조정한 뒤 val 이미지 5,000장에서 평가한다. 기존과 같은 표준화, 규제 후보, 계수 부호 허용, 이미지 단위 재표집을 사용한다. 특징 번호를 검증 AUROC로 다시 고르지 않는다.

다음 세 평가를 각각 보고한다.

1. 원본 이미지의 범주 유무를 학습하고 원본 검증 이미지에서 범주 유무를 평가한다.
2. 해당 범주가 있는 원본과 해당 범주를 가린 이미지를 구별하도록 학습하고 같은 구별을 검증한다.
3. 원본 이미지의 범주 유무를 학습한 분류기를 고정하고 원본·가림본 구별 AUROC와 이미지별 점수 감소 비율을 측정한다.

171개 전체 범주를 학습한다. 기존 상관행렬과 비교하는 주 그림과 표에서는 텍스트 대표가 없었던 ceiling-tile을 제외한 물체 80개와 배경 90개를 사용한다. 배경 91개 전체 요약도 CSV에 별도로 남긴다. 평균, 표준편차, 최솟값, 사분위수, 최댓값을 저장하고 상자 그림을 만든다.

설정 파일은 `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/configs/feature-counts-server.yaml`이며 `feature_counts`와 `objectives`로 실행 범위를 조절한다. 서버에서는 다음과 같이 실행한다.

```bash
cd /mnt/working/mm-sae
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m experiments.representation_diagnosis.feature_counts \
  --config configs/feature-counts-server.yaml
```

진행 정보는 `/mnt/working/mm-sae/runs/feature-counts-2026-09-29/progress.json`에 10초마다 저장한다. 범주·특징 수·분류 목적별 완료 파일을 저장해 중단 후 이어서 실행할 수 있다. 다음 명령으로 진행 상태와 남은 예상 시간을 확인한다.

```bash
/mnt/working/mm-sae/.venv/bin/python -m mm_sae \
  --run-dir /mnt/working/mm-sae/runs/feature-counts-2026-09-29 status
```

범주 유무를 잘 예측해도 해당 배경 자체 대신 함께 등장한 객체에 의존할 수 있다. 가림 구별을 직접 학습한 분류기는 흰색 편집 흔적에 반응할 수 있다. 각 평가가 증명하는 범위를 구분해서 해석한다.

## 네 가지 선택 기준을 비교한다

원본·가림 AUROC, 같은 이미지의 평균 활성값 감소량, 단일 좌표의 범주 분류 손실, 원본 임베딩의 개념 분류기에 대한 좌표별 기여도를 각각 사용한다. 각 기준에서 이미 만들어 둔 순위를 다시 계산하지 않고 고정한다. 네 기준 모두 동일한 특징 수와 분류 평가를 적용한다.

기준별 설정은 `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/configs/feature-counts-server.yaml`, `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/configs/feature-counts-paired_mean_drop-server.yaml`, `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/configs/feature-counts-single_logistic-server.yaml`, `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/configs/feature-counts-probe_attribution-server.yaml`에 저장했다.

전체 실행을 마친 뒤 `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/scripts/compare_feature_counts.py`에 `--config /Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/configs/feature-count-comparison.yaml`을 전달하면 네 기준의 비교 표와 그림을 만든다.
