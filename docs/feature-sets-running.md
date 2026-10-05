# 여러 특징을 사용하는 후속 실험의 실행과 조회

세 실험은 기존 SAE와 원본·가림 활성값을 재사용한다. SAE를 다시 학습하지 않는다. 범주 점수를 만드는 선형 분류기와 활성값을 예측하는 작은 회귀식만 학습한다. 실험 방법은 `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/docs/feature-set-suite-plan.md`에 고정했다.

서버에서는 다음 명령으로 실행한다. 설정의 `suites`에서 `original_rq2`, `refined_rq2`, `multi_feature_rq1` 중 실행할 묶음을 선택한다. 특징 수, 규제 강도 후보, 재표집 횟수, 작업자 수, 입력·출력 경로도 설정에 있다.

```bash
cd /mnt/working/mm-sae
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m mm_sae --config configs/feature-sets-server.yaml all
```

진행률과 현재 단계의 ETA는 다음 명령으로 확인한다. 상태 파일은 10초마다 갱신한다. 비용이 다른 작업의 평균 속도로 계산한 ETA이므로 남은 범주의 크기에 따라 바뀔 수 있다. 아직 실행하지 않은 단계의 시간을 측정한 것처럼 표시하지 않는다.

```bash
cd /mnt/working/mm-sae
PYTHONPATH=src .venv/bin/python -m mm_sae status --run-dir runs/feature-set-suite-server-2026-09-28
```

같은 명령을 다시 실행하면 완료한 단계와 범주별 작업을 재사용한다. 코드나 설정이 바뀌면 새 출력 폴더를 사용해야 한다. 원본 캐시가 바뀌면 입력 파일 해시 검사에서 중단한다. 캐시가 없을 때 자동으로 이미지 추론이나 SAE 학습을 시작하지 않는다.

| 서버 경로 | 저장한 내용 |
|---|---|
| `/mnt/working/mm-sae/runs/feature-set-suite-server-2026-09-28/progress.json` | 현재 단계, 완료 작업 수, 관측 속도와 ETA를 저장한다. |
| `/mnt/working/mm-sae/runs/feature-set-suite-server-2026-09-28/splits.json` | 학습용·조정용·평가용 이미지 ID를 저장한다. |
| `/mnt/working/mm-sae/runs/feature-set-suite-server-2026-09-28/selection` | 학습용 원본·가림 AUROC로 정한 특징 순위를 저장한다. |
| `/mnt/working/mm-sae/runs/feature-set-suite-server-2026-09-28/concepts` | 특징 결합 계수와 범주별 원본·가림 및 개념 검출 AUROC를 저장한다. |
| `/mnt/working/mm-sae/runs/feature-set-suite-server-2026-09-28/raw_prediction` | 원래 RQ2의 활성값 예측 결과와 회귀 계수를 저장한다. |
| `/mnt/working/mm-sae/runs/feature-set-suite-server-2026-09-28/shared_pairs` | 같은 1위 특징을 공유하는 범주 쌍의 직접 구별 결과를 저장한다. |
| `/mnt/working/mm-sae/runs/feature-set-suite-server-2026-09-28/intervention` | 공통 좌표에서 학습한 연결과 가림 반응 예측 결과를 저장한다. |
| `/mnt/working/mm-sae/runs/feature-set-suite-server-2026-09-28/correspondence` | 확장 RQ1의 방향별 전체 범주 쌍, 상관 구간별 분포와 대각 초과 비율을 저장한다. |
| `/mnt/working/mm-sae/runs/feature-set-suite-server-2026-09-28/report` | 그림 8개의 PNG·PDF·SVG, 요약 문서와 전체 CSV를 저장한다. |

완료한 결과에서는 모든 특징 개수에서 평가할 수 있는 같은 범주와 범주 쌍을 비교하는 보조 보고서도 만들 수 있다. 모델을 다시 학습하지 않는다. 원래 보고서의 전체 171개 범주에 대한 분석과 계산 불가능 항목은 그대로 남는다.

```bash
cd /mnt/working/mm-sae
PYTHONPATH=src:. .venv/bin/python scripts/summarize_feature_sets.py runs/feature-set-suite-server-2026-09-28
```

이 보조 보고서는 `/mnt/working/mm-sae/runs/feature-set-suite-server-2026-09-28/comparison`에 저장한다. macOS에서 실행할 때는 `--font AppleGothic`을 추가할 수 있다.

Docker에서도 같은 실행기를 사용한다. CPU 이미지의 기본 작업 경로는 `/workspace`다. 아래 명령은 저장된 캐시를 읽는 후속 분석용이며 원본 이미지 데이터 전체를 다시 내려받지 않는다. 설정의 글꼴은 컨테이너에 설치한 글꼴 이름으로 지정해야 한다.

```bash
docker build --target cpu -t mm-sae-analysis .
docker run --rm -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
  -v /mnt/working/mm-sae/runs:/workspace/runs \
  -v /mnt/working/mm-sae/configs:/workspace/configs:ro \
  mm-sae-analysis --config /workspace/configs/feature-sets-server.yaml all
```

이번 환경에서는 서버의 Python 가상환경으로 실행했다. Docker 이미지를 새로 빌드하거나 이 Docker 명령을 실행한 것은 아니다.
