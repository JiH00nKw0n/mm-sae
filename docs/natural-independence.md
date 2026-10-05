# 원본 분포의 양의 동시 등장 관계 제거

이 분석은 원본의 범주별 등장 비율을 유지하며 양의 동시 등장 상관만 0으로 만든다. 모든 이미지 범주에 대해 다른 모든 텍스트 범주를 비교하고, 자기 범주보다 점수가 높은 상대가 하나 이상 있는 이미지 범주의 비율을 계산한다. 개별 방향별 범주 쌍을 분모로 한 비율도 따로 계산한다.

원본 계산은 전체 이미지·캡션 관측을 사용하며 캡션마다 같은 비중을 준다. 이미지와 캡션 주석의 일치 여부로 자료를 거르지 않는다. 이미지 주석도 해당 캡션에 연결된 이미지의 존재 여부로 정렬한다. 모델, 활성값, 대표 특징 번호는 고정한다.

두 범주 A와 B의 원래 등장 비중을 p와 q라고 한다. 주석 상관이 양수이면 네 집단의 비중을 둘 다 없음에 (1-p)(1-q), B만 있음에 (1-p)q, A만 있음에 p(1-q), 둘 다 있음에 pq로 설정한다. 집단 내부에서는 캡션별 비중을 같게 준다. 이 분포에서 A의 이미지 특징과 A의 텍스트 특징 사이의 점수, A의 이미지 특징과 B의 텍스트 특징 사이의 점수를 모두 다시 계산한다. 원래 주석 상관이 양수가 아닌 쌍은 두 점수를 모두 원본 그대로 유지한다.

양의 상관을 제거하는 분포는 A·B 쌍마다 다르다. 따라서 전체 범주의 비율은 각각 따로 계산한 비교 중 하나라도 같은 범주 점수를 넘는지 센 결과다. 모든 범주의 관계를 동시에 제거한 하나의 자료 분포나 전역 매칭 결과가 아니다. 캡션 언급 비율까지 보존하지는 않으며 그 변화는 점수 파일에 기록한다.

주석 상관 구간은 원본에서 0.2 간격으로 정하고 제거 후에도 같은 구간에 둔다. 구간별 범주 비율은 그 구간에 비교 상대가 있는 이미지 범주를 분모로 한다. 구간별 쌍 비율은 그 구간에 속하는 방향별 범주 쌍을 분모로 한다. 두 지표를 서로 바꾸어 해석하지 않는다. 계산할 수 없는 목표 분포는 관측을 만들어 채우지 않고 별도로 표시한다.

```bash
cd /Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae
.venv/bin/python scripts/compare_natural_independence.py \
  --run runs/elice-rq1-lexicon2 \
  --out runs/elice-rq1-lexicon2-comparison/natural-independence
```

`--run`과 `--out`으로 입력·출력 경로를 지정한다. `--concept-ids`를 생략하면 모든 범주를 계산한다. `--report-only`를 추가하면 저장된 점수로 최종 집계와 그림만 다시 만든다. 원본 활성값을 사용하므로 GPU나 새 모델 추론이 필요하지 않다. `progress.json`에 처리한 조합 수와 남은 시간 추정치를 기록한다.

최종 결과는 출력 폴더의 `all_category_summary.json`, `all_category_anchors.csv`, `all_category_comparisons.csv`, `original_annotation_bins.csv`에 저장한다. `scores.csv`와 `summary.json`에는 음수도 포함해 독립 분포를 계산한 중간 진단 결과가 있으므로 최종 지표와 구분한다. 그림은 `all_categories_comparison.png`와 `original_annotation_bins.png`이며 PDF와 SVG도 저장한다.

로컬 결과와 수치 검증 기록은 `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/runs/elice-rq1-lexicon2-comparison/natural-independence/README.md`에 있다. 계산 코드는 `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/scripts/compare_natural_independence.py`에 있고, 재사용하는 가중 상관 계산은 `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/src/mm_sae/metrics/reweighting.py`에 있다.
