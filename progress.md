# Progress

> 기존 공개 문서의 수치와 설정을 옮긴 기록이다. 실험 구분과 이름은 임시이며 사람이 수정한다.
> 순서는 commit 순서이고, 날짜는 결과 문서 폴더 기준이다.

## 실험 현황 (한눈에 보기)
상태: 채택 / 보류 / 기각 / 진행중 / 확인 필요

| ID | 실험 | 상태 | 핵심 결과 | 이유 / 비고 | 날짜 |
|---|---|---|---|---|---:|
| E01 | 기본 3개, 4개 자료 검색 | 확인 필요 | 평균 R@5<br>Hungarian 21.3%<br>PLS-SVD 55.2%<br>CCA 54.2% | 확인 필요 | 11/07 |
| E02 | COCO val2017 기본 3개 | 확인 필요 | 개념 손실 / R@5<br>Hungarian 0.0761 / 12.2%<br>PLS-SVD 0.0729 / 36.4%<br>CCA 0.0715 / 35.9% | 확인 필요 | 11/07 |
| E03 | CCA + 공분산 회전 | 확인 필요 | 개념 손실 / R@5<br>회전 0.0696 / 35.9%<br>회전 후 top 64 0.0696 / 30.8% | 확인 필요 | 11/08 |
| E04 | 희소 PLS·CCA (k=16) | 확인 필요 | 개념 손실 / R@5<br>Sparse PLS-SVD 0.0706 / 22.9%<br>Sparse CCA 0.0708 / 25.6% | 확인 필요 | 11/08 |
| E05 | Sinkhorn·Procrustes (256쌍) | 확인 필요 | 개념 손실 / R@5<br>Sinkhorn 0.0761 / 12.1%<br>Procrustes 0.0761 / 12.2% | 확인 필요 | 11/08 |
| E06 | 공유 SAE (8192 좌표) | 확인 필요 | 개념 손실 / R@5<br>Group-Sparse 0.0681 / 13.0%<br>Iso-Energy 0.0643 / 12.9% | 확인 필요 | 11/08 |
| E07 | 다중 캡션 부분 관측 audit | 채택 | same-image +8.02%p<br>duplicate +0.00%p<br>cross-image −19.18%p | retrieval은 mapping 구조와 semantic overlap의 영향을 함께 받음 | 11/10 |

## 열린 질문
(사람이 추가한다)

## 날짜별 기록

### 2026-11-07 · E01 기본 3개, 4개 자료 검색
- [목적] 확인 필요
- [가설] 확인 필요
- [설정] COCO, Flickr30k, DOCCI, IIW-400, 공개 commit `c5d17fc`
- [결과] 캡션 잘림 비율 DOCCI 94.4%, IIW-400 99.0%. [표](https://github.com/JiH00nKw0n/mm-sae/blob/c5d17fc/docs/baselines-2026-11-07/README.md)
- [해석] 확인 필요
- [결정] 확인 필요
- [다음] 확인 필요

### 2026-11-07 · E02 COCO val2017 기본 3개
- [목적] 확인 필요
- [가설] 확인 필요
- [설정] CC3M 2,905,954쌍, rank 256, CCA λ=1, `configs/coco_val2017.yaml`, 공개 commit `37176f1`, provenance `87b5f13`
- [결과] [표](docs/baselines-2026-11-07/README.md)
- [해석] 확인 필요
- [결정] 확인 필요
- [다음] 확인 필요

### 2026-11-08 · E03 CCA + 공분산 회전
- [목적] 확인 필요
- [가설] 확인 필요
- [설정] `configs/mapping/cca_cov.yaml`, `configs/mapping/cca_cov64.yaml`, `configs/coco_val2017_top16.yaml`, 공개 commit `0a2b5ab`
- [결과] 회전 목적값 0.03465 → 0.38978, 종료 상태 `iteration_limit` (500회). [표](docs/baselines-2026-11-08/README.md)
- [해석] 확인 필요
- [결정] 확인 필요
- [다음] 확인 필요

### 2026-11-08 · E04 희소 PLS·CCA
- [목적] 확인 필요
- [가설] 확인 필요
- [설정] `configs/mapping/sparse_pls.yaml`, `configs/mapping/sparse_cca.yaml`, k=16, 공개 commit `e8c02e0`
- [결과] [표](docs/baselines-2026-11-08/README.md)
- [해석] 확인 필요
- [결정] 확인 필요
- [다음] 확인 필요

### 2026-11-08 · E05 Sinkhorn·Procrustes
- [목적] 확인 필요
- [가설] 확인 필요
- [설정] `configs/mapping/sinkhorn.yaml`, `configs/mapping/procrustes.yaml`, `feature_scope: matched_pairs`, Sinkhorn ε=0.05, 공개 commit `8967091`
- [결과] [표](docs/baselines-2026-11-08/README.md)
- [해석] 확인 필요
- [결정] 확인 필요
- [다음] 확인 필요

### 2026-11-08 · E06 공유 SAE
- [목적] 확인 필요
- [가설] 확인 필요
- [설정] `configs/group_sparse.yaml`, `configs/iso_align.yaml`, 8192 latent, top-k 32, 20,000쌍 제외, 공개 commit `28622e2`
- [결과] [표](docs/baselines-2026-11-08/README.md)
- [해석] 확인 필요
- [결정] 확인 필요
- [다음] 확인 필요

### 2026-11-10 · E07 다중 캡션 부분 관측 audit
- [목적] mapping 한계와 caption partial observation의 영향을 분리한다.
- [가설] 동일 이미지의 complementary caption에서만 retrieval과 margin이 개선되며, mapping 구조에 따라 추가 evidence 활용량이 다를 것이다.
- [설정] COCO val2017, 5,000×5,000 controlled retrieval, k=1/2/3/5, 10 seeds, 독립 SAE 기반 mapping 9개, pre/post-SAE aggregation, duplicate/cross-image controls, paired hierarchical bootstrap 2,000회.
- [결과] same-image k=1→5 평균 R@5 +8.02%p, duplicate +0.00%p, cross-image −19.18%p. 개선량은 dense +9.7~13.4%p, sparse +6.8~9.4%p, 1:1 +3.2~4.1%p였다. 모든 method×aggregation 조건의 95% CI가 0을 제외했다.
- [해석] partial observation은 공통 병목이지만 method gap을 제거하지 않는다. Retrieval은 mapping 구조와 관측된 semantic overlap을 함께 반영한다.
- [결정] 기존 mapping 확장과 observation-aware baseline을 동일 조건에서 비교한다. 성공 기준은 추가 캡션이 없는 k=1에서 concept loss를 유지하면서 retrieval을 개선하는 것이다. 확인: 사용자.
- [다음] Sparse CCA anchor drift 원인과 shared-SAE checkpoint를 확인한 뒤 k=1 Pareto 개선 여부를 평가한다.
