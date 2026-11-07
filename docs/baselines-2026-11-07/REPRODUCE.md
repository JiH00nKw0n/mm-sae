저장소를 복제하고 Docker GPU 이미지를 빌드합니다. [벤치마크 입력 안내](../benchmarks.md)에 따라 원본 자료를 배치합니다. 아래에서 `/datasets/mm-sae`는 평가 자료, `/cache/mm-sae/cc3m`은 기존 CC3M 임베딩 캐시, `/checkpoints/cc3m/models`는 이미지·텍스트 SAE가 들어 있는 경로를 뜻합니다.

```bash
git clone https://github.com/JiH00nKw0n/mm-sae.git
cd mm-sae
make build TARGET=gpu IMAGE=mm-sae:gpu
for method in hungarian pls cca; do
  make run TARGET=gpu IMAGE=mm-sae:gpu DATA=/datasets/mm-sae CACHE=/cache/mm-sae \
    RUNS=/results/mm-sae SAE=/checkpoints/cc3m/models \
    ARGS="run configs/paper.yaml --mapping configs/mapping/${method}.yaml"
done
```

위 명령은 저장된 SAE를 고정하고 대응 행렬을 학습한 뒤 평가합니다. `/results/mm-sae/paper/mappings/` 아래의 각 `fit.json`에 학습 데이터와 가중치의 식별값이 기록됩니다. 같은 폴더의 `evaluation` 아래에 데이터셋별 결과와 `summary.csv`가 저장됩니다.

개념 평가에서는 조합·개념별 로지스틱 분류기를 학습하고, 이미지 단위로 분리한 검증 자료에서 개념마다 손실이 가장 낮은 조합을 선택합니다. 최종 손실은 고정한 조합의 평가 손실을 개념별로 평균합니다. 학습에 양성·음성 각각 20개 이상, 검증에 양성·음성 각각 1개 이상이 있는 개념을 사용합니다. 편집 평가에서는 선택한 분류기를 그대로 사용합니다. 검색은 각 데이터셋의 전체 후보로 계산하며 한 이미지의 정답 캡션을 모두 인정합니다. 이미지 질의와 텍스트 질의를 따로 보고합니다.

서버에서는 기존 COCO 활성값의 이미지 순서와 정답 주석의 이미지 ID를 대조해 재사용했습니다. 분류기의 목적함수는 단일 조합의 이진 교차엔트로피입니다. 서버 계산은 각 축의 점수 분포에서 레이블과 무관한 합을 GPU에서 수치 적분하고, 기울기가 급한 경우 원래 표본에서 직접 계산합니다. 최종 평가 손실은 원래 표본별 점수에서 직접 구합니다.

표와 그림은 세 방법의 결과 폴더를 읽어 다시 만듭니다. 저장소 루트에서 다음 명령을 실행합니다. 각 방법마다 동일한 데이터 분할의 실행 하나를 넣습니다.

```bash
python scripts/report_baselines.py /results/mm-sae/paper docs/baselines-2026-11-07
```

이 실행에서 CC3M 캐시 식별값은 `b7850d49907b85280057c956fb2cab391edbf64a56d21df956cc9fbf2d5d0920`입니다. 입력 인코더는 `openai/clip-vit-base-patch32`의 버전 `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`을 사용했습니다. 입력 한도는 특수 토큰을 포함한 77개입니다. COCO-Stuff에서 세 방법의 36개 조합·개념 분류기를 저장소의 직접 계산 방식으로 대조했습니다. 검증 손실의 최대 절대 차이는 0.000000002 미만이었습니다.

개념 혼합도와 표현 분산도는 검증 손실이 학습 자료의 양성 비율만 사용하는 상수 분류기에 비해 얼마나 줄었는지로 계산합니다. 손실 감소량이 0보다 작으면 중요도를 0으로 둡니다. 한 조합의 중요도가 여러 개념에 퍼질수록 혼합도가 커집니다. 한 개념의 중요도가 여러 조합에 퍼질수록 분산도가 커집니다. 편집 전후 비교 표에서 제거한 개념은 확률의 감소, 추가한 개념은 확률의 증가, 유지한 개념은 작은 절대 변화를 기대합니다. CUB 원본 평가는 synCUB에 적용할 분류기를 선택하는 데 사용합니다.

실제 평가 자료의 출처는 다음과 같습니다.

| 자료 | 이 실행에서 사용한 입력 |
|---|---|
| COCO와 COCO-Stuff | COCO 2017의 instance 주석과 전체 이미지의 COCO-Stuff 164K 마스크를 사용했습니다. COCO 검색에서는 val2017의 이미지 5,000장과 캡션 25,014개를 사용했습니다. |
| Flickr30k | 공개 검색 시험 분할의 이미지 1,000장과 캡션 5,000개를 사용했습니다. [CLIP Benchmark의 시험 이미지](https://huggingface.co/datasets/clip-benchmark/wds_flickr30k)와 [원본 파일명·분할·캡션 목록](https://huggingface.co/datasets/nlphuji/flickr30k)을 캡션 집합으로 대조했습니다. |
| DOCCI | 공식 시험 설명과 이미지 ID를 사용했습니다. [이미지 배포본](https://huggingface.co/datasets/not-lain/docci)의 각 파일명과 설명을 [DOCCI 공식 자료](https://google.github.io/docci/)의 시험 ID 및 설명과 대조했습니다. |
| IIW-400 | [공식 IIW-400 주석](https://github.com/google/imageinwords/tree/main/datasets/IIW-400)의 사람 작성 설명과 해당 ID의 DOCCI-AAR 이미지를 사용했습니다. |
| Visual Genome | [공식 영역 그래프와 이미지 메타데이터](https://visualgenome.org/api/v0/api_home)를 사용합니다. 각 영역 문장에 명시한 주석만 양성으로 처리하고, 같은 이미지의 모든 문장을 같은 분할에 둡니다. 동일한 문장은 한 번 인코딩한 뒤 원래 표본 순서와 주석을 복구해 평가합니다. |
| CUB | CUB-200-2011의 원래 학습·시험 분할과 이미지별 속성 유무를 사용했습니다. [속성 주석 배포본](https://huggingface.co/datasets/Jaszczurix7652/CUB-200-2011)과 [이미지 배포본](https://huggingface.co/datasets/bentrevett/caltech-ucsd-birds-200-2011)을 파일명으로 연결했습니다. |
| synCOCO와 synCUB | [synCOCO v1.0](https://huggingface.co/datasets/jokl/syncoco/tree/v1.0)과 [synCUB v1.0](https://huggingface.co/datasets/jokl/syncub/tree/v1.0)의 편집 전후 이미지 및 개념 변화 주석을 사용합니다. |

[실행 식별값](provenance.json)에는 코드 커밋, 양쪽 SAE 가중치와 설정의 SHA-256, 세 대응 행렬의 SHA-256, 데이터 배포 버전을 기록했습니다. 그림에서 사용하는 평균은 연구자가 정한 요약 방식입니다. 데이터셋별 비교와 두 검색 방향의 차이는 본문의 세부 표에서 확인할 수 있습니다.
