# 자료와 주석

이 저장소는 공식 COCO2017 이미지·캡션과 COCO-Stuff의 픽셀 범주 주석을 이미지 식별자로 결합한다. 기본 학습 분할은 이미지 118,287장의 `train2017`, 검증 분할은 이미지 5,000장의 `val2017`이다. COCO-Stuff의 객체 80개와 배경 91개를 지원하지만 각 범주에 실제로 선택 가능한 텍스트 대표가 존재하는지는 별도로 확인한다. [COCO 자료](https://cocodataset.org/#download), [COCO-Stuff 설명](https://github.com/nightrome/cocostuff)

이미지 주석은 같은 이미지 크기의 1채널 정수 PNG다. 한 픽셀의 값은 그 위치의 범주 번호이며 객체 인스턴스 번호가 아니다. 같은 종류의 객체가 여러 개 있으면 같은 번호를 갖는다. 가림은 해당 번호의 모든 픽셀을 흰색으로 채운다. 영역의 사각형 전체를 지우지는 않는다.

기본 범주 번호는 공개 `stuffthingmaps_trainval2017.zip`의 PNG 원시 값이다. 예를 들어 사람은 0, 곰은 22, 잔디는 123이며 255는 주석에서 제외된 영역이다. 공식 범주 번호가 1부터 시작하는 목록과는 1만큼 차이가 난다. 중간에 사용하지 않는 번호가 있으므로 171개 범주를 임의로 연속 번호로 다시 붙이지 않는다. 기본 대응표는 `/workspace/src/mm_sae/resources/concepts.yaml`에 포함했다.

Hugging Face의 COCO-Stuff 자료를 확인할 수 있지만, 화면 표시를 위해 변환한 JPEG를 정수 주석으로 읽으면 안 된다. JPEG 압축은 픽셀 값을 바꾼다. 이 코드는 원본 PNG가 들어 있는 압축 파일의 고정 버전을 사용하고 파일 형식과 범주 번호를 검사한다. [Hugging Face 원본 압축 파일](https://huggingface.co/datasets/JPShi/COCO-Stuff/blob/b7af13b9d74c9ab4a6e0ffa787a40f0ab8ec1a40/stuffthingmaps_trainval2017.zip), [Hugging Face 데이터셋 설명](https://huggingface.co/datasets/GATE-engine/COCO-Stuff-164K)

자료가 이미 있다면 `data.download: false`로 설정하고 아래 구조를 연결한다. 상대 경로 패턴은 설정에서 바꿀 수 있다. 작은 예시의 식별자는 경로 형식을 설명하기 위한 것이며 파일을 직접 만들라는 뜻은 아니다.

```text
/workspace/data/coco/
  images/train2017/000000000009.jpg
  images/val2017/000000000139.jpg
  annotations/train2017/000000000009.png
  annotations/val2017/000000000139.png
  annotations/captions_train2017.json
  annotations/captions_val2017.json
```

`data.download: true`이면 필요한 파일을 준비한다. 전체 설정은 원본 압축 파일을 내려받아 푼다. 소규모 설정은 이미지 식별자 순으로 앞의 N장을 선택하며 모든 분할에 수량 제한을 요구한다. 이 제한은 데이터셋 대표성을 확보하기 위한 표집이 아니라 소규모 실행을 확실히 제한하기 위한 것이다. 캡션 JSON에는 전체 분할의 메타데이터가 들어 있지만, 선택된 이미지의 캡션만 모델 입력으로 사용한다.

**문장 객체 표지는 기본적으로 자동 규칙이며 별도 검토가 필요하다.** 범주명과 동의어의 단어 경계로 원문 표현의 시작·끝 위치를 찾는다. `hot dog` 안의 `dog`처럼 겹치는 표현은 긴 범주 표현을 우선한다. 서로 다른 범주의 표현을 분리할 수 없으면 해당 가림을 사용 불가로 기록한다. 자동 주석을 사람이 확인한 의미 정답으로 간주하지 않는다.

**2026년 9월 14일 승인한 전체 RQ1 실행은 사전 기반 주석을 사용한다.** 캡션 주석용 API는 호출하지 않는다. 실행 전에 `buses`, `benches`, `knives`, `shelves` 등의 복수형과 `stone floor`, `brick wall` 같은 어순을 바로잡았다. `puppy`, `kitten`, `sofa`, `television`, `street` 등 명시한 동의어를 추가했다. 사전 전체를 Git으로 기록하고 실행의 코드 지문에 포함한다. 결과를 본 뒤 실행 중인 사전을 바꾸지 않는다.

사전 규칙은 등록된 표현의 존재를 표시한다. 대명사의 지시 대상, 부정·가정 문맥과 일반 명사의 세부 범주를 판정하지 않는다. `luggage`를 임의로 가방이나 여행 가방에 배정하지 않고, `bat`를 문맥 없이 야구 방망이에 배정하지 않는다. 미검출을 실제 의미의 부재와 동일하게 해석하지 않는다. 의미가 넓은 배경 범주와 표현이 겹치는 잔여 범주에는 주석 오류나 대표 부재가 생길 수 있다. 사람이 검토한 주석과의 별도 검증 없이 전체 의미 대응 정확도로 보고하지 않는다.

**텍스트 가림은 객체 표현의 토큰을 unknown token으로 바꾼다.** unknown token은 토크나이저가 알 수 없는 입력을 나타내도록 지정한 기존 토큰이다. `features.text_masking: unk_token`을 사용하며 새 토큰을 추가하지 않는다. 단어 하나가 여러 토큰으로 나뉘면 그 토큰을 각각 하나씩 바꾸고, 같은 객체가 여러 번 언급되면 그 위치를 모두 바꾼다. 문장 길이와 다른 토큰의 번호·위치, attention mask는 유지한다. attention mask는 실제 문장과 길이를 맞추기 위한 빈 자리를 구별하는 값이며, 가린 토큰을 이 값으로 숨기지는 않는다.

원문은 기존 `CLIPProcessor`로 토큰화한다. 같은 모델 버전의 `CLIPTokenizerFast`가 제공하는 문자 위치로 가릴 토큰을 찾고, 두 토크나이저의 원본 토큰 번호가 정확히 같은지 검사한다. 번호가 다르거나 입력 길이 제한으로 목표 표현이 잘린 경우, 한 토큰에 다른 표현이 함께 들어 있는 경우에는 해당 가림을 사용 불가로 기록한다.

현재 CLIP의 unknown token은 `<|endoftext|>`이며 문장 종료 토큰과 같은 번호를 쓴다. 가린 위치에서 문장 벡터를 읽는 오류를 막기 위해 원본 입력의 종료 위치를 저장한다. 가림 후에는 Hugging Face 텍스트 모델의 마지막 층에서 원래 종료 위치의 벡터를 읽고 기존 투영층을 적용한다. 실제 토큰 번호와 처리 설정은 `/workspace/runs/coco-rq1/dataset.json`에 저장한다. [Hugging Face CLIP 문서](https://huggingface.co/docs/transformers/v4.50.0/en/model_doc/clip)

각 분할의 검토 목록은 `/workspace/runs/coco-rq1/index/train2017/caption_mask_review.csv`와 `/workspace/runs/coco-rq1/index/val2017/caption_mask_review.csv`에 저장한다. 원문, 원문 문자 범위, 가릴 토큰 위치, 토큰 수와 사용 불가 이유를 확인할 수 있다. 입력 길이 때문에 가릴 수 없는 경우에도 원문에 객체가 언급됐다는 주석을 부재로 바꾸지 않는다. COCO-Stuff 자체가 텍스트 의미 정답을 제공하지는 않는다.

검토한 문장을 JSONL 파일로 입력하면 해당 캡션의 자동 표지와 문자 범위를 대체한다. JSONL은 한 줄마다 JSON 객체를 하나 저장한 형식이다. 다음은 실제 COCO 캡션을 인용한 것이 아닌 입력 형식 예시다.

```json
{"caption_id": 123, "original": "A person is beside a bicycle.", "concept_ids": [0, 1], "spans": {"0": [{"text": "person", "start": 2, "end": 8}], "1": [{"text": "bicycle", "start": 21, "end": 28}]}}
```

`caption_id`와 `original`은 실제 식별자와 원문이어야 한다. `concept_ids`에는 원문에서 언급한 범주의 PNG 번호를 넣는다. `spans`의 각 키는 가릴 개념이고 값은 원문 표현과 문자 위치의 목록이다. 시작 위치는 포함하고 끝 위치는 포함하지 않는다. 개념이 언급됐지만 다른 개념과 분리해 가릴 수 없다면 해당 키를 생략한다. 삭제하거나 다시 작성한 문장을 담은 예전 `edits` 형식은 오류로 처리한다.

사람이 검토한 파일 경로는 `data.reviewed_captions`에 지정한다. 기본값 `null`이면 자동 규칙만 사용한다. 검토 파일에 없는 캡션은 자동 규칙을 계속 사용하므로 일부 검토 파일을 넣었다고 전체 자료를 사람 검토 결과로 간주하지 않는다. 각 캡션의 `annotation_status`와 전체 자동 주석 수를 저장한다. `/workspace/annotations/coco_captions/pilot16`의 모델 생성 시험 결과는 이 사람 검토 경로에 등록하지 않았다.

다른 용어 사전을 사용하려면 기본 YAML과 같은 `concepts` 목록을 만들어 `data.concepts_file`로 지정한다. 각 항목에는 `id`, `name`, `aliases`를 넣는다. 고유한 범주 번호를 요구하며 모델과 자료를 바꾸지 않고 표현 사전만 교체할 수 있다. 사전이나 검토 파일의 내용이 바뀌면 이전 실행 폴더에 이어 쓰지 않고 새 `output`을 사용한다.

**사전을 바꾼 재실행은 이전 실행의 산출물을 검증한 뒤 선택적으로 가져올 수 있다.** 새 설정에 `reuse.source_run`으로 완료된 이전 실행 폴더를 지정하면, 각 단계가 실행되는 동안 다음 항목을 검사해 통과한 것만 새 `output`으로 복사한다. 검사에 실패한 항목은 다시 계산한다. 완료 표식(`completed/*.json`)은 절대 복사하지 않으며, 각 단계는 스스로 실행을 마친 뒤 자신의 완료 표식을 남긴다. 어떤 항목을 어떤 검사로 가져왔는지, 또는 왜 다시 계산했는지는 `<output>/reuse.json`에 기록한다.

| 항목 | 재사용 조건 |
| --- | --- |
| 이미지 쪽 색인(`presence.npy`, `full_presence.npy`, `areas.npy`) | 이미지 번호 순서, 범주 번호 목록, 표지 기준(`label_scope`), 인코더 종류·리비전이 같고, 모든 이미지와 주석 PNG 파일의 SHA-256이 이전 실행 기록과 같을 때. 캡션 색인(`captions.json`, `mentions.npy`)은 항상 새 사전으로 다시 만든다. |
| 원본 CLIP 표현 | 내용 주소(인코더 설정, 이미지 파일 지문, 캡션 원문, 라이브러리 버전)로 만든 키가 같은 캐시가 `reuse.source_cache`(기본값은 이전 실행의 `cache`)에 있을 때 복사한다. |
| SAE 가중치 | `reuse.sae_weights`가 참이고 `training.image_checkpoint`를 따로 주지 않으면 이전 실행의 `models/`를 불러와 저장하고, 텐서 단위로 완전히 같은지 확인한다. 다르면 실행을 중단한다. |
| 원본 활성값(`activations/`) | SAE 가중치 지문과 원본 표현 파일의 바이트가 같을 때. |
| 이미지 객체 가림 표현·활성값(`counterfactual/<split>/<c>/image*`) | 가릴 이미지 행 목록, 해당 이미지·주석 파일 지문, 가림 색, 인코더가 같을 때. 활성값은 SAE 가중치가 같을 때만 함께 복사한다. AUROC와 변화량 CSV는 항상 다시 계산한다. |
| 텍스트 가림 표현(`counterfactual/<split>/<c>/text.npy`) | 캡션 행 단위로 캡션 번호, 원문, 가릴 토큰 위치가 정확히 같은 행만 이전 표현을 복사하고 나머지 행은 다시 부호화한다. 모든 행이 같을 때만 텍스트 활성값도 복사한다. |
| 원본 상관행렬(`panel.npz`) | 복사하지 않고 다시 계산한 뒤 이전 행렬과의 최대 절대 차이를 기록한다. |

대표 특징 선택, 연결 판정, 객체 제거 실험은 텍스트 주석에 따라 달라지므로 항상 다시 계산한다.

```yaml
reuse:
  source_run: ../runs/elice-rq1-backup/runs/elice-rq1
  source_cache: ../runs/elice-rq1-backup/cache   # 생략하면 이전 실행 설정의 cache 경로
  image_index: true
  embeddings: true
  sae_weights: true
  original_activations: true
  image_counterfactuals: true
  text_counterfactual_rows: true
```
