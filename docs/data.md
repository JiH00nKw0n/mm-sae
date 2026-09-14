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

**문장 객체 표지는 기본적으로 자동 규칙이며 별도 검토가 필요하다.** 범주명과 동의어의 단어 경계를 찾아 해당 표현을 모두 지운다. `hot dog` 안의 `dog`처럼 겹치는 표현은 긴 범주 표현을 우선한다. 두 범주의 표현이 분리되지 않으면 해당 편집을 사용할 수 없다고 기록한다. 자동 편집은 공백만 정리하며 문법을 다시 쓰거나 생성 모델로 문장을 보완하지 않는다.

각 분할의 검토 목록은 `/workspace/runs/coco-rq1/index/train2017/caption_edit_review.csv`와 `/workspace/runs/coco-rq1/index/val2017/caption_edit_review.csv`에 저장한다. 이 목록은 자동으로 찾은 표현을 검토하기 위한 출발점이다. 자동 규칙이 놓친 표현이나 부정문까지 확인하려면 원본 캡션도 검토해야 한다. COCO-Stuff 자체가 이 텍스트 의미 정답을 제공하지는 않는다.

검토한 문장을 JSONL 파일로 입력하면 해당 캡션의 자동 표지와 편집을 대체한다. JSONL은 한 줄마다 JSON 객체를 하나 저장한 형식이다. 다음은 실제 COCO 캡션을 인용한 것이 아닌 입력 형식 예시다.

```json
{"caption_id": 123, "original": "A person is beside a bicycle.", "concept_ids": [0, 1], "edits": {"0": "A bicycle is visible.", "1": "A person is visible."}}
```

`caption_id`와 `original`은 검토한 COCO 캡션의 실제 식별자와 원문이어야 한다. `concept_ids`에는 명시한 범주의 PNG 원시 번호를 넣는다. `edits`의 각 키는 제거한 개념이고 값은 편집 문장이다. 개념이 언급됐지만 다른 개념을 유지하면서 제거할 수 없다면 해당 키를 `edits`에서 생략한다. 문법 정리를 하더라도 다른 객체의 존재·행동·관계 주장을 추가하거나 삭제하지 않았는지 사람이 확인해야 한다.

파일 경로는 `data.reviewed_captions`에 지정한다. 기본값 `null`은 자동 규칙만 사용한다. 검토 파일에 없는 캡션은 자동 규칙을 계속 사용하므로 일부 검토 파일을 넣었다는 이유만으로 전체 자료가 사람 검토를 통과한 것은 아니다. 각 캡션의 `annotation_status`와 전체 자동 주석 수를 저장한다.

다른 용어 사전을 사용하려면 기본 YAML과 같은 `concepts` 목록을 만들어 `data.concepts_file`로 지정한다. 각 항목에는 `id`, `name`, `aliases`를 넣는다. 고유한 범주 번호를 요구하며 모델과 자료를 바꾸지 않고 표현 사전만 교체할 수 있다. 사전이나 검토 파일의 내용이 바뀌면 이전 실행 폴더에 이어 쓰지 않고 새 `output`을 사용한다.
