# 예시 문장을 삭제하기 전의 COCO 캡션 시험 결과

이 문서는 hot dog 예시 문장을 포함한 이전 프롬프트로 실행한 기록이다. 현재 프롬프트에서는 그 문장을 삭제했다. 같은 16개를 다시 보낸 최신 시험에서 GPT-5.4 `none`은 원문 위치 검사를 16개 모두 통과했고 GPT-5.1 `none`은 12개를 통과했다. 두 모델 모두 의미 오류가 남았다. 최신 상세 기록은 `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions/pilot16/without-hot-dog-example/README.md`에 있다.

예시 문장을 삭제하기 전에는 GPT-5.1, GPT-5.4 mini와 GPT-5.4의 `none` 설정도 각각 16개씩 시험했다. 원문 위치 검사를 각각 12개, 16개, 16개 통과했고 실제 사용량으로 계산한 비용은 각각 $0.02652875, $0.01265025, $0.0421675였다. 추론 토큰은 모두 0개였다. 세 모델의 결과와 사용량은 이 폴더의 JSONL 파일과 manifest.json에 보존했다. GPT-5.4 mini는 예시 문장을 삭제한 뒤 다시 호출하지 않았다. 아래 상세 표는 최초 GPT-4o와 GPT-5 low 비교 기록이다.

**GPT-5 `low`는 16개 모두 원문 표현과 위치 검사를 통과했지만, 모호한 후보를 한쪽으로 단정한 오류가 남았다. 전체 추출은 시작하지 않았다.** 아래 내용은 Codex가 결과와 입력을 대조한 기록이며 사람이 작성한 정답과 비교한 정확도가 아니다.

프롬프트와 Pydantic 응답 형식을 바꾸지 않고 GPT-4o와 GPT-5를 비교했다. 캡션은 학습 자료에서 특정 표현을 포함한 8개와 난수 시드 0으로 고른 학습 자료 4개, 검증 자료 4개다. 이미지가 서로 다른 16개를 사용했다. 오류 유형을 찾아보는 기능 시험이므로 모집단의 정확도를 추정하는 표본으로 해석하지 않는다.

| 측정 항목 | GPT-4o 결과 | GPT-5 low 결과 |
| --- | --- | --- |
| 원문 표현과 위치 검사 | 16개 중 9개가 통과하고 7개가 실패했다. | 16개 모두 통과했다. |
| API 호출 수 | 첫 실행 44회와 실패 진단 7회를 합해 51회 호출했다. | 16회 호출했고 재시도는 없었다. |
| 토큰 사용량 | 입력 19,656개와 출력 4,755개를 사용했다. | 입력 6,319개와 추론을 포함한 출력 10,987개를 사용했다. |
| 알려진 내용 오류 | 출현 순번을 단어 위치처럼 반환했고, puddle을 mud에 연결하거나 모호한 table을 단정했다. | puddle과 mud의 연결은 사라졌지만, 모호한 table을 단정하는 오류가 남았다. |

GPT-4o 첫 실행에는 원문 검사 실패도 재시도하는 구현 오류가 있어 28회 불필요한 재시도가 발생했다. 이를 고친 뒤 실패한 7개만 진단용으로 한 번씩 다시 요청했고 같은 유형의 오류를 확인했다. 첫 실행에서 실패한 원본 응답은 저장하지 못했다. 진단 재요청 응답 7개와 두 모델의 통과 결과는 파일에 보존했다.

## 수정이 필요한 사례

원문은 `A vegetable pizza on the edge of a table`이다. 이미지 후보에 `table`과 `dining table`이 모두 있다. GPT-5는 `table`을 일반 table에 배정하고 dining table은 빈 목록으로 반환했다. 문장만으로 어느 후보인지 구별할 수 없으므로, 승인받은 추출 규칙을 따르면 두 후보 모두 `null`이어야 한다.

기존 규칙을 바꾸지 않고 아래 예시를 추가하는 수정안을 작성했다. 아직 API 요청에는 사용하지 않았다.

```text
If a mention fits multiple candidates, return spans=null for each. For example, "table" is ambiguous between "table" and "dining table" even though one name matches exactly.
```

수정안은 `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions/prompt.proposed.txt`에 있다. 당시 원본 프롬프트는 `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions/pilot16/prompt.txt`에, 실제 응답은 모델별 JSONL에 그대로 보존한다. 현재 수정안 파일에서는 지정한 hot dog 예시 문장만 삭제했으며 모호성 예시는 여전히 실행에 사용하지 않았다.

## GPT-5의 16개 결과

아래 표에는 추출된 표현과 `null`만 적었다. 표에서 생략한 후보의 빈 목록까지 포함한 전체 응답은 `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions/pilot16/gpt5-low.jsonl`에 있다. 시작·끝 위치도 그 파일에서 확인할 수 있다.

| 캡션 번호 | 실제 원문 | 추출 결과 | 입력과 대조해 확인한 내용 |
| --- | --- | --- | --- |
| 28 | A vegetable pizza on the edge of a table | `pizza` 범주에서 추출한 표현은 `vegetable pizza`이다. `table` 범주에서 추출한 표현은 `table`이다. | 수정이 필요하다. table과 dining table이 모두 후보인데 table에만 배정했다. 두 후보 모두 null이어야 한다. |
| 80 | Light colored teddy bear dressed in a hoodie and pants | `teddy bear` 범주에서 추출한 표현은 `teddy bear`이다. | teddy bear를 추출했다. hoodie와 pants를 textile-other에 연결하지 않았다. |
| 249 | A meal of a hot dog cut in half, on top of either bread or crackers, with a certain green vegetable that is stuffed on the side. | `hot dog` 범주에서 추출한 표현은 `hot dog`이다. | hot dog를 음식 표현으로 추출했다. 동물 dog가 입력 후보에 없어 음식과 동물의 동시 구분을 검사한 사례는 아니다. |
| 388 | This is two dogs sniffing a birthday cake | `dog` 범주에서 추출한 표현은 `dogs`이다. `cake` 범주에서 추출한 표현은 `birthday cake`이다. | dogs와 birthday cake를 각각 추출했다. |
| 638 | Three yellow city buses driving down the street | `bus` 범주에서 추출한 표현은 `city buses`이다. `road` 범주에서 추출한 표현은 `street`이다. | city buses와 street를 추출했고 출현 순번 오류가 사라졌다. |
| 1050 | A man sleeping next to a dachshund puppy. | `person` 범주에서 추출한 표현은 `man`이다. `dog` 범주에서 추출한 표현은 `dachshund`, `puppy`이다. | dachshund와 puppy를 같은 dog 범주의 별도 표현으로 추출했다. 명사구 하나로 묶는지까지는 이번 형식이 고정하지 않는다. |
| 2689 | A white kitchen without doors on the cabinets. | `cabinet` 범주에서 추출한 표현은 `cabinets`이다. | cabinets를 추출했다. door는 입력 후보에 없어 부정 표현 제외 규칙을 직접 검증한 사례는 아니다. |
| 9797 | Two pieces of luggage sitting on a wooden floor. | `backpack` 범주의 결과는 `null`이다. `suitcase` 범주의 결과는 `null`이다. `floor-wood` 범주에서 추출한 표현은 `wooden floor`이다. | luggage를 backpack과 suitcase 중 하나로 단정하지 않고 두 후보 모두 null로 처리했다. |
| 157169 | A herd of elephants play in a puddle in a black and white photo. | `elephant` 범주에서 추출한 표현은 `elephants`이다. | elephants만 추출했다. GPT-4o에서 보인 puddle과 mud의 근거 없는 연결이 사라졌다. |
| 239297 | Two men in grassy field playing a game with frisbee. | `person` 범주에서 추출한 표현은 `men`이다. `frisbee` 범주에서 추출한 표현은 `frisbee`이다. `grass` 범주에서 추출한 표현은 `grassy`이다. | men, frisbee와 grassy를 추출했다. |
| 522614 | A bench sitting on top of a stone walkway. | `bench` 범주에서 추출한 표현은 `bench`이다. `floor-stone` 범주에서 추출한 표현은 `stone walkway`이다. | bench와 stone walkway를 추출했다. |
| 557503 | a person playing with a kite on the beach | `person` 범주에서 추출한 표현은 `person`이다. `kite` 범주에서 추출한 표현은 `kite`이다. | person과 kite를 추출했다. beach만으로 sand가 명시됐다고 단정하지 않았다. |
| 14844 | The opened  laptop is sitting on the table along with other technological tools. | `laptop` 범주에서 추출한 표현은 `laptop`이다. `table` 범주에서 추출한 표현은 `table`이다. | laptop과 table을 추출했다. 나머지 technological tools를 개별 기기 후보에 임의로 배정하지 않았다. |
| 370551 | A young boy in a baseball uniform holding a bat over his shoulder. | `person` 범주에서 추출한 표현은 `boy`이다. `baseball bat` 범주에서 추출한 표현은 `bat`이다. `clothes` 범주에서 추출한 표현은 `baseball uniform`이다. | boy, bat와 baseball uniform을 추출했다. |
| 381033 | A dog looks interested as he sits in the front seat of a car. | `dog` 범주에서 추출한 표현은 `dog`, `he`이다. | dog와 이를 지칭하는 he를 추출했다. car는 원문에 있지만 이미지 주석 후보에 없어 추출 대상에 포함되지 않았다. |
| 502931 | A woman standing next to a  brown and white dog. | `person` 범주에서 추출한 표현은 `woman`이다. `dog` 범주에서 추출한 표현은 `dog`이다. | woman과 dog를 추출했다. 연속 공백이 있는 원문 위치도 일치했다. |
