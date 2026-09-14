# GPT-5.4와 GPT-5.1의 캡션 16개 비교

**GPT-5.4는 원문 위치 검사를 16개 모두 통과했고, GPT-5.1은 12개를 통과했다. GPT-5.4에서도 모호한 표현을 범주에 잘못 배정하는 오류가 남았다. 이번 결과만으로 어느 모델도 전체 추출에 사용할 준비가 끝났다고 판단하지 않았다.** 전체 캡션 추출과 SAE 전체 실험은 시작하지 않았다.

2026년 9월 14일에 같은 실제 COCO 캡션 16개를 두 모델에 각각 한 번씩 보냈다. 기존 프롬프트에서 `A hot dog is not an animal dog.` 문장만 삭제했다. 나머지 지시와 Pydantic 응답 형식은 유지했다. 별도로 작성했던 table 모호성 예시는 추가하지 않았다. 이전 프롬프트로 실행한 결과와 이번 결과를 섞지 않았다.

각 요청에는 캡션 하나와 대응 이미지의 주석에 나타난 범주 번호·이름만 넣었다. 이미지는 보내지 않았다. 두 모델 모두 `reasoning_effort="none"`, `temperature=0`, `top_p=1`, `n=1`, `max_completion_tokens=2048`, `store=false`를 사용했다. `AsyncOpenAI`와 `asyncio.Semaphore(512)`를 사용했으며 모델별 입력은 16개였다. 두 모델을 동시에 실행해 전체 동시 요청은 최대 32개였다. 재시도는 없었다.

| 측정 항목 | GPT-5.4 none 결과 | GPT-5.1 none 결과 |
| --- | --- | --- |
| 모델 버전 | `gpt-5.4-2026-03-05`를 사용했다. | `gpt-5.1-2025-11-13`을 사용했다. |
| 원문 표현과 위치 검사 | 캡션 16개 모두 통과했다. | 캡션 12개가 통과하고 4개가 실패했다. |
| 입력 토큰 수 | 6,175개를 사용했다. | 6,175개를 사용했다. |
| 출력 토큰 수 | 1,779개를 사용했다. | 1,855개를 사용했다. |
| 추론 토큰 수 | API가 0개로 보고했다. | API가 0개로 보고했다. |
| 16개 요청의 비용 | 실제 사용량으로 계산하면 $0.0421225다. | 실제 사용량으로 계산하면 $0.02626875다. |
| 학습 캡션 591,753개 비용 추정 | 약 $1,558로 추정했다. | 약 $972로 추정했다. |
| 검증 캡션 25,014개 비용 추정 | 약 $66로 추정했다. | 약 $41로 추정했다. |
| 전체 캡션 616,767개 비용 추정 | 약 $1,624로 추정했다. | 약 $1,013으로 추정했다. |
| 동시 요청 16개의 완료 시간 | 3.40초 걸렸다. | 2.83초 걸렸다. |

GPT-5.4의 일반 API 요금은 입력 100만 토큰당 $2.50, 출력 100만 토큰당 $15.00이다. GPT-5.1은 각각 $1.25와 $10.00이다. 입력 단가는 두 배이고 출력 단가는 1.5배다. 실제 출력 길이까지 반영한 이번 비용은 GPT-5.4가 약 1.60배였다. [GPT-5.4 공식 요금](https://developers.openai.com/api/docs/models/gpt-5.4), [GPT-5.1 공식 요금](https://developers.openai.com/api/docs/models/gpt-5.1)

전체 비용은 16개 요청의 비용을 16으로 나눈 뒤 전체 캡션 수를 곱했다. 실패 응답도 비용에 포함했다. 이 16개는 특정 표현을 포함한 학습 캡션 8개, 난수 시드 0으로 고른 학습 캡션 4개와 검증 캡션 4개이며 이미지가 모두 다르다. 오류를 찾기 위한 기능 시험이므로 전체 평균 비용이나 정확도를 대표한다고 볼 수 없다. 재시도, 세금과 서버 비용은 추정에서 제외했다. 실제 캐시 적중은 없었고 Batch 할인도 사용하지 않았다. 16개의 완료 시간으로 전체 처리 시간을 추정하지 않았다.

## 위치 오류와 의미 오류를 구분한 결과

원문 위치 검사는 범주 번호·순서가 입력과 같고, 반환한 문자열과 출현 순번이 실제 원문에 존재하는지 확인한다. `occurrence=2`는 해당 문자열이 원문에서 두 번째로 나온다는 뜻이다. 문장에서 두 번째 단어라는 뜻이 아니다. 이 검사를 통과해도 범주 배정이 의미상 맞는지는 별도 문제다.

**GPT-5.1은 문장 속 단어 위치를 출현 순번으로 답하는 오류를 반복했다.** 캡션 381033의 `A dog looks interested as he sits in the front seat of a car.`에서 dog는 한 번 나오지만 두 번째 출현이라고 반환했다. 같은 오류가 캡션 28, 557503과 14844에도 있다. 이 네 응답은 자동으로 고치거나 정답으로 포함하지 않고 실패 응답으로 보존했다.

**GPT-5.4는 원문 위치를 맞혔지만 후보 간 모호성을 해결하지 못했다.** 다음 표는 입력과 출력의 의미를 대조한 검토 기록이다. 사람이 작성한 정답과 비교해 계산한 정확도가 아니다.

| 실제 문장과 주어진 후보 | GPT-5.4의 응답 | GPT-5.1의 응답 | 현재 규칙에서 확인한 문제 |
| --- | --- | --- | --- |
| 캡션 28은 `A vegetable pizza on the edge of a table`이다. 후보에 table과 dining table이 모두 있다. | 같은 `table`을 두 범주 모두에 긍정 표현으로 배정했다. | `table` 범주에만 배정했고 출현 순번도 8로 잘못 반환했다. | 문장만으로 두 후보를 구별할 수 없으므로 두 후보 모두 `null`이어야 한다. |
| 캡션 9797은 `Two pieces of luggage sitting on a wooden floor.`이다. 후보에 backpack과 suitcase가 모두 있다. | `luggage`를 suitcase에만 배정했다. | `luggage`를 suitcase에만 배정했다. | luggage만으로 backpack과 suitcase 중 무엇인지 단정할 수 없다. 두 후보 모두 `null`이어야 한다. |
| 캡션 157169는 `A herd of elephants play in a puddle in a black and white photo.`이다. 후보에 mud가 있다. | mud에 해당하는 표현을 추출하지 않았다. | `puddle`을 mud에 배정했다. | 물웅덩이라는 표현만으로 진흙을 지칭한다고 볼 수 없다. GPT-5.1이 문장에 없는 의미를 추정했다. |
| 캡션 557503은 `a person playing with a kite on the beach`이다. 후보에 sand가 있다. | `beach`를 sand에 배정했다. | `beach`를 sand에 배정했고 출현 순번도 잘못 반환했다. | 해변이라는 장소 표현만으로 모래라는 재료가 명시됐다고 볼 수 없다. 두 모델 모두 추론을 포함했다. |
| 캡션 381033은 `A dog looks interested as he sits in the front seat of a car.`이다. 후보에 dog가 있다. | dog만 추출하고 이를 지칭하는 `he`를 빠뜨렸다. | dog만 추출하고 이를 지칭하는 `he`를 빠뜨렸다. | 모든 지칭 표현을 추출한다는 현재 지시를 따르면 he도 포함해야 한다. dog만 가리면 같은 객체의 대명사가 남는다. |

캡션 28에서 GPT-5.4는 `pizza`를 pizza 범주에, 겹치는 `vegetable pizza`를 food-other 범주에 배정했다. 이런 서로 다른 범주의 범위 중복은 원문 위치 검사만으로는 실패하지 않는다. 실제 토큰 가림을 준비할 때는 다른 범주와 겹치는 범위를 가림 불가로 처리하므로 16개 통과를 곧바로 16개 사용 가능으로 읽으면 안 된다. 캡션 80의 hoodie와 pants를 textile-other에 포함할지도 범주의 정의와 대조해야 한다. 이번에 그 판단을 확정하지 않았다.

이번 요청에서는 프롬프트의 지정 문장만 삭제했다. 모호성 규칙, 대명사 처리, 응답 형식을 추가로 바꾸거나 실패를 자동 보정하지 않았다. GPT-5.1은 현재 형식에서 원문 위치 오류가 남고, GPT-5.4는 위치가 더 안정적이지만 의미 오류가 남는다는 것이 이번 시험의 결론이다.

## 16개 응답의 실제 추출 표현

아래에는 빈 목록을 제외한 응답을 모두 적었다. `null`은 어느 후보에 배정할지 모호하다는 모델의 답이다. 괄호 안 숫자는 모델이 반환한 출현 순번이며 오류도 그대로 남겼다. 순번이 1인 경우는 생략했다. 원문 위치 검사 실패 응답도 비교에서 제외하지 않았다.

| 캡션 번호와 원문 | GPT-5.4의 응답 | GPT-5.1의 응답 |
| --- | --- | --- |
| 28. `A vegetable pizza on the edge of a table` | pizza 범주의 결과는 `pizza`이다. dining table 범주의 결과는 `table`이다. food-other 범주의 결과는 `vegetable pizza`이다. table 범주의 결과는 `table`이다. | 위치 검사에 실패했다. pizza 범주의 결과는 `pizza` (출현 순번 3)이다. food-other 범주의 결과는 `vegetable` (출현 순번 2)이다. table 범주의 결과는 `table` (출현 순번 8)이다. |
| 80. `Light colored teddy bear dressed in a hoodie and pants` | teddy bear 범주의 결과는 `teddy bear`이다. textile-other 범주의 결과는 `hoodie`, `pants`이다. | teddy bear 범주의 결과는 `teddy bear`이다. |
| 249. `A meal of a hot dog cut in half, on top of either bread or crackers, with a certain green vegetable that is stuffed on the side.` | hot dog 범주의 결과는 `hot dog`이다. | hot dog 범주의 결과는 `hot dog`이다. |
| 388. `This is two dogs sniffing a birthday cake` | dog 범주의 결과는 `dogs`이다. cake 범주의 결과는 `birthday cake`이다. | dog 범주의 결과는 `dogs`이다. cake 범주의 결과는 `cake`이다. |
| 638. `Three yellow city buses driving down the street` | bus 범주의 결과는 `buses`이다. road 범주의 결과는 `street`이다. | bus 범주의 결과는 `buses`이다. road 범주의 결과는 `street`이다. |
| 1050. `A man sleeping next to a dachshund puppy.` | person 범주의 결과는 `man`이다. dog 범주의 결과는 `dachshund puppy`이다. | person 범주의 결과는 `man`이다. dog 범주의 결과는 `dachshund puppy`이다. |
| 2689. `A white kitchen without doors on the cabinets.` | cabinet 범주의 결과는 `cabinets`이다. | cabinet 범주의 결과는 `cabinets`이다. |
| 9797. `Two pieces of luggage sitting on a wooden floor.` | suitcase 범주의 결과는 `luggage`이다. floor-wood 범주의 결과는 `wooden floor`이다. | suitcase 범주의 결과는 `luggage`이다. floor-wood 범주의 결과는 `wooden floor`이다. textile-other 범주의 결과는 `null`이다. |
| 157169. `A herd of elephants play in a puddle in a black and white photo.` | elephant 범주의 결과는 `elephants`이다. | elephant 범주의 결과는 `elephants`이다. mud 범주의 결과는 `puddle`이다. |
| 239297. `Two men in grassy field playing a game with frisbee.` | person 범주의 결과는 `men`이다. frisbee 범주의 결과는 `frisbee`이다. grass 범주의 결과는 `grassy`이다. | person 범주의 결과는 `men`이다. frisbee 범주의 결과는 `frisbee`이다. grass 범주의 결과는 `grassy`이다. |
| 522614. `A bench sitting on top of a stone walkway.` | bench 범주의 결과는 `bench`이다. floor-stone 범주의 결과는 `stone walkway`이다. | bench 범주의 결과는 `bench`이다. floor-stone 범주의 결과는 `stone walkway`이다. |
| 557503. `a person playing with a kite on the beach` | person 범주의 결과는 `person`이다. kite 범주의 결과는 `kite`이다. sand 범주의 결과는 `beach`이다. | 위치 검사에 실패했다. person 범주의 결과는 `person` (출현 순번 2)이다. kite 범주의 결과는 `kite` (출현 순번 6)이다. sand 범주의 결과는 `beach` (출현 순번 9)이다. |
| 14844. `The opened  laptop is sitting on the table along with other technological tools.` | laptop 범주의 결과는 `laptop`이다. table 범주의 결과는 `table`이다. | 위치 검사에 실패했다. laptop 범주의 결과는 `laptop` (출현 순번 3)이다. table 범주의 결과는 `table` (출현 순번 9)이다. |
| 370551. `A young boy in a baseball uniform holding a bat over his shoulder.` | person 범주의 결과는 `boy`이다. baseball bat 범주의 결과는 `bat`이다. clothes 범주의 결과는 `uniform`이다. | person 범주의 결과는 `boy`이다. baseball bat 범주의 결과는 `bat`이다. clothes 범주의 결과는 `uniform`이다. |
| 381033. `A dog looks interested as he sits in the front seat of a car.` | dog 범주의 결과는 `dog`이다. | 위치 검사에 실패했다. dog 범주의 결과는 `dog` (출현 순번 2)이다. |
| 502931. `A woman standing next to a  brown and white dog.` | person 범주의 결과는 `woman`이다. dog 범주의 결과는 `dog`이다. | person 범주의 결과는 `woman`이다. dog 범주의 결과는 `dog`이다. |

## 저장한 파일

이번 결과는 `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions/pilot16/without-hot-dog-example`에 저장했다. 원본 API 응답과 실행 기록은 Git에서 제외하고, 캡션의 추출 결과와 실패 이유, 사용량, 프롬프트, 내용 지문을 보존했다.

| 파일 | 저장한 내용 |
| --- | --- |
| `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions/pilot16/without-hot-dog-example/prompt.txt` | 이번 두 모델에 실제로 보낸 프롬프트다. |
| `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions/pilot16/without-hot-dog-example/gpt54-none.jsonl` | GPT-5.4의 위치 검사 통과 응답 16개다. 의미 오류도 수정하지 않았다. |
| `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions/pilot16/without-hot-dog-example/gpt51-none.jsonl` | GPT-5.1의 위치 검사 통과 응답 12개다. 의미 오류도 수정하지 않았다. |
| `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions/pilot16/without-hot-dog-example/gpt51-none-rejected.jsonl` | GPT-5.1의 위치 검사 실패 응답 4개와 실패 이유다. |
| `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions/pilot16/without-hot-dog-example/manifest.json` | 모델 설정, 사용량, 요금, 비용 계산, 입력·출력 지문을 저장했다. |
