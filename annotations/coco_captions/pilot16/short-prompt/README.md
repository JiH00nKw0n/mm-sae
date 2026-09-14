# 짧은 프롬프트로 바꾼 GPT-5.4 mini의 16개 비교

**범주 판단은 기존 긴 프롬프트와 같았지만 원문 복사 오류가 한 건 발생했다. 비용은 약 5.5% 줄었다.** 따라서 제안한 짧은 문구가 기존 품질을 그대로 유지했다고 판단하지 않았다. 기본 프롬프트는 바꾸지 않았고 전체 캡션 추출과 SAE 전체 실험도 시작하지 않았다.

2026년 9월 14일에 앞서 제시한 네 줄의 축약안을 그대로 GPT-5.4 mini에 보냈다. 같은 실제 COCO 캡션 16개, 같은 이미지 범주 목록, 같은 Pydantic 응답 형식, 같은 모델과 요청 설정을 사용했다. 기존 긴 문구의 응답은 저장된 결과와 비교했다. 기존 조건을 다시 호출하거나 실패한 응답을 재시도하지 않았다.

| 비교 항목 | 기존 긴 프롬프트 | 이번 짧은 프롬프트 |
| --- | --- | --- |
| 공백 기준 단어 수 | 92개였다. | 46개였다. |
| 원문 표현 검사 | 16개 모두 통과했다. | 15개가 통과하고 1개가 실패했다. |
| 임시 범주 기준과의 일치 | 131개 중 125개로 95.4%였다. | 131개 중 125개로 95.4%였다. |
| 모호한 후보 4개 | 모호함으로 답한 후보가 0개였다. | 모호함으로 답한 후보가 0개였다. |
| 원문 검사와 판정 대상 범주가 모두 일치한 캡션 | 16개 중 12개였다. | 16개 중 11개였다. |
| 입력 토큰 수 | 5,455개를 사용했다. | 4,543개를 사용했다. |
| 출력 토큰 수 | 1,512개를 사용했다. | 1,532개를 사용했다. |
| 16개 요청 비용 | $0.01089525였다. | $0.01030125였다. |
| 전체 616,767개 예상 비용 | 약 $420로 추정했다. | 약 $397로 추정했다. |

단가는 입력 100만 토큰당 $0.75, 출력 100만 토큰당 $4.50를 적용했다. 실패 응답의 비용도 포함했다. 단어 수는 절반이 됐지만 입력의 범주 목록·응답 형식과 출력 비용은 그대로 발생하므로 전체 비용 감소는 5.5%였다. 전체 비용은 16개 평균을 환산한 대략적인 추정이며 재시도·세금·서버 비용을 포함하지 않는다. [GPT-5.4 mini 공식 요금](https://developers.openai.com/api/docs/models/gpt-5.4-mini)

모델은 `gpt-5.4-mini-2026-03-17`이고 `reasoning_effort="none"`, `temperature=0`, `top_p=1`, `n=1`, `max_completion_tokens=2048`, `store=false`를 유지했다. `AsyncOpenAI`와 `asyncio.Semaphore(512)`를 사용했으며 요청 하나당 캡션 하나를 보냈다. 이번에는 최대 16개를 동시에 처리했다. API가 보고한 추론 토큰은 0개이고 캐시 적중과 재시도도 없었다. 16개 요청은 약 2.00초에 끝났다.

## 실제로 사용한 짧은 프롬프트

```text
For each supplied category, extract its exact referring expressions from the caption, including synonyms, inflections, and pronouns.
Return every category in input order.
Use spans=[] if unmentioned and spans=null if ambiguous between categories.
Do not infer mentions from the category list. Exclude negated or hypothetical mentions.
```

응답은 객체별 `spans: list[str] | None`이다. 모델이 반환한 문자열의 모든 일치 위치를 코드가 찾는다. 이번 축약안은 대명사를 명시했고, 원문 대소문자·복합어·동일 표현의 중복·JSON 출력 등을 따로 설명하던 문장을 없앴다. 따라서 단어 수만 줄인 비교가 아니라 지시의 표현도 바뀐 비교다.

## 원문 복사 오류의 내용

실패한 캡션 14844의 원문은 다음과 같다. opened와 laptop 사이에는 공백이 두 개 있다.

```text
The opened  laptop is sitting on the table along with other technological tools.
```

짧은 프롬프트의 모델 응답은 laptop 범주에 `["opened laptop", "laptop"]`이었다. 첫 표현은 원문의 공백 두 개를 한 개로 바꿨으므로 현재의 정확한 문자열 검색에서 찾을 수 없다. 두 번째 표현 laptop은 원문에 있지만, 현재 검사는 한 표현이라도 불일치하면 캡션 전체를 실패로 처리한다. 긴 프롬프트는 laptop만 반환해 통과했다.

객체를 다른 범주로 해석한 오류와 공백을 다르게 복사한 오류는 구분해야 한다. 이번에는 객체 범주 선택은 같았고 원문 복사에서만 추가 실패가 생겼다. 실패한 표현을 삭제하거나 공백을 자동 보정하지 않았으며 원래 응답을 그대로 저장했다.

## 의미 판단과 가림 범위의 변화

136개 후보 전체에서 표현 있음·표현 없음·모호함이라는 상태가 바뀐 항목은 없었다. 이전에 Codex가 정한 임시 기준도 수정하지 않았다. 범주 정의나 처리 규칙이 불확실했던 5개를 계속 보류하고 나머지 131개를 평가했다. 표현 있음 27개와 표현 없음 100개, 모호함 4개 중 일치한 수는 각각 27개, 98개, 0개였다.

위치 검사에 실패한 한 응답도 JSON에서 범주 상태는 읽을 수 있어 131개 비교에 포함했다. 이를 제외하고 통과 응답만 비교한 수치가 아니다. 95.4%는 원문을 정확히 가릴 수 있는 비율도, 사람이 검수한 정답과의 정확도도 아니다. 이 16개는 앞선 시험들에서 반복 사용한 기능 시험 자료이며 전체 정확도를 대표하지 않는다.

표현의 범위는 달라졌다. 예를 들어 dogs가 two dogs로, elephants가 A herd of elephants로, boy가 A young boy로 바뀌었다. 같은 범주를 선택해도 더 많은 토큰을 가릴 수 있다. dog를 지칭하는 he는 두 프롬프트 모두 추출했지만, `A young boy ... his shoulder`의 his는 짧은 프롬프트에서도 빠졌다. 대명사를 명시한 것만으로 모든 지칭 표현의 누락이 해결되지는 않았다.

모호한 table과 dining table, luggage와 backpack·suitcase의 처리 오류는 그대로였다. puddle을 mud로, beach를 sand로 연결한 것도 같았다. 이번 결과에서는 축약으로 범주 판단이 개선되거나 악화됐다는 증거를 얻지 못했다. 단 한 번의 16개 비교로 원문 복사 오류가 항상 재현된다고 단정하지도 않는다.

현재 축약안은 기본 프롬프트로 채택하지 않았다. 다음 수정에서는 원문의 공백과 대소문자까지 그대로 복사한다는 지시를 짧게 명시하는 것을 검토할 수 있다. 그 추가 문구는 이번 시험에 사용하지 않았다. 문자열 검색 규칙과 오류 처리도 바꾸지 않았다.

## 16개 응답을 그대로 비교한 기록

빈 목록을 제외한 범주의 답을 적었다. 실패한 캡션도 제외하지 않았고 그 행에는 실제 모델 응답을 표시했다.

| 캡션 번호와 원문 | 기존 긴 프롬프트의 답 | 이번 짧은 프롬프트의 답 |
| --- | --- | --- |
| 28. `A vegetable pizza on the edge of a table` | pizza 범주의 답은 `pizza`이다. dining table 범주의 답은 `table`이다. food-other 범주의 답은 `vegetable pizza`이다. table 범주의 답은 `table`이다. | pizza 범주의 답은 `pizza`이다. dining table 범주의 답은 `table`이다. food-other 범주의 답은 `vegetable pizza`이다. table 범주의 답은 `table`이다. |
| 80. `Light colored teddy bear dressed in a hoodie and pants` | teddy bear 범주의 답은 `teddy bear`이다. | teddy bear 범주의 답은 `teddy bear`이다. |
| 249. `A meal of a hot dog cut in half, on top of either bread or crackers, with a certain green vegetable that is stuffed on the side.` | hot dog 범주의 답은 `hot dog`이다. | hot dog 범주의 답은 `hot dog`이다. |
| 388. `This is two dogs sniffing a birthday cake` | dog 범주의 답은 `dogs`이다. cake 범주의 답은 `birthday cake`이다. | dog 범주의 답은 `two dogs`이다. cake 범주의 답은 `a birthday cake`이다. |
| 638. `Three yellow city buses driving down the street` | bus 범주의 답은 `buses`이다. road 범주의 답은 `street`이다. | bus 범주의 답은 `buses`이다. road 범주의 답은 `street`이다. |
| 1050. `A man sleeping next to a dachshund puppy.` | person 범주의 답은 `A man`이다. dog 범주의 답은 `dachshund puppy`이다. | person 범주의 답은 `A man`이다. dog 범주의 답은 `a dachshund puppy`이다. |
| 2689. `A white kitchen without doors on the cabinets.` | cabinet 범주의 답은 `cabinets`이다. | cabinet 범주의 답은 `cabinets`이다. |
| 9797. `Two pieces of luggage sitting on a wooden floor.` | suitcase 범주의 답은 `luggage`이다. floor-wood 범주의 답은 `wooden floor`이다. | suitcase 범주의 답은 `Two pieces of luggage`이다. floor-wood 범주의 답은 `wooden floor`이다. |
| 157169. `A herd of elephants play in a puddle in a black and white photo.` | elephant 범주의 답은 `elephants`이다. mud 범주의 답은 `puddle`이다. | elephant 범주의 답은 `A herd of elephants`이다. mud 범주의 답은 `puddle`이다. |
| 239297. `Two men in grassy field playing a game with frisbee.` | person 범주의 답은 `Two men`이다. frisbee 범주의 답은 `frisbee`이다. grass 범주의 답은 `grassy field`이다. | person 범주의 답은 `Two men`이다. frisbee 범주의 답은 `frisbee`이다. grass 범주의 답은 `grassy field`이다. |
| 522614. `A bench sitting on top of a stone walkway.` | bench 범주의 답은 `bench`이다. floor-stone 범주의 답은 `stone walkway`이다. | bench 범주의 답은 `bench`이다. floor-stone 범주의 답은 `stone walkway`이다. |
| 557503. `a person playing with a kite on the beach` | person 범주의 답은 `person`이다. kite 범주의 답은 `kite`이다. sand 범주의 답은 `beach`이다. | person 범주의 답은 `person`이다. kite 범주의 답은 `kite`이다. sand 범주의 답은 `beach`이다. |
| 14844. `The opened  laptop is sitting on the table along with other technological tools.` | laptop 범주의 답은 `laptop`이다. table 범주의 답은 `table`이다. | 원문 검사에 실패했다. laptop 범주의 답은 `opened laptop`, `laptop`이다. table 범주의 답은 `table`이다. |
| 370551. `A young boy in a baseball uniform holding a bat over his shoulder.` | person 범주의 답은 `boy`이다. baseball bat 범주의 답은 `bat`이다. clothes 범주의 답은 `uniform`이다. | person 범주의 답은 `A young boy`이다. baseball bat 범주의 답은 `a bat`이다. clothes 범주의 답은 `a baseball uniform`이다. |
| 381033. `A dog looks interested as he sits in the front seat of a car.` | dog 범주의 답은 `dog`, `he`이다. | dog 범주의 답은 `A dog`, `he`이다. |
| 502931. `A woman standing next to a  brown and white dog.` | person 범주의 답은 `A woman`이다. dog 범주의 답은 `dog`이다. | person 범주의 답은 `A woman`이다. dog 범주의 답은 `a  brown and white dog`이다. |

시험 기록은 `/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/annotations/coco_captions/pilot16/short-prompt`에 저장했다. prompt.txt와 response_format.json은 실제 요청 조건이고, gpt54-mini-none.jsonl은 통과한 15개 결과다. gpt54-mini-none-rejected.jsonl은 실패한 1개의 원래 응답이다. gpt54-mini-evaluation.json에는 실패 응답까지 포함한 범주 비교를, manifest.json에는 비용과 설정, 파일 지문을 기록했다. API 키와 원본 API 응답, SQLite와 진행 로그는 Git에 넣지 않았다.
