from pathlib import Path
import json
from oracle_error_modal import render as render_error_modal
HERE=Path(__file__).resolve().parent
DATA=HERE.parent/'oracle-pursuit-2026-10-06/warm-all-losses-backup/results'
methods=[('Ridge (MSE)', 'mse', '0.01'),('MSE', 'mse','0'),('Binary CE','binary_ce','0.01'),('Binary CE','binary_ce','0'),('CE','ce','0.01'),('CE','ce','0')]
def render():
 cards=[]
 for cond,title in [('coco-coco','COCO2017'),('cc3m-coco','CC3M')]:
  values=[]
  for label,method,reg in methods:
   d=json.loads((DATA/('warmstart-'+cond)/(method+'_l2_'+reg+'_retrieval.json')).read_text())
   values.append([100*d[direction]['recall'][str(k)] for direction in ('image_to_text','text_to_image') for k in (1,5,10)])
  ranks=[sorted({round(v[j],2) for v in values},reverse=True) for j in range(6)]
  best=[r[0] for r in ranks]
  second=[r[1] if len(r)>1 else None for r in ranks]
  rows=''
  for (label,method,reg),v in zip(methods,values):
   style=' style="background:#f3ebed"' if label.startswith('Ridge') else ''
   cells=''.join('<td>'+ (f'<b>{x:.2f}</b>' if round(x,2)==best[j] else f'<span class="sparse-second">{x:.2f}</span>' if round(x,2)==second[j] else f'{x:.2f}')+'</td>' for j,x in enumerate(v))
   rows+=f'<tr{style}><th>{label}<small>λ={reg}</small></th>{cells}</tr>'
  cards.append(f'<div class="result-card"><h3>SAE 학습 · {title}<br>대응 학습 · COCO2017</h3><table class="baseline-table"><colgroup><col style="width:28%"><col span="6" style="width:12%"></colgroup><thead><tr><th rowspan="2">학습 목적</th><th colspan="3">이미지로 텍스트 검색</th><th colspan="3">텍스트로 이미지 검색</th></tr><tr><th>R@1</th><th>R@5</th><th>R@10</th><th>R@1</th><th>R@5</th><th>R@10</th></tr></thead><tbody>{rows}</tbody></table></div>')
 return '''<section class="slide annotation-results" id="oracle-objective-variants"><p class="eyebrow">ORACLE MAPPING · 학습 목적 비교</p><h2>Oracle 매핑의 학습 목적 변경 결과</h2><div class="content"><p style="font-size:19px;line-height:1.5;margin-bottom:15px">Ridge에서 시작해 <b>개념당 최대 16개 activation의 선택과 가중치를 재최적화</b>하고, 가중치 크기 규제 λ=0·0.01을 비교</p><div style="font-size:17px;line-height:1.6;margin-bottom:18px"><b>MSE</b>는 개념 유무의 제곱오차 · <b>Binary CE</b>는 개념별 유무의 이진 교차엔트로피 · <b>CE</b>는 존재하는 개념들에 정답 확률을 균등 배분한 다중 클래스 교차엔트로피</div><div class="annotation-results-grid">'''+''.join(cards)+'''</div><ul style="font-size:21px;line-height:1.5;padding-left:24px;margin-top:23px"><li><b>규제를 유지한 Binary CE·CE는 Ridge와 대체로 비슷하며, Ridge보다 일관되게 나은 검색 성능은 없었음</b></li><li style="margin-top:10px">개념 예측만을 학습하고 정렬을 직접 유도하지 않았으므로, <b>예측 목적을 바꾸더라도 검색 성능 개선이 보장되지는 않음</b></li></ul><button type="button" onclick="document.getElementById('oracle-error-dialog').showModal()" style="margin-top:12px;border:0;background:none;color:var(--accent);font-size:18px;text-decoration:underline;cursor:pointer">학습·검증 데이터의 예측 오차 보기</button></div><aside class="speaker-notes">warm-all-losses-backup의 저장된 결과를 사용한다. 각 목적에 따라 좌표 선택과 가중치를 다시 최적화했다. Ridge 초기화 MSE λ=.01은 기존 Ridge와 같은 결과다. CE는 단일 정답 범주가 아니라 존재하는 모든 개념으로 확률을 균등 배분한다. 검색에서는 sigmoid나 softmax를 사용하지 않는다. 통계적 유의성이나 동등성을 검정한 주장이 아니다.</aside>'''+render_error_modal(DATA,methods)+'''</section>'''
if __name__=='__main__':
 (HERE/'oracle-variants.html').write_text(render())
