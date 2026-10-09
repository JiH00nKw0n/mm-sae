from pathlib import Path
import json,copy,html
R=Path('/Users/jihoonkwon/Desktop/projects/research/MM-SAE');O=R/'artifacts/real-results-2026-10-09';C=R/'artifacts/caption-matching-2026-10-09/json-results'
d=json.loads((O/'results.json').read_text());main=copy.deepcopy(d['main']);evaluated=0;tot=0;denominators=set()
for cond,rows in main.items():
 for row in rows:
  key=row['key'];key='hungarian' if key.startswith('hungarian') else 'sparse_cca_16' if key=='cca_16' else key
  keys=[key+'_seed'+str(i) for i in range(3)] if key.startswith('nce_') else [key+'_seed0'] if key=='sparse_cca_unitnorm' else [key]
  vals=[]
  for k in keys:
   p=C/cond/'results'/('test__'+k+'.json')
   if p.exists():
    j=json.loads(p.read_text());n=j['n_evaluated_categories'];denominators.add(n)
    vals.append([j['positive_only_sensitivity']['summary']['agree_at1_count'],j['summary']['agree_at1_count']])
  row['old_image_label_matching']=row['v'][6:];row['v'][6:]=[None,None];tot+=1
  if len(vals)==len(keys):
   row['v'][6:]=[sum(v[i] for v in vals)/len(vals) for i in range(2)];evaluated+=1
   row['matching_sources']=[str(C/cond/'results'/('test__'+k+'.json')) for k in keys]
assert len(denominators)<=1
n=next(iter(denominators),None)
heads=['方法','이미지 질의<br>R@1','이미지 질의<br>R@5','이미지 질의<br>R@10','텍스트 질의<br>R@1','텍스트 질의<br>R@5','텍스트 질의<br>R@10',f'자체 캡션 주석 /{n}<br>높은 활성만',f'자체 캡션 주석 /{n}<br>반응 방향 허용']
def table(rows):
 ranks=[sorted(set(round(r['v'][i],2) for r in rows if r['v'][i] is not None),reverse=True) for i in range(8)]
 out=['<div class="scroll"><table><tr>'+''.join('<th>'+h.replace('方法','방법 및 조건')+'</th>' for h in heads)+'</tr>']
 for r in rows:
  out.append('<tr><th>'+html.escape(r['label'])+'</th>')
  for i,v in enumerate(r['v']):
   t='계산 중' if v is None else f'{v:.2f}'
   if v is not None:
    rank=ranks[i].index(round(v,2))
    if rank==0:t='<b>'+t+'</b>'
    elif rank==1:t='<u>'+t+'</u>'
   out.append('<td>'+t+'</td>')
  out.append('</tr>')
 return ''.join(out)+'</table></div>'
template=O/'previous-image-label-report.html'
if not template.exists():template.write_text((O/'report.html').read_text())
old=template.read_text()
if '<!--CAPTION-START-->' in old:old=old[:old.index('<!--CAPTION-START-->')]+old[old.index('<!--CAPTION-END-->')+len('<!--CAPTION-END-->'):]
start=old.index('<h1>');end=old.index('<h2>추가로 평가한 대표 좌표 매칭</h2>')
content=f'''<!--CAPTION-START--><h1>검색 성능과 이미지·캡션 각각의 범주 주석을 사용한 매칭</h1><p><b>텍스트 대표 좌표를 캡션 자체의 범주 언급으로 다시 골랐습니다.</b> 이미지에서는 객체가 있는지, 텍스트에서는 해당 범주를 언급하는지로 각각 AUROC가 가장 높은 좌표를 선택하고 고정된 대응에서 같은 번호인지 확인했습니다. CCA의 같은 k를 비교한다는 정의는 그대로입니다.</p><p>캡션 주석은 저장된 동의어 사전 기반 mentions.npy입니다. 사람이 의미를 전부 판정한 정답은 아니며 간접 표현·다의어·부정 표현에 오류가 있을 수 있습니다. 각 선택 집단에서 양성이 50개 이상인 기준을 유지했을 때 공통 평가 범주는 {n}개입니다. 이전의 이미지 주석을 캡션에 전달한 50개 평가와 수치를 직접 비교하지 않습니다.</p><p>검색은 COCO val2017 이미지 5,000장·캡션 25,014개이며 Recall 수치는 %입니다. 검색 결과는 기존 값을 유지했고, 매칭만 다시 계산했습니다. 이미지 질의는 캡션 검색, 텍스트 질의는 이미지 검색입니다. 1등은 <b>굵게</b>, 2등은 <u>밑줄</u>로 표시하며 소수 둘째 자리 동률은 공동 순위입니다. InfoNCE는 세 번의 평균입니다.</p><p>주요 표의 매칭 평가 {evaluated}/{tot}개를 반영했습니다. 후보 좌표 수는 방법마다 다릅니다. CCA·PLS는 학습한 조합 쌍, Sinkhorn은 검색에 사용한 변환 후 좌표 쌍, Procrustes는 검색에 사용한 SVD 공통 좌표를 평가했습니다. 특히 Procrustes의 점수는 공통 좌표를 어떻게 정하느냐에 의존하며 회전 불변인 의미 점수가 아닙니다. Oracle은 범주 주석을 사용해 학습한 별도 조건입니다.</p>'''
for cond,rows in main.items():
 a,b={'coco-coco':('COCO2017','COCO2017'),'cc3m-cc3m':('CC3M','CC3M'),'cc3m-coco':('CC3M','COCO2017')}[cond]
 content+=f'<h2>SAE는 {a}, 대응은 {b}로 학습</h2>'+table(rows)
content+='<!--CAPTION-END--><h2>이전 평가 기록</h2><p>아래 접은 영역은 연결된 이미지 주석을 텍스트에도 전달했던 이전 결과입니다. 위의 자체 캡션 주석 기준 매칭과 구분합니다. 전체 검색 조건의 결과는 그대로 보존했습니다.</p><details><summary>이전 매칭과 전체 검색 기록 펼치기</summary>'
old=old[:start]+content+old[end:];old=old.replace('</html>','</details></html>')
(O/'report.html').write_text(old);(O/'caption-results.json').write_text(json.dumps(dict(main=main,denominator=n,completed=evaluated,total=tot),ensure_ascii=False,indent=2));print(evaluated,tot,n)
