import json,csv,html,re
from pathlib import Path
from collections import defaultdict
R=Path('/Users/jihoonkwon/Desktop/projects/research/MM-SAE'); O=R/'artifacts/real-results-2026-10-09'; RUN=R/'mm-sae/runs'
conditions={'coco-coco':('COCO2017','COCO2017','mapping-ablation-2026-10-04'),'cc3m-cc3m':('CC3M','CC3M','mapping-ablation-cc3m-2026-10-04'),'cc3m-coco':('CC3M','COCO2017','cc3m-followup-2026-10-05/coco-fit/ablation')}
def load(p):return json.loads(p.read_text())
def metrics(d):
 if not isinstance(d,dict):return None
 d=d.get('retrieval',d)
 try:
  a=d['image_to_text'];b=d['text_to_image']
  if (a['query_count'],a['candidate_count'],b['query_count'],b['candidate_count'])!=(5000,25014,25014,5000):return None
  return [100*float(t['recall'][str(k)]) for t in [a,b] for k in [1,5,10]]
 except (KeyError,TypeError):return None

def label(key):
 names={'hungarian':'헝가리안','sinkhorn':'Sinkhorn','procrustes':'Procrustes','cross_svd':'PLS-SVD','cca':'CCA','sparse_cca':'Sparse CCA','sparse_pls':'Sparse PLS-SVD','sparse_transport':'연결 가중치가 희소한 수송','sparse_factorization':'희소 행렬 분해','greedy':'가장 높은 점수부터 선택','topk':'좌표별 상위 연결 선택','propagated_presence':'Oracle Ridge · 이미지 주석을 양쪽에 사용','caption_mentions':'Oracle Ridge · 캡션에 언급한 범주 사용','same_budget_many':'총 연결 수를 맞춘 다대다','global_top_edges':'전체에서 높은 연결 선택','partial_many':'일부 좌표 다대다','partial_one':'일부 좌표 일대일'}
 for k in sorted(names,key=len,reverse=True):
  if key==k or key.startswith(k+'_'):key=names[k]+key[len(k):];break
 for x,y in [('text_projected_to_image','텍스트를 이미지 좌표로 변환'),('image_projected_to_text','이미지를 텍스트 좌표로 변환'),('standardized','평균 제거 및 표준편차 보정'),('centered','평균 제거'),('raw','원래 값'),('common','양쪽 공통 좌표'),('renormalized','제거 후 다시 정규화'),('pruned','작은 계수 제거')]:key=key.replace(x,y)
 return key.replace('__',' · ').replace('_',' ')

agreement={}
for cond in conditions:
 paths=list((R/'artifacts/representative-agreement-2026-10-09/results'/cond/'results').glob('test__*.json'))
 if cond=='cc3m-cc3m':paths+=list((O/'server/cc3m-agreement').glob('test__*.json'))
 if cond=='cc3m-coco':paths+=list((RUN/'cc3m-followup-2026-10-05/coco-fit/representative-agreement/results').glob('test__*.json'))
 if cond=='coco-coco':paths+=list((RUN/'representative-agreement-2026-10-04/results').glob('test__*.json'))
 for p in paths:
  d=load(p)
  if d.get('n_evaluated_categories')!=50:continue
  pos=d.get('positive_only_sensitivity',{})
  agreement[cond,p.stem.removeprefix('test__')]=[pos.get('summary',{}).get('agree_at1_count'),d['summary']['agree_at1_count']]

main={}; allrows=[]
keys=['hungarian__standardized__text_projected_to_image','sinkhorn__standardized__text_projected_to_image','sinkhorn__standardized__image_projected_to_text','procrustes','cross_svd_256','cca_256']
for cond,(sae,mp,folder) in conditions.items():
 p=RUN/folder/'retrieval_summary.csv'; groups={}
 for d in csv.DictReader(p.open(encoding='utf-8-sig')):
  key=d['key'];groups.setdefault(key,[None]*6)
  start=0 if d['direction']=='image_to_text' else 3
  groups[key][start:start+3]=[100*float(d['recall_at_'+str(k)]) for k in [1,5,10]]
 rows=[]
 for key in keys:
  sem='hungarian' if key.startswith('hungarian') else key
  rows.append(dict(label=label(key),v=groups[key]+agreement.get((cond,sem),[None,None]),source=str(p),key=key))
 for method in ['sinkhorn_text','sinkhorn_image','pls','cca']:
  p=RUN/'sparsity-sweep-2026-10-06'/cond/'results'/f'{method}_16.json'
  if p.exists():
   names={'sinkhorn_text':'Sparse Sinkhorn · 텍스트 변환 · 最大16','sinkhorn_image':'Sparse Sinkhorn · 이미지 변환 · 最大16','pls':'Sparse PLS-SVD · 조합당 최대16','cca':'Sparse CCA · 조합당 최대16'}
   rows.append(dict(label=names[method].replace('最大','최대 '),v=metrics(load(p))+agreement.get((cond,'sparse_cca_16' if method=='cca' else method),[None,None]),source=str(p),key=method+'_16'))
 if cond!='cc3m-cc3m':
  folder='oracle-sets-2026-10-04' if cond=='coco-coco' else 'cc3m-followup-2026-10-05/coco-fit/oracle'
  # Locate the published annotation baseline, retaining its actual source.
  candidates=list((RUN/folder).glob('results/propagated_presence_16.json'))
  if not candidates and cond=='cc3m-coco':candidates=list((RUN/'cc3m-followup-2026-10-05/coco-fit').glob('annotation-sets/results/propagated_presence_16.json'))
  if candidates:
   p=candidates[0];v=metrics(load(p))
   if v:rows.append(dict(label='Oracle Ridge · 개념당 최대16 · 개념171개',v=v+[None,None],source=str(p),key='oracle_ridge_16'))
 main[cond]=rows
for d in load(R/'artifacts/contrastive-concepts-2026-10-09/summary.json'):
 if d['method']=='sparse_cca_16':continue
 main['cc3m-coco'].append(dict(label=d['label']+(' · 3회 평균' if d['repeats']==3 else ''),v=d['recall_mean']+d['semantic_mean'][:2],source=str(R/'artifacts/contrastive-concepts-2026-10-09/summary.json'),key=d['method']))

# Every successful full COCO retrieval result in relevant experiment directories is retained.
roots=[p for p in RUN.iterdir() if p.is_dir() and any(p.name.startswith(s) for s in ['mapping-','cc3m-followup','corpus-','matrix-comparison','oracle-','sinkhorn-fusion','sparse-pls','sparsity-sweep']) and not any(s in p.name for s in ['smoke','initial-scaling','pre-provenance'])]
roots += [R/'artifacts/oracle-pursuit-2026-10-06/server-backup/results',R/'artifacts/oracle-pursuit-2026-10-06/warm-all-losses-backup/results',O/'server']
for root in roots:
 for p in root.rglob('*.json'):
  if any(s in str(p) for s in ['node_modules','/candidates/','/transforms/']):continue
  try:d=load(p)
  except Exception:continue
  if not isinstance(d,dict):continue
  v=metrics(d)
  if v is None:continue
  rel=str(p.relative_to(root));full=str(p)
  if 'cc3m-coco' in full or '/coco-fit/' in full or 'warmstart-cc3m-coco' in full:cond='cc3m-coco'
  elif 'cc3m-cc3m' in full or '/cc3m-fit/' in full or ('cc3m' in root.name and 'followup' not in root.name):cond='cc3m-cc3m'
  elif root.name.startswith('corpus-'):
   cond='cc3m-coco' if 'fit_coco' in p.name else 'cc3m-cc3m'
  else:cond='coco-coco'
  key=d.get('key',p.stem);semkey=key
  group=root.name+'/'+str(p.parent.relative_to(root))
  allrows.append(dict(condition=cond,group=group,label=label(key),key=key,v=v+agreement.get((cond,semkey),[None,None]),source=full))
 for p in root.rglob('retrieval_summary.csv'):
  groups={}
  for d in csv.DictReader(p.open(encoding='utf-8-sig')):
   if int(d.get('query_count',0)) not in (5000,25014):continue
   key=d.get('key') or d.get('method','')+'__'+d.get('space','')
   groups.setdefault(key,([None]*6,d))
   ix=0 if d['direction']=='image_to_text' else 3
   groups[key][0][ix:ix+3]=[100*float(d['recall_at_'+str(k)]) for k in [1,5,10]]
  full=str(p)
  cond='cc3m-coco' if '/coco-fit/' in full else 'cc3m-cc3m' if 'cc3m' in full else 'coco-coco'
  for key,(v,d) in groups.items():
   if None in v:continue
   lab=d.get('method','')+' · '+d.get('condition_description','')+' · '+d.get('space_description','')
   allrows.append(dict(condition=cond,group=str(p.parent.relative_to(RUN)) if p.is_relative_to(RUN) else str(p.parent),label=lab if d.get('condition_description') else label(key),key=key,v=v+agreement.get((cond,key),[None,None]),source=full))
# Collapse only identical records within the same experiment group; never choose a favorable duplicate.
seen=set();unique=[]
for row in allrows:
 sig=(row['condition'],row['group'],row['key'],tuple(row['v']))
 if sig in seen:continue
 seen.add(sig);unique.append(row)
allrows=unique
heads=['방법 및 조건','이미지 질의 R@1','이미지 질의 R@5','이미지 질의 R@10','텍스트 질의 R@1','텍스트 질의 R@5','텍스트 질의 R@10','기존 이미지 주석 전달 /50<br>높은 활성만','기존 이미지 주석 전달 /50<br>반응 방향 허용']
def table(rows):
 ranks=[sorted(set(round(r['v'][j],2) for r in rows if r['v'][j] is not None),reverse=True) for j in range(8)]
 out=['<div class="scroll"><table><thead><tr>'+''.join('<th>'+s+'</th>' for s in heads)+'</tr></thead><tbody>']
 for r in rows:
  out.append('<tr><th>'+html.escape(r['label'])+'</th>')
  for j,v in enumerate(r['v']):
   t='미평가' if v is None else f'{v:.2f}'
   if v is not None:
    rank=ranks[j].index(round(v,2))
    if rank==0:t='<b>'+t+'</b>'
    elif rank==1:t='<u>'+t+'</u>'
   out.append('<td>'+t+'</td>')
  out.append('</tr>')
 out.append('</tbody></table></div>');return ''.join(out)
parts=['''<!doctype html><html lang="ko"><meta charset="utf-8"><title>실제 데이터 검색과 범주 매칭 결과</title><style>body{font-family:Arial,"Apple SD Gothic Neo",sans-serif;color:#20232a;max-width:1560px;margin:42px auto;padding:0 24px;line-height:1.65;background:#fafafa}h1{font-size:30px}h2{margin-top:44px;font-size:22px}p{max-width:1100px}table{border-collapse:collapse;width:100%;background:white;font-size:14px;font-variant-numeric:tabular-nums}th,td{padding:10px 9px;border-bottom:1px solid #ddd;text-align:right;white-space:nowrap}th:first-child{text-align:left;white-space:normal;min-width:265px;max-width:510px}thead th{background:#e9edf3;font-size:12px}b{font-weight:800;color:#850029}u{text-decoration-thickness:2px;text-underline-offset:4px}.scroll{overflow:auto}details{margin:18px 0}summary{cursor:pointer;font-weight:600}small{color:#555}a{color:#164b84}.notice{padding:18px;background:#fff0d7;border-left:4px solid #b57814}</style><h1>실제 데이터의 검색 성능과 범주 매칭</h1><div class="notice"><b>현재 아래 매칭 수치는 텍스트에도 연결 이미지의 주석을 사용한 이전 평가입니다.</b> 캡션이 실제로 해당 범주를 표현하는지 평가한 수치가 아닙니다. 이미지의 객체 주석과 캡션 자체의 범주 언급 주석을 각각 사용하여 모든 주요 방법을 재평가하고 있습니다. 새 평가에서는 평가 가능한 범주 수를 다시 계산합니다. 이전 수치는 참고 기록으로만 남겼습니다.</div><p>검색은 COCO val2017 이미지 5,000장과 캡션 25,014개로 평가했습니다. 이미지 질의는 이미지로 캡션을 찾으며, 텍스트 질의는 캡션으로 이미지를 찾습니다. Recall@K는 상위 K개 안에 정답이 하나 이상 있는 질의의 비율이며 수치는 %입니다.</p><p>매칭은 이미지와 텍스트에서 같은 범주를 가장 잘 구별하는 대표 좌표를 각각 고른 뒤, 그 두 좌표가 학습한 대응에서 짝을 이루는 범주의 수입니다. 평가 가능한 50개 범주를 사용했습니다. 높은 활성만 사용하는 평가와 낮은 활성도 개념을 나타낼 수 있도록 반응 방향을 허용하는 평가를 함께 표시했습니다. 후보 좌표 수가 방법마다 다르며, 이는 개념만을 순수하게 표현한다는 보장은 아닙니다.</p><p>각 표의 각 열에서 1등은 <b>굵게</b>, 2등은 <u>밑줄</u>로 표시했습니다. 소수 둘째 자리에서 같은 값은 공동 순위로 처리했습니다. 미평가는 동일한 50개 범주 평가 결과를 확보하지 않은 경우입니다. 검색만 평가한 방법이 많아 매칭 결과가 없는 행이 많습니다. 0개를 맞혔다는 뜻이 아닙니다. InfoNCE는 3회 평균이며 다른 주요 방법은 저장된 단일 결과입니다.</p><div class="notice">복원 목적 추가, 출력 활성 희소성 추가, FastICA는 합성 자료에서만 평가했습니다. 이 세 방법의 실제 데이터 검색 및 50개 범주 매칭은 미측정입니다. Multi-view ICA의 실제 데이터 결과도 없습니다. Oracle의 개념 예측 AUROC를 50개 범주 매칭으로 대체하지 않았습니다.</div>''']
for cond,rows in main.items():
 a,b,_=conditions[cond];parts.append(f'<h2>SAE는 {a}, 대응은 {b}로 학습</h2>'+table(rows))
parts.append('<h2>추가로 평가한 대표 좌표 매칭</h2><p>희소성 개수와 공통 좌표 수를 바꾼 경우까지 포함합니다. 위 표에 없는 검색 조건은 아래 전체 기록에서 확인할 수 있습니다.</p>')
for cond in conditions:
 rows=[]
 for (c,k),v in agreement.items():
  if c==cond:rows.append(dict(label=label(k),v=[None]*6+v))
 a,b,_=conditions[cond]
 parts.append(f'<h3>SAE {a}, 대응 {b}</h3>'+table(rows))
parts.append(f'<h2>전체 검색 평가 기록 {len(allrows)}개</h2><p>희소성 개수, 정규화, 공통 좌표 수, Sinkhorn 설정, Oracle 손실과 초기화 등을 바꾼 결과를 실험별로 묶었습니다. 같은 모델의 재평가 기록이 별도 실험 폴더에 있으면 남겼으므로 아래 개수는 서로 다른 방법 수가 아닙니다. 시험 실행, 합성 자료, 과거 수정 전 Oracle 기록은 제외했습니다. 각 표 내부에서 순위를 표시하며, 서로 다른 표의 굵은 값을 전체 1등으로 해석하지 않습니다.</p>')
groups=defaultdict(list)
for r in allrows:groups[r['condition'],r['group']].append(r)
for (cond,group),rows in sorted(groups.items()):
 a,b,_=conditions[cond]
 parts.append(f'<details><summary>SAE {a}, 대응 {b} · {html.escape(group)} · {len(rows)}개</summary>'+table(rows)+'<p><small>원본 결과 파일</small></p><ul>'+''.join('<li><small>'+html.escape(r['source'])+'</small></li>' for r in rows)+'</ul></details>')
parts.append('<p>원본 경로와 수치는 <a href="results.json">results.json</a>에 저장했습니다. 어떤 행에서도 합성 진단 점수나 개념 예측 AUROC를 매칭 개수로 사용하지 않았습니다.</p></html>')
(O/'report.html').write_text(''.join(parts));(O/'results.json').write_text(json.dumps(dict(main=main,all=allrows),ensure_ascii=False,indent=2))
for c,rows in main.items():
 print(c)
 for r in rows:print(r['label'],','.join('—' if v is None else f'{v:.2f}' for v in r['v']))
print('inventory',len(allrows),'groups',len(groups))
