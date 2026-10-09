from pathlib import Path
import json
root=Path(__file__).resolve().parent
old=root.parent/'representative-agreement-2026-10-09/results'
rows=[]
for condition,sae,mapping in [('coco-coco','COCO2017','COCO2017'),('cc3m-cc3m','CC3M','CC3M'),('cc3m-coco','CC3M','COCO2017')]:
 for method in ['hungarian','sparse_cca_16']:
  d=json.loads((old/condition/'results'/f'test__{method}.json').read_text())
  assert d['n_evaluated_categories']==50
  assert sum(bool(v.get('agree_at1',False)) for v in d['per_category'])==d['summary']['agree_at1_count']
  rows.append(dict(sae=sae,mapping=mapping,method=method,evaluated=50,positive=d['positive_only_sensitivity']['summary']['agree_at1_count'],signed=d['summary']['agree_at1_count'],candidate_coordinates=d['n_candidate_coordinates']))
contr=root.parent/'contrastive-concepts-2026-10-09'
for method in ['nce_centered','nce_uncentered','nce_nonnegative']:
 vals=[]
 for dirname in ['results','results-seed1','results-seed2']:
  d=json.loads((contr/dirname/f'{method}-evaluation.json').read_text())
  vals.append([d['positive_agreement']['summary']['agree_at1_count'],d['signed_agreement']['summary']['agree_at1_count']])
 rows.append(dict(sae='CC3M',mapping='COCO2017',method=method,evaluated=50,positive=sum(v[0] for v in vals)/3,signed=sum(v[1] for v in vals)/3,per_seed=vals))
(root/'previous-agreement.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
print(json.dumps(rows,ensure_ascii=False,indent=2))
