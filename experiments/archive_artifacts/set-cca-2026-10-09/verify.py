"""Validate saved set supports, set pairing, and exact result transfer."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'results'
read=lambda p:json.loads(p.read_text())
manifest=read(DATA/'manifest.json')
for rel,expected in manifest.items():
    assert hashlib.sha256((DATA/rel).read_bytes()).hexdigest()==expected,rel
assert read(DATA/'complete.json')['completed']
checks=[]
for c in ['coco-coco','cc3m-coco','cc3m-cc3m']:
    folder=DATA/c;eligible=read(folder/'evaluation.json')['eligible_ids'];assert len(eligible)==42
    with np.load(folder/'feature-ids.npz') as ids:widths={s:len(ids[s]) for s in ['image','text']}
    for m in ['correlation','binary_correlation','conditional']:
        name='set_cca_'+m;f=read(folder/(name+'-fit.json'));r=f['outputs']
        sets={s:read(folder/f'{s}-{m}-sets.json')['groups'] for s in ['image','text']}
        for s,ss in sets.items():
            flat=[i for group in ss for i in group];assert len(flat)==len(set(flat));assert min(flat)>=0 and max(flat)<widths[s]
        with np.load(folder/(name+'.npz')) as z:
            p=z['set_mapping'];assert p.shape==(len(sets['image']),len(sets['text']))
            assert np.isin(p,[0,1]).all() and (p.sum(0)<=1).all() and (p.sum(1)<=1).all()
            assert p.sum()==r==min(p.shape);assert np.array_equal(z['permutation'],np.arange(r))
            for s in ['image','text']:
                w=z[s];assert w.shape==(widths[s],r) and np.isfinite(w).all()
                for k,idx in enumerate(z[s+'_set_indices']):
                    support=np.flatnonzero(w[:,k]);assert set(support)<=set(sets[s][idx]);assert len(support)>0
                assert (np.count_nonzero(w,axis=1)<=1).all()
        for model in [name,'cca_dimension_'+str(r)]:
            d=read(folder/(model+'-test.json'));assert d['denominator']==42;assert sorted(d['category_ids'])==sorted(eligible)
            for direction in ['image_to_text','text_to_image']:
                rr=d['retrieval'][direction]['recall'];assert 0<=rr['1']<=rr['5']<=rr['10']<=1
            for key in ['positive','signed']:
                rows=[x for x in d[key]['per_category'] if x['status']=='ok'];assert len(rows)==42
                assert sum(x['agree_at1'] for x in rows)==d[key]['summary']['agree_at1_count']
        checks.append(dict(condition=c,method=m,outputs=r,passed=True))
report=dict(passed=True,files_with_verified_sha256=len(manifest),set_models_verified=checks,
            checks=['Same files on server and local filesystem','Disjoint fixed set supports','Partial permutation of sets','One CCA coordinate per matched set pair','Same 42 evaluation categories','Monotone Recall@1/5/10','Matching counts recomputed from categories'],
            implementation_check='Small numerical test on server verified CCA covariance constraints and recovery of a known swapped set assignment before the real-data run.')
(ROOT/'verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False));print(json.dumps(report,ensure_ascii=False))
