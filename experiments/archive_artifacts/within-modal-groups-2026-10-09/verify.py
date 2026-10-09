"""Independent saved-weight constraints and representative-index recalculation."""
from pathlib import Path
import json
import hashlib
import numpy as np

root=Path(__file__).resolve().parent
data=root/'results'
checks=[]
for condition in ['coco-coco','cc3m-coco','cc3m-cc3m']:
    base=data/condition
    assert (base/'complete.json').exists() and (base/'binary-control-complete.json').exists()
    with np.load(base/'split.npz') as z:
        assert len(z['fit_rows'])==32768 and len(z['tune_rows'])==2048
        assert not np.intersect1d(z['fit_rows'],z['tune_rows']).size
        assert len(np.unique(z['fit_rows']))==32768
    for model in ['cca_support_control','correlation_groups','binary_correlation_groups','conditional_groups']:
        with np.load(base/(model+'.npz')) as z:
            permutation=z['permutation']
            np.testing.assert_array_equal(np.sort(permutation),np.arange(256))
            assert not bool(z['centered'])
            for side in ['image','text']:
                w=z[side]
                assert np.isfinite(w).all() and (w>=0).all()
                assert ((w!=0).sum(0)<=16).all()
                np.testing.assert_allclose(np.linalg.norm(w,axis=0),1,atol=2e-5)
                assert (w[~z[side+'_support']]==0).all()
                checks.append(dict(condition=condition,model=model,side=side,
                     min_nonzero=int((w!=0).sum(0).min()),max_nonzero=int((w!=0).sum(0).max())))
    for file in base.glob('*-test.json'):
        r=json.loads(file.read_text());assert r['denominator']==42
        with np.load(base/(file.stem+'-auc.npz')) as z:
            ia,ta=z['image_a'],z['text_b']
            valid=z['image_a_variable']&z['text_b_variable']
            eligible=z['eligible'];assert int(eligible.sum())==42
            for signed in [False,True]:
                agreed=0
                for c in np.flatnonzero(eligible):
                    good=valid & np.isfinite(ia[:,c]) & np.isfinite(ta[:,c])
                    selected=[];directions=[]
                    for auc in [ia[:,c],ta[:,c]]:
                        scores=np.maximum(auc,1-auc) if signed else auc
                        maximum=scores[good].max()
                        index=np.flatnonzero(good&(maximum-scores<=4*np.finfo(float).eps))[0]
                        selected.append(index);directions.append(-1 if signed and auc[index]<.5 else 1)
                    agreed += selected[0]==selected[1] and directions[0]==directions[1]
                key='signed' if signed else 'positive'
                assert agreed==r[key]['summary']['agree_at1_count'],(file,key,agreed)
                checks.append(dict(condition=condition,model=r['method'],metric=key,recomputed=int(agreed)))
        for direction,ret in r['retrieval'].items():
            values=[ret['recall'][str(k)] for k in [1,5,10]]
            assert 0<=values[0]<=values[1]<=values[2]<=1
    # Frozen matching population must be identical across all methods.
    cat_lists=[json.loads(p.read_text())['category_ids'] for p in base.glob('*-test.json')]
    assert len(cat_lists)==6 and all(x==cat_lists[0] for x in cat_lists)
manifest=json.loads((root/'remote-sha256.json').read_text())
for name,wanted in manifest.items():
    found=hashlib.sha256((data/name).read_bytes()).hexdigest()
    assert found==wanted,name
for condition in ['coco-coco','cc3m-coco','cc3m-cc3m']:
    protocol=json.loads((data/condition/'protocol.json').read_text())
    assert protocol['code_sha256']==hashlib.sha256((root/'run-initial.py').read_bytes()).hexdigest()
    protocol=json.loads((data/condition/'binary-control-protocol.json').read_text())
    assert protocol['code_sha256']==hashlib.sha256((root/'run.py').read_bytes()).hexdigest()
result=dict(passed=True,files_verified=len(manifest),checks=checks,
            scope='Artifact hashes, fitting split separation, saved support and coefficient constraints, permutation validity, independent re-computation of both matching counts, common category population. Retrieval kernel reused from previously validated evaluation.')
(root/'verification.json').write_text(json.dumps(result,indent=2))
print('Verified',len(manifest),'files and',len(checks),'checks.')
