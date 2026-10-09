"""Descriptive Bernoulli-exponential checks on fit images only, never a GOF p-value."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json
from pathlib import Path
import numpy as np
from scipy import sparse

root=Path('/mnt/working/mm-sae')
source=root/'runs/cc3m-followup-2026-10-05/cc3m-sae-coco-activations'
parent=root/'runs/cc3m-followup-2026-10-05/coco-fit/mapping'
out=root/'runs/contrastive-concepts-2026-10-09'
pop=json.loads((parent/'population.json').read_text())
records=json.loads((source/'index/train2017/images.json').read_text())
keep=np.isin([r['image_id'] for r in records],pop['fit_image_ids'])
parents=np.load(source/'index/train2017/parents.npy')
results={}
with np.load(parent/'moments.npz') as moments:
    for side,mask in [('image',keep),('text',keep[parents])]:
        z=sparse.load_npz(source/'activations/train2017'/f'{side}.npz').tocsr()[mask][:,moments[side+'_ids']].tocsc()
        rows=[]
        for j in range(z.shape[1]):
            v=z.data[z.indptr[j]:z.indptr[j+1]]
            v=v[v>0].astype(float)
            if len(v)<100:continue
            ordered=np.sort(v/v.mean())
            cdf=1-np.exp(-ordered)
            n=len(v)
            ks=max(np.max(np.arange(1,n+1)/n-cdf),np.max(cdf-np.arange(n)/n))
            rows.append(dict(feature_id=int(moments[side+'_ids'][j]),positive_count=n,
                positive_fraction=n/z.shape[0],positive_cv=float(v.std()/v.mean()),exponential_cdf_distance=float(ks)))
        results[side]=dict(n_samples=z.shape[0],n_coordinates=z.shape[1],n_with_100_positives=len(rows),
            median_positive_cv=float(np.median([r['positive_cv'] for r in rows])),
            median_exponential_cdf_distance=float(np.median([r['exponential_cdf_distance'] for r in rows])),
            fraction_cv_between_08_12=float(np.mean([.8<=r['positive_cv']<=1.2 for r in rows])),
            per_coordinate=rows)
results['interpretation']='Exponential positive amplitudes have population coefficient of variation 1. Distances use a fitted mean, so no standard KS p-value is reported. This does not test the cross-modal joint distribution.'
(out/'distribution.json').write_text(json.dumps(results,indent=2))
print(json.dumps({s:{k:v for k,v in results[s].items() if k!='per_coordinate'} for s in ('image','text')}))
