"""Independent CPU checks and fixed-center / within-image wrong-phrase controls."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'results'
rows=json.loads((DATA/'records.json').read_text())
summary=json.loads((DATA/'summary.json').read_text())
METHODS=['patch_token','patch_phrase','local_token','local_phrase']
byimage=defaultdict(list)
for row in rows: byimage[row['image_id']].append(row)
center_patches=np.array([24,17,23,25,31])
checks=0
controls=[]
paired=[]

def evaluate(scores,target):
    selected=np.argsort(-scores,kind='stable')[:5]
    mask=np.zeros(49,bool);mask[selected]=True
    mask=np.repeat(np.repeat(mask.reshape(7,7),32,0),32,1)
    n=(mask&target).sum()
    return {'point_hit':float(target[16::32,16::32].reshape(-1)[np.argmax(scores)]),
            'precision':float(n/mask.sum()),'recall':float(n/target.sum()),
            'iou':float(n/(mask|target).sum())}

for row in rows:
    path=DATA/'cases'/str(row['image_id'])/f"{row['category_id']}.npz"
    arr=np.load(path)
    assert arr['target'].shape==(224,224)
    assert abs(arr['target'].mean()-row['area_fraction'])<1e-10
    for method in METHODS:
        checked=evaluate(arr[method],arr['target'])
        for metric,value in checked.items():
            assert abs(value-row['methods'][method][metric])<1e-9
        assert len(set(row['methods'][method]['selected_patches']))==5
        checks+=1
    # Predetermined central cross. No score or category information enters its choice.
    scores=np.zeros(49);scores[center_patches]=np.arange(5,0,-1)
    ctr=evaluate(scores,arr['target'])
    controls.append({'image_id':row['image_id'],'category_id':row['category_id'],**ctr})
    other=[r for r in byimage[row['image_id']] if r['category_id']!=row['category_id']]
    if other:
        wrong=[]
        for r in other:
            a=np.load(DATA/'cases'/str(r['image_id'])/f"{r['category_id']}.npz")
            wrong.append(evaluate(a['local_phrase'],arr['target']))
        paired.append({'image_id':row['image_id'],'category_id':row['category_id'],
                       'correct':{k:row['methods']['local_phrase'][k] for k in ctr},
                       'wrong_phrase':{k:float(np.mean([a[k] for a in wrong])) for k in ctr},
                       'center':ctr,'random':{k:row['random'][k] for k in ctr}})

result={'checked_method_mentions':checks,'all_metric_checks_passed':True,
        'center_patch_indices':center_patches.tolist(),'center':{},
        'within_image_comparison':{'mentions':len(paired),'images':len({r['image_id'] for r in paired}),'methods':{}},
        'paired_rows':paired}
result['local_phrase_minus_center']={}
for metric in ['point_hit','precision','recall']:
    grouped=defaultdict(list)
    for row,ctr in zip(rows,controls):
        grouped[row['image_id']].append(row['methods']['local_phrase'][metric]-ctr[metric])
    arr=np.array([(sum(v),len(v)) for v in grouped.values()])
    draw=np.random.default_rng(41).integers(len(arr),size=(2000,len(arr)))
    bootstrap=arr[draw,0].sum(1)/arr[draw,1].sum(1)
    result['local_phrase_minus_center'][metric]={'mean':float(arr[:,0].sum()/arr[:,1].sum()),
                                               'image_bootstrap_ci95':np.quantile(bootstrap,[.025,.975]).tolist()}
for metric in ['point_hit','precision','recall','iou']:
    result['center'][metric]={'mean':float(np.mean([r[metric] for r in controls])),'n':len(controls)}
for metric in summary['methods']['random']:
    if metric not in result['center']:
        result['center'][metric]={'mean':None,'n':0}
for name in ['correct','wrong_phrase','center','random']:
    result['within_image_comparison']['methods'][name]={k:float(np.mean([r[name][k] for r in paired]))
                                                        for k in ['point_hit','precision','recall','iou']}
cluster=defaultdict(list)
for r in paired: cluster[r['image_id']].append(r['correct']['precision']-r['wrong_phrase']['precision'])
a=np.array([(sum(v),len(v)) for v in cluster.values()])
idx=np.random.default_rng(55).integers(len(a),size=(2000,len(a)))
bs=a[idx,0].sum(1)/a[idx,1].sum(1)
result['within_image_comparison']['correct_minus_wrong_precision']={
    'mean':float(a[:,0].sum()/a[:,1].sum()),'image_bootstrap_ci95':np.quantile(bs,[.025,.975]).tolist()}
(ROOT/'audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='paired_rows'},indent=2))
