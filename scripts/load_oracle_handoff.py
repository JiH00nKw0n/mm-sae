"""Load the handoff without requiring the research repository or a GPU."""
import argparse
from pathlib import Path
import numpy as np
from scipy import sparse


def load_projection(condition_directory, target='propagated_presence', k=16, split='val2017'):
    directory=Path(condition_directory)
    with np.load(directory/'matrices'/f'{target}_{k}.npz',allow_pickle=False) as saved:
        model={key:saved[key].copy() for key in saved.files}
    projections={}
    for side,key in [('image','A'),('text','B')]:
        raw=sparse.load_npz(directory/f'activations/{split}/{side}.npz')
        ids=model[side+'_feature_ids']
        weights=model[key]/model[side+'_feature_scale'][:,None]
        projections[side]=np.asarray(raw[:,ids]@weights)-model[side+'_feature_mean']@weights
    parents=np.load(directory/f'index/{split}/parents.npy',allow_pickle=False)
    assert projections['text'].shape[0]==len(parents)
    assert projections['image'].shape[1]==projections['text'].shape[1]==len(model['concept_ids'])
    return projections,parents,model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('condition_directory',type=Path)
    parser.add_argument('--target',choices=['propagated_presence','caption_mentions'],default='propagated_presence')
    parser.add_argument('--k',default='16')
    args=parser.parse_args()
    scores,parents,model=load_projection(args.condition_directory,args.target,args.k)
    norm={side:value/np.maximum(np.linalg.norm(value,axis=1,keepdims=True),1e-12) for side,value in scores.items()}
    # The first image query. P is the 171 by 171 identity in these Oracle runs.
    similarity=norm['image'][0]@model['P']@norm['text'].T
    top=np.argsort(-similarity,kind='stable')[:10]
    print('A shape:',model['A'].shape,'B shape:',model['B'].shape,'P shape:',model['P'].shape)
    print('Image projections:',scores['image'].shape,'Text projections:',scores['text'].shape)
    print('Top caption row indices:',top.tolist())
    print('Correct parent image among top 10:',bool(np.any(parents[top]==0)))


if __name__=='__main__':main()
