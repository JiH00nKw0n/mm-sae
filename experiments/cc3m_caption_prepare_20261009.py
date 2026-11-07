import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
from types import SimpleNamespace
import json,numpy as np,pyarrow as pa,pyarrow.parquet as pq
from mm_sae.models.encoder import CLIPEncoder
ROOT=Path('/mnt/working/mm-sae');OUT=ROOT/'runs/cc3m-caption-concepts-2026-10-09';CACHE=ROOT/'cache/cc3m-clip-b32'
raw=pa.concat_tables([pq.read_table(p) for p in sorted((OUT/'source/unlabeled').glob('*.parquet'))]);assert raw.num_rows==3318333
permutation=np.random.RandomState(0).permutation(raw.num_rows)
manifest=json.loads((CACHE/'manifest.json').read_text());pop=json.loads((ROOT/'runs/mapping-suite-cc3m-2026-10-04/population.json').read_text());fitset=set(pop['fit_image_ids']);tuneset=set(pop['tune_image_ids']);samples=[];keys=[];rawrows=[];globalrows=[];offset=0;rng=np.random.default_rng(9409)
for part in manifest['parts']:
 records=json.loads((CACHE/part['keys_path']).read_text());local=int(rng.integers(len(records)));row=records[local];numeric=int(row['key']);orig=int(permutation[numeric]);samples.append(dict(shard=part['shard'],key=row['key'],original_row=orig,caption=raw['caption'][orig].as_py(),expected=np.load(CACHE/part['text_path'],mmap_mode='r')[local].copy()))
 for j,row in enumerate(records):
  ident=f"{manifest['config']['dataset_id']}:{part['shard']}:{row['key']}"
  if ident in fitset:keys.append(ident);rawrows.append(int(permutation[int(row['key'])]));globalrows.append(offset+j)
 offset+=len(records)
assert len(keys)==pop['fit_images'];assert not set(keys)&tuneset
encfg=json.loads((ROOT/'runs/cc3m-sae-2026-10-04/resolved-config.json').read_text())['encoder'];encoder=CLIPEncoder(SimpleNamespace(**encfg));actual=np.concatenate([encoder.texts([x['caption'] for x in samples[i:i+64]]) for i in range(0,len(samples),64)]);expected=np.stack([x['expected'] for x in samples]);delta=np.max(np.abs(actual-expected),axis=1);cos=(actual*expected).sum(1);assert cos.min()>.99999,(cos.min(),delta.max())
audit=dict(source='google-research-datasets/conceptual_captions unlabeled train',cached_source='pixparse/cc3m-wds',mapping='Original rows permuted with numpy RandomState(0).permutation(3318333); WDS numeric key indexes permutation.',verified_embedding_samples=len(samples),one_sample_per_shard=True,min_embedding_cosine=float(cos.min()),max_embedding_difference=float(delta.max()),fit_captions=len(keys),tune_captions_excluded=len(tuneset),all_selected_in_fit=True,samples=[{k:v for k,v in s.items() if k!='expected'} for s in samples])
(OUT/'alignment-verification.json').write_text(json.dumps(audit,indent=2))
table=pa.table(dict(image_id=keys,activation_row=globalrows,original_caption_row=rawrows,caption=raw['caption'].take(pa.array(rawrows))))
pq.write_table(table,OUT/'fit-captions.parquet',compression='zstd');print(json.dumps({k:v for k,v in audit.items() if k!='samples'}),flush=True)
