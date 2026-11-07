"""Frozen dog-coordinate intervention. Masks are evaluation-only, no fitting."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json,time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from mm_sae.models.sae import TopKSAE
from experiments import apbt_search_20261009 as p
REVISION='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
def transform_mask(processor,value):
 proc=processor.image_processor
 value=proc.resize(value[None],size=proc.size,resample=Image.Resampling.NEAREST,input_data_format='channels_first',data_format='channels_first')
 return proc.center_crop(value,size=proc.crop_size,input_data_format='channels_first',data_format='channels_first')[0]
ROOT=p.ROOT;OUT=ROOT/'runs/rank-candidate-masking-2026-10-10';OUT.mkdir(exist_ok=True)

@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(4)
 cfg=p.old.load_config(ROOT/p.CONFIG['cc3m-cc3m']);data,_=p.eval_population(cfg,'test',OUT)
 c=next(i for i,c in enumerate(data['concepts']) if c['name']=='dog');dogid=data['concepts'][c]['id'];coords=[9,297]
 idx=Path(cfg['source_run'])/'index/val2017';infos=json.loads((idx/'images.json').read_text());assert len(infos)==len(data['masks']['image_b'])
 selected=np.flatnonzero(data['masks']['image_b'] & data['labels']['image'][:,c]);assert len(selected)==79
 wp=ROOT/'runs/matched-statistic-rotation-2026-10-10/pls-512/0-transforms.npz'
 with np.load(wp) as z:w=z['image'][:,coords]/data['scales']['image'][:,None]
 offset=data['means']['image']@w;cached=np.asarray(data['raw']['image']@w)-offset;sd=cached.std(0)
 p.save(OUT/'mask-protocol.json',dict(condition='CC3M SAE / CC3M mapping, PLS512 + covariance rotation, seed0',coordinates=coords,category='dog',category_id=dogid,selection='Previously selected on image half A; all79 dog-positive images in halfB. Exploratory follow-up to selected example, not independent confirmation across categories.',interventions=['remove dog union','keep dog only','remove person union if present','remove couch union if present'],fill='zero in normalized CLIP pixels, equivalent to CLIP channel mean',controls='For each removed region, 5 uniformly random toroidal translations of the exact mask with zero overlap with dog and the removed region; try at most500 shifts. Same area and pixel mask shape up to boundary wrap. If not possible, leave control missing and report paired subset. Keep-dog-only has no equivalent outside-dog control.',measurement='Original minus changed coordinate output divided by original coordinate standard deviation across5000 val images. No refitting or normalization after masking.',limits='Masking changes image distribution and may remove interactions. Context dependence is not proof of meaningless coordinate. Only dog / two preselected coordinates.'))
 model=CLIPModel.from_pretrained('openai/clip-vit-base-patch32',revision=REVISION,local_files_only=True).eval().cuda()
 proc=CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32',revision=REVISION,local_files_only=True,use_fast=False)
 sae=TopKSAE.from_pretrained(ROOT/'runs/cc3m-sae-2026-10-04/models/image').eval().cuda()
 wt=torch.tensor(w,dtype=torch.float32,device='cuda');ids=torch.tensor(data['feature_ids']['image'],device='cuda');off=torch.tensor(offset,dtype=torch.float32,device='cuda')
 def encode(pixels):
  emb=F.normalize(model.get_image_features(pixel_values=pixels),dim=-1)
  values,indices=sae.encode(emb);dense=torch.zeros((len(emb),sae.config.latent_size),device='cuda').scatter_(1,indices,values)
  return (dense[:,ids]@wt-off).cpu().numpy()
 records=[];maxerr=0.;examples=OUT/'mask-examples';examples.mkdir(exist_ok=True)
 person=next(c['id'] for c in data['concepts'] if c['name']=='person');couch=next(c['id'] for c in data['concepts'] if c['name']=='couch')
 exampleids={331075,182805,241326,90003}
 for no,rowidx in enumerate(selected):
  inf=infos[rowidx];photo=Image.open(inf['image']).convert('RGB');pixels=proc(images=photo,return_tensors='pt').pixel_values.cuda()
  label=transform_mask(proc,np.asarray(Image.open(inf['mask'])));dog=label==dogid;assert dog.any()
  masks=[np.zeros_like(dog),dog,~dog];names=['original','remove_dog','keep_dog']
  for name,cat in [('remove_person',person),('remove_couch',couch)]:
   if (label==cat).any():names.append(name);masks.append(label==cat)
  base=list(zip(names[1:],masks[1:]));controlkeys={};rng=np.random.default_rng(101010+int(inf['image_id']))
  for name,mask in base:
   if name=='keep_dog':continue
   forbidden=dog|mask;controls=[];shifts=set()
   if mask.sum()<=np.count_nonzero(~forbidden):
    for attempt in range(500):
     shift=tuple(rng.integers(0,224,2).tolist())
     if shift in shifts:continue
     shifts.add(shift);other=np.roll(mask,shift,(0,1))
     if not (other&forbidden).any():
      cname=f'{name}_control{len(controls)}';names.append(cname);masks.append(other);controls.append(cname)
      if len(controls)==5:break
   controlkeys[name]=controls
  batch=pixels.repeat(len(masks),1,1,1)
  for j,mask in enumerate(masks):batch[j,:,torch.tensor(mask,device='cuda')]=0
  scores=encode(batch);err=float(np.max(np.abs(scores[0]-cached[rowidx])));maxerr=max(maxerr,err)
  if maxerr>1e-3:raise RuntimeError(f'Original score mismatch with cached scores {maxerr}')
  vals={name:dict(score=scores[j].tolist(),drop_sd=((scores[0]-scores[j])/sd).tolist(),mask_fraction=float(masks[j].mean())) for j,name in enumerate(names)}
  records.append(dict(image_id=inf['image_id'],row_index=int(rowidx),values=vals,controls=controlkeys))
  if inf['image_id'] in exampleids:
   folder=examples/str(inf['image_id']);folder.mkdir(exist_ok=True)
   mean=torch.tensor(proc.image_processor.image_mean,device='cuda')[:,None,None];scale=torch.tensor(proc.image_processor.image_std,device='cuda')[:,None,None]
   for j,name in enumerate(names):
    if '_control' in name and not name.endswith('0'):continue
    rgb=((batch[j]*scale+mean).clamp(0,1)*255).byte().permute(1,2,0).cpu().numpy();Image.fromarray(rgb).save(folder/f'{name}.png')
  if no%10==0 or no==len(selected)-1:
   p.save(OUT/'mask-progress.json',dict(images=no+1,total=len(selected),seconds=time.time()-start,cache_max_abs_error=maxerr));p.save(OUT/'mask-records.json',records);print(no+1,len(selected),time.time()-start,flush=True)
 p.save(OUT/'mask-complete.json',dict(seconds=time.time()-start,cache_max_abs_error=maxerr,coordinate_stds=sd.tolist(),images=len(records)))
if __name__=='__main__':main()
