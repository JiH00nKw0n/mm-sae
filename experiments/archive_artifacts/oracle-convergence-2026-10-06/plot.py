import json,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/runs/oracle-losses-2026-10-06')
records=[]
for line in (root/'run.log').read_text().splitlines():
 try: d=json.loads(line)
 except (ValueError,TypeError): continue
 if d.get('state')=='fitting_ce' and 'training_loss' in d: records.append(d)
fig,axs=plt.subplots(2,2,figsize=(12,8))
rows=[]
for row,method in enumerate(['binary_ce','ce']):
 for col,condition in enumerate(['coco-coco','cc3m-coco']):
  ax=axs[row,col]
  inset=ax.inset_axes([.46,.30,.50,.40])
  for side,color in [('image','#D96969'),('text','#5087CE')]:
   audit=json.loads((root/condition/f'{method}_{side}.json').read_text())
   pts={int(d['function_evaluations']):d['training_loss'] for d in records if d['condition']==condition and d['method']==method and d['side']==side}
   pts[1]=audit['initial_loss']; pts[audit['function_evaluations']]=audit['final_loss']
   points=sorted(pts.items()); x,y=zip(*points)
   label=side.capitalize()+(' (gradient check passed)' if audit['stationary_at_1e_4'] else ' (gradient check not passed)')
   ax.plot(x,y,'o-',ms=3,lw=1.8,color=color,label=label)
   late=[(a,b) for a,b in points if a>=max(x)*.6]
   inset.plot(*zip(*late),'o-',ms=2,lw=1,color=color)
   for a,b in points: rows.append(dict(condition=condition,method=method,side=side,loss_evaluation=a,regularized_training_loss=b))
  inset.set_title('Last 40% of evaluations',fontsize=9)
  inset.tick_params(labelsize=7);inset.ticklabel_format(axis='y',style='plain',useOffset=False)
  ax.set_title(('Binary CE' if method=='binary_ce' else 'Soft-label CE')+' | SAE trained on '+('COCO' if col==0 else 'CC3M'))
  ax.set_xlabel('Loss evaluation count (not epochs)');ax.set_ylabel('Training loss + L2 penalty')
  ax.grid(alpha=.2);ax.legend(loc='upper right',fontsize=8)
fig.suptitle('Oracle mapping: recorded training loss',fontsize=17)
fig.text(.5,.015,'Recorded every 10 loss evaluations, plus initial/final values. Insets show late-stage changes. Mapping trained on COCO.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.04,1,.95])
fig.savefig(root/'training_loss_curves.png',dpi=180)
fig.savefig(root/'training_loss_curves.svg')
with (root/'training_loss_curves.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print(root/'training_loss_curves.png')
for c in ['coco-coco','cc3m-coco']:
 for m in ['binary_ce','ce']:
  for s in ['image','text']:
   a=json.loads((root/c/f'{m}_{s}.json').read_text());print(c,m,s,a['iterations'],a['function_evaluations'],a['gradient_inf'])
