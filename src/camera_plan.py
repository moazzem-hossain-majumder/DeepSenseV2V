from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/camera_v1'
CACHE=ROOT/'data/processed/camera_v1'
PREVIOUS=ROOT/'results/conservative_v1'
def entry(i,title,kind,**kw):
    training=kind in ('pilot','expand')
    return dict(id=i,title=f'Stage {i} - {title}',kind=kind,directory=f'results/camera_v1/stage{i}',models=['fusion_ce' if training else 'analysis'],epochs=3 if kind=='pilot' else 8 if training else 1,unit='epochs' if training else 'analysis passes',seed=42,holdout=None,**kw)
PLAN=[entry(81,'rear image cache and source freeze','cache')]
for i,(camera,correction) in enumerate((('front','tanh_bf16'),('front','tanh_fp32'),('front','linear'),('rear','linear'),('both','linear')),82):
    PLAN.append(entry(i,f'{camera} {correction} pilot','pilot',camera=camera,correction=correction))
PLAN.append(entry(87,'learning health and pilot selection','gate'))
for i,seed in enumerate((42,7,13),88):
    p=entry(i,f'selected method seed {seed}','expand');p['seed']=seed;PLAN.append(p)
for i,s in enumerate((36,37,38,39),91):
    p=entry(i,f'selected method held scenario {s}','expand');p['holdout']=s;PLAN.append(p)
PLAN.append(entry(95,'camera comparison and evidence report','report'))
