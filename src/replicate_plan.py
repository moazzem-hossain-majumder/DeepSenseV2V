from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/replicate_v1'
PREVIOUS=ROOT/'results/conservative_v1'
WINDOW=ROOT/'results/autonomous_5h_window.json'

def entry(i,title,kind,seed=42,**kw):
    return dict(id=i,title=f'Stage {i} - {title}',kind=kind,directory=f'results/replicate_v1/stage{i}',models=['fusion_ce' if kind=='train' else 'analysis'],epochs=12 if kind=='train' else 1,unit='epochs' if kind=='train' else 'analysis passes',seed=seed,holdout=None,k=20,bounded=False,objective='hard',penalty=0.,**kw)

PLAN=[entry(140,'matched seed replication protocol','prepare')]
for seed in (7,13):
    for vision in (False,True):
        i=140+len(PLAN)
        PLAN.append(entry(i,f'seed {seed} '+('visual' if vision else 'GPS'),'train',seed=seed,vision=vision))
PLAN.append(entry(145,'three-seed matched comparison','report'))
