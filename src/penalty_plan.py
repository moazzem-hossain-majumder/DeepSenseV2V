from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/penalty_v1'
PREVIOUS=ROOT/'results/conservative_v1'
WINDOW=ROOT/'results/autonomous_5h_window.json'

def entry(i,title,kind,**kw):
    return dict(id=i,title=f'Stage {i} - {title}',kind=kind,directory=f'results/penalty_v1/stage{i}',models=['fusion_ce' if kind=='train' else 'analysis'],epochs=12 if kind=='train' else 1,unit='epochs' if kind=='train' else 'analysis passes',seed=42,holdout=None,k=20,bounded=False,objective='hard',**kw)

PLAN=[entry(134,'penalty and convergence protocol','prepare')]
for penalty in (.1,0.):
    for vision in (False,True):
        i=135+len(PLAN)-1
        PLAN.append(entry(i,f'penalty {penalty} '+('visual' if vision else 'GPS'),'train',penalty=penalty,vision=vision))
PLAN.append(entry(139,'penalty comparison and next action','report'))
