from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/factorial_v1'
PREVIOUS=ROOT/'results/conservative_v1'
WINDOW=ROOT/'results/autonomous_5h_window.json'
def entry(i,title,kind,**kw):
    return dict(id=i,title=f'Stage {i} - {title}',kind=kind,directory=f'results/factorial_v1/stage{i}',models=['fusion_ce' if kind=='train' else 'analysis'],epochs=3 if kind=='train' else 1,unit='epochs' if kind=='train' else 'analysis passes',seed=42,holdout=None,k=20,**kw)
PLAN=[entry(124,'factorial protocol and five-hour window','prepare')]
i=125
for bounded in (True,False):
    for objective in ('soft','hard'):
        for vision in (False,True):
            PLAN.append(entry(i,f'{"bounded" if bounded else "unbounded"} {objective} '+('visual' if vision else 'GPS'),'train',bounded=bounded,objective=objective,vision=vision));i+=1
PLAN.append(entry(133,'factorial comparison and next action','report'))
