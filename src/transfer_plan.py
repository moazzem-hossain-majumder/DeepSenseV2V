from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/transfer_v1'
PREVIOUS=ROOT/'results/conservative_v1'
def entry(i,title,kind,holdout=None,vision=True):
    return dict(id=i,title=f'Stage {i} - {title}',kind=kind,directory=f'results/transfer_v1/stage{i}',models=['fusion_ce' if kind=='train' else 'analysis'],epochs=12 if kind=='train' else 1,unit='epochs' if kind=='train' else 'analysis passes',seed=42,holdout=holdout,k=20,bounded=False,objective='hard',penalty=0.,vision=vision,resolution=192)
PLAN=[entry(164,'prespecified four-scenario transfer protocol','prepare')]
for s in (36,37,38,39):
    for vision in (False,True):PLAN.append(entry(164+len(PLAN),f'exclude S{s} '+('192px visual' if vision else 'GPS'),'train',holdout=s,vision=vision))
PLAN.append(entry(173,'paired transfer comparison','report'))
