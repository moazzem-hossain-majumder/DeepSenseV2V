from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rescheck_v1'
PREVIOUS=ROOT/'results/conservative_v1'
def entry(i,title,kind,seed=42,resolution=96,vision=True):
    return dict(id=i,title=f'Stage {i} - {title}',kind=kind,directory=f'results/rescheck_v1/stage{i}',models=['fusion_ce' if kind=='train' else 'analysis'],epochs=12 if kind=='train' else 1,unit='epochs' if kind=='train' else 'analysis passes',seed=seed,holdout=None,k=20,bounded=False,objective='hard',penalty=0.,vision=vision,resolution=resolution)
PLAN=[entry(155,'matched resolution replication protocol','prepare')]
for seed in (42,7,13):
    PLAN.append(entry(155+len(PLAN),f'GPS control seed {seed}','train',seed=seed,vision=False))
    if seed!=42:
        for size in (96,192):PLAN.append(entry(155+len(PLAN),f'{size}px visual seed {seed}','train',seed=seed,resolution=size))
PLAN.append(entry(163,'three-seed resolution and GPS comparison','report'))
