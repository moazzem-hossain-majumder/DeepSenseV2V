from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/steady_v1'
WINDOW=ROOT/'results/autonomous_17h_window.json'
def entry(i,title,kind,seed=42,holdout=None,vision=True):
    return dict(id=i,title=f'Stage {i} - {title}',kind=kind,directory=f'results/steady_v1/stage{i}',models=['fusion_ce' if kind=='train' else 'analysis'],epochs=12 if kind=='train' else 1,unit='epochs' if kind=='train' else 'analysis passes',seed=seed,holdout=holdout,k=20,bounded=False,objective='hard',penalty=0.,vision=vision,resolution=192,image_dropout=.5 if vision else 0.,consistency=.1 if vision else 0.)
PLAN=[entry(190,'robustness replication and tradeoff protocol','prepare')]
for seed in (42,7,13):PLAN.append(entry(190+len(PLAN),f'pooled consistency seed {seed}','train',seed=seed))
for seed in (7,13):
    for s in (36,37,38,39):
        for vision in (False,True):PLAN.append(entry(190+len(PLAN),f'exclude S{s} seed {seed} '+('consistency' if vision else 'GPS'),'train',seed,s,vision))
PLAN.append(entry(210,'pooled and transfer replication report','report'))
