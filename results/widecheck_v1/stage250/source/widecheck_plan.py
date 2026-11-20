from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/widecheck_v1'
WINDOW=ROOT/'results/autonomous_17h_window.json'
def entry(i,title,kind,seed=42,holdout=None,vision=True):
    return dict(id=i,title=f'Stage {i} - {title}',kind=kind,directory=f'results/widecheck_v1/stage{i}',models=['fusion_ce' if kind=='train' else 'analysis'],epochs=12 if kind=='train' else 1,unit='epochs' if kind=='train' else 'analysis passes',seed=seed,holdout=holdout,k=40,bounded=False,objective='hard',penalty=0.,vision=vision,resolution=192,image_dropout=.5 if vision else 0.,consistency=.1 if vision else 0.)
PLAN = [entry(250, 'candidate width replication protocol', 'prepare')]
for seed in (7, 13):
    for held in (36, 37, 38, 39):
        for vision in (False, True):
            stage = 250 + len(PLAN)
            title = f'40 candidates, seed {seed}, excluded S{held}, '+('visual' if vision else 'GPS')
            PLAN.append(entry(stage, title, 'train', seed=seed, holdout=held, vision=vision))
PLAN.append(entry(267, 'three-seed candidate width comparison', 'report'))
