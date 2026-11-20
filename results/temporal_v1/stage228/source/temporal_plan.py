from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/temporal_v1'
WINDOW=ROOT/'results/autonomous_17h_window.json'
def entry(i,title,kind,seed=42,holdout=None,vision=True):
    return dict(id=i,title=f'Stage {i} - {title}',kind=kind,directory=f'results/temporal_v1/stage{i}',models=['fusion_ce' if kind=='train' else 'analysis'],epochs=12 if kind=='train' else 1,unit='epochs' if kind=='train' else 'analysis passes',seed=seed,holdout=holdout,k=20,bounded=False,objective='hard',penalty=0.,vision=vision,resolution=192,image_dropout=.5 if vision else 0.,consistency=.1 if vision else 0.)
PLAN = [entry(228, 'temporal feature replication protocol', 'prepare')]
for seed in (7, 13):
    for scenario in (None, 36, 37, 38, 39):
        stage = 228 + len(PLAN)
        title = f'endpoint feature difference, seed {seed}, scope {scenario or "pooled"}'
        PLAN.append(entry(stage, title, 'train', seed=seed, holdout=scenario))
PLAN.append(entry(239, 'three-seed temporal feature report', 'report'))
