from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/sensitivity_v1'
PREVIOUS=ROOT/'results/conservative_v1'
WINDOW=ROOT/'results/autonomous_5h_window.json'
def entry(i,title,kind,**kw):
    return dict(id=i,title=f'Stage {i} - {title}',kind=kind,directory=f'results/sensitivity_v1/stage{i}',models=['analysis'],epochs=1,unit='analysis passes',**kw)
PLAN=[entry(146,'fixed-checkpoint visual dependence protocol','prepare')]
for i,seed,source,directory in [(147,42,138,'penalty'),(148,7,142,'replicate'),(149,13,144,'replicate')]:
    PLAN.append(entry(i,f'seed {seed} image interventions','audit',seed=seed,source=source,source_directory=f'results/{directory}_v1/stage{source}'))
PLAN.append(entry(150,'visual dependence report','report'))
