from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/variation_v1'
def entry(i,title,kind,scenario=None):
    return dict(id=i,title=f'Stage {i} - {title}',kind=kind,directory=f'results/variation_v1/stage{i}',models=['analysis'],epochs=1,unit='analysis passes',scenario=scenario)
PLAN = [entry(218, 'paired trajectory uncertainty protocol', 'prepare'),
        entry(219, 'measure paired effects on validation trajectories', 'audit'),
        entry(220, 'report trajectory and scenario sensitivity', 'report')]
