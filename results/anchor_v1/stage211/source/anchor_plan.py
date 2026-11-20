from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/anchor_v1'
def entry(i,title,kind,scenario=None):
    return dict(id=i,title=f'Stage {i} - {title}',kind=kind,directory=f'results/anchor_v1/stage{i}',models=['analysis'],epochs=1,unit='analysis passes',scenario=scenario)
PLAN=[entry(211,'protected beam choices protocol','prepare')]
for sid, scenario in enumerate((None,36,37,38,39),212):
    PLAN.append(entry(sid,f'protected top three, scope {scenario or "pooled"}','audit',scenario))
PLAN.append(entry(217,'protected beam comparison','report'))
