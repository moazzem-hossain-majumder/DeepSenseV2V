from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/diagnostic_v1'
PREVIOUS=ROOT/'results/conservative_v1'

def entry(i,title,kind,source=None):
    return dict(id=i,title=f'Stage {i} - {title}',kind=kind,source=source,
        directory=f'results/diagnostic_v1/stage{i}',models=['analysis'],epochs=1,unit='analysis passes')

PLAN=[entry(62,'historical input and timestamp audit','alignment'),entry(63,'scenario and trajectory failure audit','failures')]
for i,source in enumerate((49,50,51,53,55,57,59),64):
    PLAN.append(entry(i,f'image intervention for checkpoint {source}','intervention',source))
PLAN.append(entry(71,'diagnostic decision and next experiment specification','report'))
