from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/failure_v1'
def entry(i,title,kind,scenario=None):
    return dict(id=i,title=f'Stage {i} - {title}',kind=kind,directory=f'results/failure_v1/stage{i}',models=['analysis'],epochs=1,unit='analysis passes',scenario=scenario)
PLAN=[entry(174,'transfer failure audit protocol','prepare')]+[entry(175+i,f'S{s} shift and correction audit','audit',s) for i,s in enumerate((36,37,38,39))]+[entry(179,'failure synthesis and next experiment','report')]
