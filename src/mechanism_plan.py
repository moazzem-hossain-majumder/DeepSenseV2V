from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/mechanism_v1'
PREVIOUS=ROOT/'results/conservative_v1'
def entry(i,title,kind,mode=None):
    return dict(id=i,title=f'Stage {i} - {title}',kind=kind,mode=mode,directory=f'results/mechanism_v1/stage{i}',models=['analysis'],epochs=1,unit='analysis passes')
PLAN=[entry(117,'training failure sample and visual sheets','sample')]
for i,mode in enumerate(('current_loss','bounded_ce','unbounded_ce','per_example_control'),118):
    PLAN.append(entry(i,mode+' memorization test','memorize',mode))
PLAN += [entry(122,'official baseline implementation checks','benchmark'),entry(123,'mechanism audit decision','report')]
