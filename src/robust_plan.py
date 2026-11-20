from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/robust_v1'
def entry(i,title,kind,holdout=None,consistency=0.):
    return dict(id=i,title=f'Stage {i} - {title}',kind=kind,directory=f'results/robust_v1/stage{i}',models=['fusion_ce' if kind=='train' else 'analysis'],epochs=12 if kind=='train' else 1,unit='epochs' if kind=='train' else 'analysis passes',seed=42,holdout=holdout,k=20,bounded=False,objective='hard',penalty=0.,vision=True,resolution=192,image_dropout=.5,consistency=consistency)
PLAN=[entry(180,'fixed visual robustness protocol','prepare')]
for s in (36,37,38,39):
    for weight in (0.,.1):PLAN.append(entry(180+len(PLAN),f'exclude S{s} dropout'+(' plus consistency' if weight else ''),'train',s,weight))
PLAN.append(entry(189,'robustness comparison without held-label selection','report'))
