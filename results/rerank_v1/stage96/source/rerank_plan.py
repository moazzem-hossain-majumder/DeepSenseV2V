from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rerank_v1'
PREVIOUS=ROOT/'results/conservative_v1'
def entry(i,title,kind,**kw):
    return dict(id=i,title=f'Stage {i} - {title}',kind=kind,directory=f'results/rerank_v1/stage{i}',models=['fusion_ce' if kind in ('pilot','expand') else 'analysis'],epochs=3 if kind=='pilot' else 8 if kind=='expand' else 1,unit='epochs' if kind in ('pilot','expand') else 'analysis passes',seed=42,holdout=None,**kw)
PLAN=[entry(96,'shortlist coverage and baseline protocol audit','coverage')]
for i,(k,vision) in enumerate(((10,False),(10,True),(20,False),(20,True)),97):
    PLAN.append(entry(i,f'top {k} '+('GPS and images' if vision else 'GPS control'),'pilot',k=k,vision=vision))
PLAN.append(entry(101,'matched pilot selection','gate'))
i=102
for seed in (42,7,13):
    for vision in (False,True):
        p=entry(i,f'seed {seed} '+('GPS and images' if vision else 'GPS control'),'expand',vision=vision);p['seed']=seed;PLAN.append(p);i+=1
for s in (36,37,38,39):
    for vision in (False,True):
        p=entry(i,f'held S{s} '+('GPS and images' if vision else 'GPS control'),'expand',vision=vision);p['holdout']=s;PLAN.append(p);i+=1
PLAN.append(entry(116,'paired reranking evidence report','report'))
