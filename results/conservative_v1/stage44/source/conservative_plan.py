from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/conservative_v1'
PARENT=ROOT/'results/publication_v1'


def entry(i,title,kind,method=None,seed=42,holdout=None):
    return dict(id=i,title=f'Stage {i} - {title}',kind=kind,method=method,seed=seed,holdout=holdout,
        directory=f'results/conservative_v1/stage{i}',models=['gps' if method=='compact_gps' else 'fusion_ce' if method else 'analysis'],
        epochs=8 if kind=='train' else 1,unit='epochs' if kind=='train' else 'analysis passes')


PLAN=[entry(44,'coordinates and geometry audit','geometry'),entry(45,'help-versus-harm analysis','diagnostics')]
i=46
for method in ('compact_gps','bounded_multimodal'):
    for seed in (42,7,13):
        PLAN.append(entry(i,f'{method} seed {seed}','train',method,seed));i+=1
for scenario in (36,37,38,39):
    for method in ('compact_gps','bounded_multimodal'):
        PLAN.append(entry(i,f'{method} unseen S{scenario}','train',method,42,scenario));i+=1
PLAN.extend([entry(60,'paired uncertainty and transfer','compare'),entry(61,'evidence report and figures','report')])
