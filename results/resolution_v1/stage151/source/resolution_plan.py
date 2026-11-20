from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/resolution_v1'
PREVIOUS=ROOT/'results/conservative_v1'
CACHE=ROOT/'data/processed/resolution_v1'
WINDOW=ROOT/'results/autonomous_5h_window.json'
def entry(i,title,kind,**kw):
    return dict(id=i,title=f'Stage {i} - {title}',kind=kind,directory=f'results/resolution_v1/stage{i}',models=['fusion_ce' if kind=='train' else 'analysis'],epochs=12 if kind=='train' else 1,unit='epochs' if kind=='train' else 'analysis passes',seed=42,holdout=None,k=20,bounded=False,objective='hard',penalty=0.,vision=True,**kw)
PLAN=[entry(151,'native 96/192 frozen feature extraction','prepare'),entry(152,'96-pixel feature control','train',resolution=96),entry(153,'192-pixel feature comparison','train',resolution=192),entry(154,'resolution comparison report','report')]
