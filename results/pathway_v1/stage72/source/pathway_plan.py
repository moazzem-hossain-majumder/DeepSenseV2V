from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/pathway_v1'
PREVIOUS=ROOT/'results/conservative_v1'
def entry(i,title,kind,source=None):
    return dict(id=i,title=f'Stage {i} - {title}',kind=kind,source=source,directory=f'results/pathway_v1/stage{i}',models=['analysis'],epochs=1,unit='analysis passes')
PLAN=[entry(72,'timestamp provenance and sequence offsets','timing')]
for i,source in enumerate((49,50,51,53,55,57,59),73):
    PLAN.append(entry(i,f'RGB gradient and saturation audit {source}','probe',source))
PLAN.append(entry(80,'learning-pathway evidence and experiment specification','report'))
