"""Predeclared finite experimental plan. Changing it requires a new version."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/publication_v1'
CACHE=ROOT/'data/processed/publication_v1'
VARIANTS={'gps':'B1','rgb':'B2','fusion_ce':'P1','profile_original':'P3','power_aware':'P3'}
SEEDS=(42,7,13)
EPOCHS=8
COMMON=dict(size=96,epochs=EPOCHS,batch=16,workers=2,lr=1e-4,weight_decay=1e-3,
            dropout=.25,freeze_batchnorm=True,brightness=.1,precision='bf16',
            primary='macro_scenario_within_3db_at_5',delta_db=3.,target_coverage=.9)


def stage(sid,title,kind,**extra):
    return dict(id=sid,title=title,kind=kind,directory=f'results/publication_v1/stage{sid}',
                models=[extra.get('variant','analysis')],epochs=EPOCHS if kind=='train' else 1,
                unit='epochs' if kind=='train' else 'analysis passes',job='publication',**extra)


PLAN=[stage(8,'Stage 8 - protocol and cache','prepare'),stage(9,'Stage 9 - audit and GPS baseline','audit')]
sid=10
for variant in VARIANTS:
    for seed in SEEDS:
        PLAN.append(stage(sid,f'Stage {sid} - {variant} seed {seed}','train',variant=variant,seed=seed,holdout=None))
        sid+=1
for scenario in (36,37,38,39):
    for variant in ('fusion_ce','power_aware'):
        PLAN.append(stage(sid,f'Stage {sid} - {variant} unseen S{scenario}','train',variant=variant,seed=42,holdout=scenario))
        sid+=1
PLAN += [stage(33,'Stage 33 - evidence and freeze gate','gate'),
         stage(34,'Stage 34 - calibration (gated)','calibrate'),
         stage(35,'Stage 35 - locked test (gated)','test'),
         stage(36,'Stage 36 - research evidence package','report')]


def config(plan):
    return {**COMMON,**{k:plan[k] for k in ('variant','seed','holdout')}}
