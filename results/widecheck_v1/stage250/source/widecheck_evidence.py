"""Compare the two prespecified widths using identical seed and fold pairs."""
import numpy as np
import pandas as pd
from src.width_plan import PLAN as PILOT
from src.widecheck_plan import ROOT, PLAN
from src.anchor_analysis import source
from src.width_attribution import restriction,loss
from src.publication_plan import CACHE
from src.camera_evidence import macro_interval
from src.acquire_scenarios import save_json


def report(folder,progress):
    m=pd.read_json(CACHE/'manifest.jsonl',lines=True)
    rows=[]
    blocks_by_fold=[]
    scenarios=[]
    changes={name:[] for name in ('visual_width','gps_width','visual_vs_gps','visual_vs_geometry')}
    for held in (36,37,38,39):
        ids=m.index[m.scenario.eq(held)&m.split.eq('val')].to_numpy()
        diffs={key:[] for key in changes}
        with np.load(ROOT/f'results/conservative_v1/stage44/baseline_without{held}_unseen.npz') as a:
            assert np.array_equal(ids,a['indices'])
            geo,power=a['logits'],a['profiles']
        baseline=loss(power,np.argsort(-geo,axis=1,kind='stable')[:,:5]).astype(float)
        for seed in (42,7,13):
            values={}
            for vision in (False,True):
                plan=next(p for p in PILOT+PLAN if p['kind']=='train' and
                          (p['holdout'],p['seed'],p['vision'])==(held,seed,vision))
                arm='visual' if vision else 'gps'
                wide=ROOT/plan['directory']
                if (wide/'skipped.json').exists():continue
                for k,path in [(20,ROOT/source(seed,held,vision)),(40,wide)]:
                    with np.load(path/'unseen_predictions.npz') as a:
                        assert np.array_equal(a['indices'],ids) and np.array_equal(a['profiles'],power)
                        scores=a['logits']
                    if k==40:
                        top,restricted,used=restriction(geo,scores)
                        restricted_loss=loss(power,restricted).astype(float)
                    else:
                        top=np.argsort(-scores,axis=1,kind='stable')[:,:5]
                    values[arm,k]=loss(power,top).astype(float)
                old,new=values[arm,20],values[arm,40]
                rows.append(dict(scenario=held,seed=seed,arm=arm,success20=float((old<=3).mean()),
                    success40=float((new<=3).mean()),mean20=float(old.mean()),mean40=float(new.mean()),
                    geometry=float(baseline.mean()),geometry_success=float((baseline<=3).mean()),
                    used_fraction=float(used.mean()),access_benefit_db=float((restricted_loss-new).mean())))
            if len(values)==4:
                pairs=[(values['visual',40],values['visual',20]),(values['gps',40],values['gps',20]),
                       (values['visual',40],values['gps',40]),(values['visual',40],baseline)]
                for key,(a,b) in zip(changes,pairs):
                    diffs[key].append(np.stack([(a<=3).astype(float)-(b<=3),a-b]))
        if len(diffs['visual_width'])==3:
            for key in changes:changes[key].append(np.mean(diffs[key],axis=0))
            blocks_by_fold.extend(m.block_key.iloc[ids]);scenarios.extend([held]*len(ids))
    contrasts={}
    if len(changes['visual_width'])==4:
        sc=np.asarray(scenarios);blocks=np.asarray(blocks_by_fold)
        for key,entries in changes.items():
            contrasts[key]={}
            for metric,d in zip(('success','loss_db'),np.concatenate(entries,axis=1)):
                contrasts[key][metric]=dict(difference=float(np.mean([d[sc==s].mean() for s in np.unique(sc)])),
                                             interval=macro_interval(d,blocks,sc))
    save_json(folder/'evidence.json',dict(records=rows,contrasts=contrasts,scope='Exploratory transfer validation'))
    lines=['# Three-seed candidate width comparison','']
    for k,metrics in contrasts.items():
        lines.append(f"- {k}: success {100*metrics['success']['difference']:+.3f} points, mean power loss {metrics['loss_db']['difference']:+.3f} dB.")
    (folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
