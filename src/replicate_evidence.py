import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from src.replicate_plan import ROOT,OUT,PREVIOUS,PLAN
from src.publication_plan import CACHE
from src.publication_metrics import per_sample
from src.camera_evidence import macro_interval
from src.acquire_scenarios import save_json

def report(folder,progress):
    m=pd.read_json(CACHE/'manifest.jsonl',lines=True);records=[];outcomes={}
    with np.load(PREVIOUS/'stage44/baseline_pooled_val.npz') as a:
        ids=a['indices'];geo=per_sample(a['logits'],a['profiles'],a['labels'])['success'].astype(float)
    scenarios=m.scenario.to_numpy()[ids];blocks=m.block_key.to_numpy()[ids]
    baseline=float(np.mean([geo[scenarios==s].mean() for s in np.unique(scenarios)]))
    runs=[dict(id=137,directory='results/penalty_v1/stage137',seed=42,vision=False),dict(id=138,directory='results/penalty_v1/stage138',seed=42,vision=True)]+[p for p in PLAN if p['kind']=='train']
    for p in runs:
        path=ROOT/p['directory']
        if (path/'skipped.json').exists():continue
        r=json.loads((path/'result.json').read_text())
        with np.load(path/'val_predictions.npz') as a:
            if not np.array_equal(ids,a['indices']):raise ValueError('Prediction IDs differ')
            success=per_sample(a['logits'],a['profiles'],a['labels'])['success'].astype(float)
        outcomes[p['seed'],p['vision']]=success
        records.append(dict(stage=p['id'],seed=p['seed'],vision=p['vision'],score=r['val']['macro_success5'],best_epoch=r['best_epoch'],by_scenario={str(s):float(success[scenarios==s].mean()) for s in np.unique(scenarios)}))
    seeds=[s for s in (42,7,13) if (s,False) in outcomes and (s,True) in outcomes]
    visual=np.mean([outcomes[s,True] for s in seeds],axis=0);gps=np.mean([outcomes[s,False] for s in seeds],axis=0)
    macro=lambda a:float(np.mean([a[scenarios==s].mean() for s in np.unique(scenarios)]))
    pairs=[dict(seed=s,gps=macro(outcomes[s,False]),visual=macro(outcomes[s,True]),gain=macro(outcomes[s,True]-outcomes[s,False])) for s in seeds]
    ci=macro_interval(visual-gps,blocks,scenarios);geo_ci=macro_interval(visual-geo,blocks,scenarios)
    supported=len(seeds)==3 and ci[0]>0 and geo_ci[0]>0 and all(p['gain']>0 for p in pairs)
    action='Proceed to a prespecified transfer and image-dependence audit; this remains exploratory validation evidence.' if supported else 'Replication is not sufficient for a robust visual-gain claim. Inspect per-seed/scenario patterns before a separately controlled image representation experiment.'
    evidence=dict(publication_ready=False,calibration_test_used=False,seeds=seeds,baseline=baseline,records=records,pairs=pairs,visual_mean=macro(visual),gps_mean=macro(gps),interval_vs_gps=ci,interval_vs_geometry=geo_ci,replication_supported=supported,next_action=action,
        uncertainty_scope='Paired scenario-stratified block bootstrap of seed-averaged success; does not account fully for seed uncertainty, repeated validation selection, or selection of seed 42 during pilot screening.',
        by_scenario={str(s):dict(visual=float(visual[scenarios==s].mean()),gps=float(gps[scenarios==s].mean()),geometry=float(geo[scenarios==s].mean())) for s in np.unique(scenarios)})
    save_json(folder/'evidence.json',evidence)
    lines=['# Matched seed replication','',f'Geometry {baseline:.2%}; visual mean {macro(visual):.2%}; GPS mean {macro(gps):.2%}.','']
    for p in pairs:lines.append(f"- Seed {p['seed']}: visual {p['visual']:.2%}, GPS {p['gps']:.2%}, difference {p['gain']*100:+.3f} percentage points.")
    lines+=['',f'Paired macro block-bootstrap interval vs GPS: [{ci[0]*100:.3f}, {ci[1]*100:.3f}] percentage points.',action,'',evidence['uncertainty_scope'],'Calibration and test remain reserved. No publication-readiness or SOTA claim.']
    (folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    with (ROOT/'paper/AUTONOMOUS_5H_TRACKER.md').open('a',encoding='utf-8') as f:f.write(f'\nQueue report at {datetime.now(timezone.utc).isoformat()}: Stage 145 complete. {action}\n')
