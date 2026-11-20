import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from src.sensitivity_plan import ROOT,PLAN
from src.publication_plan import CACHE
from src.publication_metrics import per_sample
from src.camera_evidence import macro_interval
from src.acquire_scenarios import save_json
def report(folder,progress):
    m=pd.read_json(CACHE/'manifest.jsonl',lines=True);records=[];differences={'gray':[],'shuffled':[]}
    for p in PLAN:
        if p['kind']!='audit':continue
        path=ROOT/p['directory']
        if (path/'skipped.json').exists():continue
        record=json.loads((path/'interventions.json').read_text());record['seed']=p['seed'];records.append(record)
        with np.load(path/'original.npz') as a:
            ids=a['indices'];clean=per_sample(a['logits'],a['profiles'],a['labels'])['success'].astype(float)
        for mode in differences:
            with np.load(path/f'{mode}.npz') as a:
                if not np.array_equal(ids,a['indices']):raise ValueError('Intervention IDs differ')
                other=per_sample(a['logits'],a['profiles'],a['labels'])['success'].astype(float)
            differences[mode].append(clean-other)
    contrasts={}
    for mode,values in differences.items():
        if not values:continue
        delta=np.mean(values,axis=0);scenarios=m.scenario.to_numpy()[ids];blocks=m.block_key.to_numpy()[ids]
        contrasts[mode]=dict(original_minus_intervention=float(np.mean([delta[scenarios==s].mean() for s in np.unique(scenarios)])),interval=macro_interval(delta,blocks,scenarios))
    action='Review paired image-dependence contrasts before a separately controlled native-resolution feature experiment. Sensitivity alone is not evidence of useful visual reasoning or generalization.'
    save_json(folder/'evidence.json',dict(publication_ready=False,calibration_test_used=False,records=records,contrasts=contrasts,next_action=action))
    lines=['# Fixed-checkpoint visual dependence','', 'Validation-only interventions; no fitting or model selection. Gray is an out-of-distribution diagnostic; shuffling uses validation clips from the same scenario and preserves each five-frame sequence.','']
    for r in records:
        lines.append(f"- Seed {r['seed']}: "+', '.join(f"{mode} {v['metrics']['macro_success5']:.2%}" for mode,v in r['modes'].items()))
    for mode,c in contrasts.items():lines.append(f"- Original minus {mode}: {c['original_minus_intervention']*100:+.3f} percentage points; paired block interval {c['interval']} (fraction units).")
    lines+=['',action,'Repeated validation exploration and seed-42 pilot selection limit confirmatory claims.']
    (folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    with (ROOT/'paper/AUTONOMOUS_5H_TRACKER.md').open('a',encoding='utf-8') as f:f.write(f'\nQueue report at {datetime.now(timezone.utc).isoformat()}: Stage 150 complete. {action}\n')
