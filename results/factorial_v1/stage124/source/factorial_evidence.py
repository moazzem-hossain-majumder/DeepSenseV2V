import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from src.factorial_plan import ROOT,OUT,PREVIOUS,PLAN
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
    for p in PLAN:
        if p['kind']!='train':continue
        path=ROOT/p['directory']
        if (path/'skipped.json').exists():continue
        r=json.loads((path/'result.json').read_text())
        with np.load(path/'val_predictions.npz') as a:
            if not np.array_equal(ids,a['indices']):raise ValueError('Prediction IDs differ')
            success=per_sample(a['logits'],a['profiles'],a['labels'])['success'].astype(float)
        outcomes[p['id']]=success
        records.append(dict(stage=p['id'],bounded=p['bounded'],objective=p['objective'],vision=p['vision'],score=r['val']['macro_success5'],best_epoch=r['best_epoch'],
            by_scenario={str(s):float(success[scenarios==s].mean()) for s in np.unique(scenarios)},interval_vs_geometry=macro_interval(success-geo,blocks,scenarios)))
    pairs=[]
    for bounded in (True,False):
        for objective in ('soft','hard'):
            group={r['vision']:r for r in records if r['bounded']==bounded and r['objective']==objective}
            if len(group)==2:
                visual,gps=group[True],group[False];diff=visual['score']-gps['score']
                pairs.append(dict(bounded=bounded,objective=objective,visual_stage=visual['stage'],visual_score=visual['score'],gps_score=gps['score'],visual_gain=diff,
                    interval_vs_gps=macro_interval(outcomes[visual['stage']]-outcomes[gps['stage']],blocks,scenarios),
                    eligible=bool(visual['score']>=max(baseline,gps['score'])+.0025)))
    chosen=max((r for r in pairs if r['eligible']),key=lambda r:r['visual_score'],default=None)
    action='Run a matched multi-seed replication of the qualifying pair before claiming improvement.' if chosen else 'No pair meets the preliminary gain screen. Review training curves and per-scenario effects; consider a separately controlled resolution/feature test, not automatic test-set evaluation.'
    evidence=dict(publication_ready=False,baseline=baseline,records=records,pairs=pairs,chosen=chosen,next_action=action,calibration_test_used=False)
    save_json(folder/'evidence.json',evidence)
    lines=['# Controlled objective/constraint factorial','',f'Geometry: {baseline:.2%}. Single-seed, three-epoch exploratory validation.','']
    for r in records:lines.append(f"- Stage {r['stage']}: bounded={r['bounded']}, objective={r['objective']}, images={r['vision']}: {r['score']:.2%}, best epoch {r['best_epoch']}.")
    lines+=['',action,'','Identical regularization, shortlist and features within all pairs. Head arithmetic is FP32. Metrics above K=20 are not shortlist-contained claims. All test/calibration data remain reserved.']
    (folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    with (ROOT/'paper/AUTONOMOUS_5H_TRACKER.md').open('a',encoding='utf-8') as f:f.write(f"\nQueue report at {datetime.now(timezone.utc).isoformat()}: Stage 133 complete. {action} See results/factorial_v1/stage133/REPORT.md.\n")
