import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from src.resolution_plan import ROOT,PREVIOUS,PLAN
from src.publication_plan import CACHE
from src.publication_metrics import per_sample
from src.camera_evidence import macro_interval
from src.acquire_scenarios import save_json
def report(folder,progress):
    m=pd.read_json(CACHE/'manifest.jsonl',lines=True);records=[];outcomes={}
    with np.load(PREVIOUS/'stage44/baseline_pooled_val.npz') as a:
        ids=a['indices'];geo=per_sample(a['logits'],a['profiles'],a['labels'])['success'].astype(float)
    scenarios=m.scenario.to_numpy()[ids];blocks=m.block_key.to_numpy()[ids]
    for p in PLAN:
        if p['kind']!='train':continue
        path=ROOT/p['directory']
        if (path/'skipped.json').exists():continue
        r=json.loads((path/'result.json').read_text())
        with np.load(path/'val_predictions.npz') as a:
            if not np.array_equal(ids,a['indices']):raise ValueError('Prediction IDs differ')
            success=per_sample(a['logits'],a['profiles'],a['labels'])['success'].astype(float)
        outcomes[p['resolution']]=success
        records.append(dict(resolution=p['resolution'],score=r['val']['macro_success5'],best_epoch=r['best_epoch'],interval_vs_geometry=macro_interval(success-geo,blocks,scenarios)))
    contrast=None
    if len(outcomes)==2:
        delta=outcomes[192]-outcomes[96]
        contrast=dict(gain=float(np.mean([delta[scenarios==s].mean() for s in np.unique(scenarios)])),interval=macro_interval(delta,blocks,scenarios))
    action='Review paired resolution effect before seed replication or transfer. Both arms omit brightness augmentation and use cached frozen features, so compare them directly, not as an isolated resolution change from older training.'
    save_json(folder/'evidence.json',dict(publication_ready=False,calibration_test_used=False,records=records,contrast=contrast,next_action=action))
    lines=['# Native-image resolution comparison','',action,'']
    for r in records:lines.append(f"- {r['resolution']} pixels per view: {r['score']:.2%}, selected epoch {r['best_epoch']}.")
    if contrast:lines.append(f"192 minus 96: {contrast['gain']*100:+.3f} percentage points; paired block interval {contrast['interval']} (fraction units).")
    lines+=['','Single-seed exploratory comparison; no publication or SOTA claim.']
    (folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    with (ROOT/'paper/AUTONOMOUS_5H_TRACKER.md').open('a',encoding='utf-8') as f:f.write(f'\nQueue report at {datetime.now(timezone.utc).isoformat()}: Stage 154 complete. {action}\n')
