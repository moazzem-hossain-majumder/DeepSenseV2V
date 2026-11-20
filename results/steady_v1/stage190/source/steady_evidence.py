import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from src.steady_plan import ROOT,PLAN
from src.publication_plan import CACHE
from src.publication_metrics import per_sample
from src.camera_evidence import macro_interval
from src.acquire_scenarios import save_json

def report(folder,progress):
    m=pd.read_json(CACHE/'manifest.jsonl',lines=True);summaries={}
    def read(path,filename):
        with np.load(ROOT/path/filename) as a:return a['indices'],per_sample(a['logits'],a['profiles'],a['labels'])['success'].astype(float)
    for held in (None,36,37,38,39):
        scope='pooled' if held is None else f'without{held}';filename='val_predictions.npz' if held is None else 'unseen_predictions.npz'
        with np.load(ROOT/f'results/conservative_v1/stage44/baseline_{scope}_{"val" if held is None else "unseen"}.npz') as a:
            ids=a['indices'];geo=per_sample(a['logits'],a['profiles'],a['labels'])['success'].astype(float)
        sc=m.scenario.to_numpy()[ids];blocks=m.block_key.to_numpy()[ids]
        macro=lambda x:float(np.mean([x[sc==s].mean() for s in np.unique(sc)]))
        groups={'gps':[],'robust':[]};records=[]
        for seed in (42,7,13):
            if held is None:
                gps_path=f'results/rescheck_v1/stage{dict(zip((42,7,13),(156,157,160)))[seed]}'
                robust_path=next(p['directory'] for p in PLAN if p['kind']=='train' and p['seed']==seed and p['holdout'] is None)
            elif seed==42:
                gps_path=f'results/transfer_v1/stage{165+2*(held-36)}';robust_path=f'results/robust_v1/stage{182+2*(held-36)}'
            else:
                gps_path=next(p['directory'] for p in PLAN if p['kind']=='train' and p['seed']==seed and p['holdout']==held and not p['vision'])
                robust_path=next(p['directory'] for p in PLAN if p['kind']=='train' and p['seed']==seed and p['holdout']==held and p['vision'])
            if any((ROOT/path/'skipped.json').exists() for path in (gps_path,robust_path)):continue
            record={'seed':seed}
            for arm,path in [('gps',gps_path),('robust',robust_path)]:
                actual,outcome=read(path,filename)
                if not np.array_equal(ids,actual):raise ValueError('Prediction IDs differ')
                if held is not None:
                    val_ids,_=read(path,'val_predictions.npz')
                    if (m.scenario.iloc[val_ids]==held).any():raise ValueError('Held scenario used for selection')
                groups[arm].append(outcome);record[arm]=macro(outcome)
            records.append(record)
        if not records:summaries[scope]={'incomplete':True};continue
        robust=np.mean(groups['robust'],axis=0);gps=np.mean(groups['gps'],axis=0)
        summaries[scope]=dict(records=records,robust=macro(robust),gps=macro(gps),geometry=macro(geo),interval_vs_gps=macro_interval(robust-gps,blocks,sc),interval_vs_geometry=macro_interval(robust-geo,blocks,sc))
    action='Assess whether robustness preserves pooled visual gains and consistently improves transfer beyond geometry. Recovery to geometry is not evidence of added visual value. If it merely suppresses corrections, prioritize training-only inner-split analysis or a narrower supported paper claim rather than indefinite outer-validation tuning.'
    save_json(folder/'evidence.json',dict(publication_ready=False,calibration_test_used=False,scopes=summaries,next_action=action,scope='Exploratory selected validation; prior robustness screening and repeated use of scenarios prevent an independent-confirmation claim.'))
    lines=['# Robustness replication and pooled/transfer tradeoff','']
    for name,r in summaries.items():
        if 'robust' in r:lines.append(f"- {name}: robust {r['robust']:.2%}, GPS {r['gps']:.2%}, geometry {r['geometry']:.2%}, {len(r['records'])} seeds.")
    lines+=['',action,'Calibration/final test remain reserved.']
    (folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    with (ROOT/'paper/AUTONOMOUS_17H_TRACKER.md').open('a',encoding='utf-8') as f:f.write(f'\nStage 210 report completed {datetime.now(timezone.utc).isoformat()}. {action}\n')
