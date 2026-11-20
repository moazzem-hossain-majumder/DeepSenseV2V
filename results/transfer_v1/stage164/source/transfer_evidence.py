import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from src.transfer_plan import ROOT,PREVIOUS,PLAN
from src.publication_plan import CACHE
from src.publication_metrics import per_sample
from src.camera_evidence import macro_interval
from src.acquire_scenarios import save_json
def report(folder,progress):
    m=pd.read_json(CACHE/'manifest.jsonl',lines=True);records=[];all_ids=[];all_geo=[];all_gps=[];all_visual=[]
    for s in (36,37,38,39):
        with np.load(PREVIOUS/f'stage44/baseline_without{s}_unseen.npz') as a:
            ids=a['indices'];geo=per_sample(a['logits'],a['profiles'],a['labels'])['success'].astype(float)
        if not ((m.scenario.iloc[ids]==s)&(m.split.iloc[ids]=='val')).all():raise ValueError('Wrong held scenario rows')
        outcomes={};epochs={}
        for p in [p for p in PLAN if p['kind']=='train' and p['holdout']==s]:
            path=ROOT/p['directory'];r=json.loads((path/'result.json').read_text())
            with np.load(path/'val_predictions.npz') as a:
                if (m.scenario.iloc[a['indices']]==s).any():raise ValueError('Held scenario entered checkpoint selection')
            with np.load(path/'unseen_predictions.npz') as a:
                if not np.array_equal(ids,a['indices']):raise ValueError('Held prediction IDs differ')
                outcomes[p['vision']]=per_sample(a['logits'],a['profiles'],a['labels'])['success'].astype(float)
            epochs[str(p['vision'])]=r['best_epoch']
        gps,visual=outcomes[False],outcomes[True];scenarios=m.scenario.to_numpy()[ids];blocks=m.block_key.to_numpy()[ids]
        records.append(dict(scenario=s,gps=float(gps.mean()),visual=float(visual.mean()),geometry=float(geo.mean()),selected_epochs=epochs,interval_vs_gps=macro_interval(visual-gps,blocks,scenarios),interval_vs_geometry=macro_interval(visual-geo,blocks,scenarios)))
        all_ids.extend(ids);all_geo.extend(geo);all_gps.extend(gps);all_visual.extend(visual)
    ids=np.asarray(all_ids);visual=np.asarray(all_visual);gps=np.asarray(all_gps);geo=np.asarray(all_geo);scenarios=m.scenario.to_numpy()[ids];blocks=m.block_key.to_numpy()[ids]
    macro=lambda x:float(np.mean([x[scenarios==s].mean() for s in np.unique(scenarios)]))
    evidence=dict(publication_ready=False,calibration_test_used=False,records=records,means=dict(visual=macro(visual),gps=macro(gps),geometry=macro(geo)),interval_vs_gps=macro_interval(visual-gps,blocks,scenarios),interval_vs_geometry=macro_interval(visual-geo,blocks,scenarios),
        scope='Seed-42 leave-one-scenario-out development evaluation. Held-scenario validation labels excluded from fitting and checkpoint selection. These scenarios were previously explored in pooled and earlier transfer studies; this is not an untouched external test.',
        next_action='Review transfer consistency and failures before additional seeds or final protocol decisions. Do not tune on these held-scenario results and present them as independent confirmation.')
    save_json(folder/'evidence.json',evidence)
    lines=['# Four-scenario transfer comparison','',evidence['scope'],'']
    for r in records:lines.append(f"- Excluded S{r['scenario']}: visual {r['visual']:.2%}, GPS {r['gps']:.2%}, geometry {r['geometry']:.2%}.")
    lines+=['',f'Macro means: {evidence["means"]}',evidence['next_action'],'Calibration and final test remain reserved.']
    (folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    with (ROOT/'paper/EXPERIMENT_TRACKER.md').open('a',encoding='utf-8') as f:f.write(f'\nStage 173 completed {datetime.now(timezone.utc).isoformat()}; see results/transfer_v1/stage173/REPORT.md.\n')
