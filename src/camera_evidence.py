import json
import numpy as np
import pandas as pd
from src.camera_plan import ROOT,OUT,PLAN,PREVIOUS
from src.publication_plan import CACHE
from src.publication_metrics import per_sample
from src.acquire_scenarios import save_json

def eligible(score,baseline,health):
    return health['finite'] and health['rgb_gradient_norm']>0 and health['gray_delta']>1e-6 and score>=baseline+.0025

def gate(folder,progress):
    from src.camera_training import health
    baseline=json.loads((PREVIOUS/'stage44/report.json').read_text())['configurations']['pooled']['selection_val']['macro_success5']
    records=[]
    for j,p in enumerate(x for x in PLAN if x['kind']=='pilot'):
        path=ROOT/p['directory'];h=health(p);r=json.loads((path/'result.json').read_text());score=r['val']['macro_success5']
        records.append(dict(stage=p['id'],camera=p['camera'],correction=p['correction'],score=score,health=h,eligible=eligible(score,baseline,h)))
        progress(status='checking_pilot_health',model='analysis',batch=j+1,batches=5)
    candidates=[r for r in records if r['eligible']];chosen=max(candidates,key=lambda r:r['score']) if candidates else None
    save_json(folder/'selection.json',dict(chosen=chosen,records=records,baseline=baseline,
        scope='Exploratory validation selection; gain threshold 0.25 percentage points, finite positive RGB gradients and gray sensitivity above 1e-6. No publication claim.'))

def macro_interval(delta,blocks,scenarios):
    rng=np.random.default_rng(8195);groups={s:np.unique(blocks[scenarios==s]) for s in np.unique(scenarios)}
    sums={b:delta[blocks==b].sum() for b in np.unique(blocks)};counts={b:int((blocks==b).sum()) for b in sums};draws=[]
    for _ in range(2000):
        means=[]
        for keys in groups.values():
            sampled=rng.choice(keys,len(keys),replace=True);means.append(sum(sums[b] for b in sampled)/sum(counts[b] for b in sampled))
        draws.append(np.mean(means))
    return np.quantile(draws,[.025,.975]).tolist()

def report(folder,progress):
    selection=json.loads((OUT/'stage87/selection.json').read_text());lines=['# Camera and correction study','',selection['scope'],'']
    for r in selection['records']:lines.append(f"- Pilot {r['stage']}: {r['camera']} / {r['correction']}: {r['score']:.2%}; expansion eligible: {r['eligible']}.")
    evidence=dict(publication_ready=False,calibration_test_used=False,pilot_selection=selection)
    if selection['chosen']:
        m=pd.read_json(CACHE/'manifest.jsonl',lines=True);scores=[];outcomes=[]
        with np.load(PREVIOUS/'stage44/baseline_pooled_val.npz') as a:
            ids=a['indices'];baseline=per_sample(a['logits'],a['profiles'],a['labels'])['success'].astype(float)
        for i in (88,89,90):
            scores.append(json.loads((OUT/f'stage{i}/result.json').read_text())['val']['macro_success5'])
            with np.load(OUT/f'stage{i}/val_predictions.npz') as a:
                if not np.array_equal(ids,a['indices']):raise ValueError('Window mismatch')
                outcomes.append(per_sample(a['logits'],a['profiles'],a['labels'])['success'].astype(float))
        interval=macro_interval(np.mean(outcomes,axis=0)-baseline,m.block_key.to_numpy()[ids],m.scenario.to_numpy()[ids])
        geo=json.loads((PREVIOUS/'stage44/report.json').read_text())['configurations'];transfer={}
        for i,s in enumerate((36,37,38,39),91):
            value=json.loads((OUT/f'stage{i}/unseen_report.json').read_text())['macro_success5']
            transfer[str(s)]=dict(model=value,geometry=geo[f'without{s}']['unseen_val']['macro_success5'])
        wins=sum(r['model']>r['geometry']+.0025 for r in transfer.values())
        evidence.update(mean_success=float(np.mean(scores)),seed_scores=scores,paired_macro_interval95=interval,transfer=transfer,
            development_screen_passed=bool(np.mean(scores)>selection['baseline']+.01 and interval[0]>0 and wins>=3))
        lines += ['',f"Three-seed mean: {np.mean(scores):.2%}; geometry {selection['baseline']:.2%}; paired macro interval: {interval}.",f'Held-scenario wins: {wins}/4.']
    else:lines+=['','No pilot passed the predeclared expansion screen; longer runs were skipped.']
    lines+=['','All results are development evidence. Camera/correction choice used pooled validation; subsequent held-scenario checks exclude their scenario during fitting and checkpoint selection but are not independent of architecture selection. Prior repeated validation inspection further limits confirmation claims.',
        'Physical timestamp conversion remains unresolved. No time shifts or calibration/test evaluation were performed. Novelty and exact published-baseline reproduction remain outstanding.']
    save_json(folder/'evidence.json',evidence);(folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
