import json
import re
import numpy as np
import pandas as pd
from src.diagnostic_plan import ROOT,OUT,PREVIOUS,PLAN
from src.publication_plan import CACHE
from src.publication_metrics import per_sample
from src.acquire_scenarios import save_json

def clock_seconds(value):
    match=re.search(r'(\d{2})-(\d{2})-(\d{2}\.\d+)',str(value))
    if not match:raise ValueError(f'No timestamp in {value}')
    h,m,s=map(float,match.groups());return 3600*h+60*m+s

def clock_delta(a,b):return (np.asarray(a)-np.asarray(b)+43200)%86400-43200

def distribution(values):
    a=np.asarray(values,dtype=float)
    if not len(a):return dict(n=0)
    if not np.isfinite(a).all():raise ValueError('Nonfinite diagnostic values')
    return dict(n=len(a),minimum=float(a.min()),p05=float(np.quantile(a,.05)),median=float(np.median(a)),p95=float(np.quantile(a,.95)),maximum=float(a.max()))

def alignment(folder,progress):
    m=pd.read_json(CACHE/'manifest.jsonl',lines=True);records={}
    for s in (36,37,38,39):
        f=pd.read_csv(ROOT/f'data/raw/scenario{s}/scenario{s}.csv')
        sub=m[(m.scenario==s)&m.split.isin(['train','val'])]
        x=np.stack(sub.x_indices);y=sub.y_index.to_numpy();rows=np.unique(x)
        if not (x[:,-1]<y).all():raise ValueError('Future row included in inputs')
        if not (np.diff(x,axis=1)>0).all():raise ValueError('Nonmonotonic input rows')
        seq=f.seq_index.to_numpy()
        if not (seq[x]==seq[y,None]).all():raise ValueError('Input crosses sequence')
        times=np.array([clock_seconds(v) for v in f.timestamp])
        offsets={}
        for col in ('unit1_rgb5','unit1_rgb6','unit1_gps1','unit2_gps1'):
            stamp=np.array([clock_seconds(v) for v in f[col].iloc[rows]])
            offsets[col]=distribution(clock_delta(times[rows],stamp))
        row_to_rgb={int(i):clock_seconds(v) for i,v in zip(rows,f.unit1_rgb5.iloc[rows])}
        last_rgb=np.array([row_to_rgb[int(i)] for i in x[:,-1]])
        actual_horizon=clock_delta(times[y],times[x[:,-1]])
        discrepancy=np.abs(actual_horizon-sub.horizon_seconds.to_numpy())
        if discrepancy.max()>1e-5:raise ValueError('Recorded target horizon disagrees with CSV')
        records[str(s)]=dict(windows=len(sub),input_rows=len(rows),row_minus_sensor_seconds=offsets,
            row_forecast_horizon=distribution(actual_horizon),target_minus_last_image_filename_seconds=distribution(clock_delta(times[y],last_rgb)),
            nonpositive_input_timestamp_steps=int((clock_delta(times[x[:,1:]],times[x[:,:-1]])<=0).sum()),
            reused_front_frame_fraction=float(1-f.unit1_rgb5.iloc[rows].nunique()/len(rows)))
        progress(status='auditing_alignment',model='analysis',batch=s-35,batches=4)
    save_json(folder/'alignment.json',dict(scenarios=records,scope='Train/validation historical input rows only. Filename clock offsets do not establish physical synchronization error; no row remapping performed.'))

def failure_summary(model,baseline,profiles,labels,mask):
    if not np.any(mask):return dict(n=0)
    model,baseline,profiles,labels=model[mask],baseline[mask],profiles[mask],labels[mask]
    count=len(labels);mask=np.ones(count,bool)
    a=per_sample(model,profiles,labels);b=per_sample(baseline,profiles,labels)
    return dict(n=int(mask.sum()),model_success=float(a['success'][mask].mean()),geometry_success=float(b['success'][mask].mean()),
        helped=int((a['success']&~b['success']&mask).sum()),harmed=int((b['success']&~a['success']&mask).sum()),
        model_regret_db=float(a['regret'][mask].mean()),geometry_regret_db=float(b['regret'][mask].mean()))

def failures(folder,progress):
    m=pd.read_json(CACHE/'manifest.jsonl',lines=True);context=np.load(PREVIOUS/'stage44/context.npy');result={}
    for j,(stage,held) in enumerate(((49,None),(50,None),(51,None),(53,36),(55,37),(57,38),(59,39))):
        suffix='unseen' if held else 'val';key=f'without{held}' if held else 'pooled'
        source='unseen_predictions.npz' if held else 'val_predictions.npz'
        with np.load(PREVIOUS/f'stage{stage}/{source}') as a,np.load(PREVIOUS/f'stage44/baseline_{key}_{suffix}.npz') as b:
            ids=a['indices'];sc=a['scenarios'];blocks=m.block_key.to_numpy()[ids]
            if not np.array_equal(ids,b['indices']):raise ValueError('Prediction alignment mismatch')
            def summary(mask):return failure_summary(a['logits'],b['logits'],a['profiles'],a['labels'],mask)
            result[str(stage)]=dict(overall=summary(np.ones(len(ids),bool)),
                scenarios={str(s):summary(sc==s) for s in np.unique(sc)},
                blocks={str(k):summary(blocks==k) for k in np.unique(blocks)},
                slow_heading={str(v):summary(context[ids,6]==v) for v in (0,1)})
        progress(status='auditing_failures',model='analysis',batch=j+1,batches=7)
    save_json(folder/'failures.json',dict(results=result,scope='Exploratory validation analysis. Held-scenario predictions include the previously selected fallback policy. No model selection on these breakdowns is an independent confirmation.'))

def report(folder,progress):
    timing=json.loads((OUT/'stage62/alignment.json').read_text());cases=[]
    for p in PLAN:
        if p['kind']=='intervention':cases.append(json.loads((ROOT/p['directory']/'interventions.json').read_text()))
    offsets={s:r['row_minus_sensor_seconds']['unit1_rgb5']['median'] for s,r in timing['scenarios'].items()}
    timing_flag=any(abs(v)>.25 for v in offsets.values())
    primary=next(r for r in cases if r['source_stage']==57)
    best=max(primary['modes'],key=lambda k:primary['modes'][k]['metrics']['macro_success5'])
    next_action=('Resolve image filename clock conventions and verify physical synchronization before retraining with any shifted images.' if timing_flag else
        'Audit camera visibility and train a controlled RGB-dropout ablation using training-only fitting and a fresh validation protocol.')
    lines=['# Diagnostic cycle: Stages 62–71','',f'Front-image filename offsets (row minus filename, seconds): {offsets}.',
        'Filename offsets alone do not prove an alignment bug. No image rows were shifted.',
        '',f'Scenario 38 raw-checkpoint intervention with highest exploratory success: {best}.',
        'Gray and shuffled inputs are distribution shifts; their results diagnose sensitivity, not the performance of a separately trained GPS-only model.','']
    for r in cases:
        values=', '.join(f"{k}: {v['metrics']['macro_success5']:.2%}" for k,v in r['modes'].items())
        lines.append(f"- Checkpoint {r['source_stage']}, raw corrections (fallback disabled for diagnosis): {values}.")
    lines+=['',next_action,'','No calibration/test evaluation, publication claim, or automatic timestamp correction is authorized by these diagnostic results. Further training requires a justified, recorded protocol; repeatedly inspecting validation does not provide independent confirmation.']
    (folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    save_json(folder/'decision.json',dict(publication_ready=False,calibration_test_used=False,timestamp_convention_review_required=timing_flag,next_action=next_action))
