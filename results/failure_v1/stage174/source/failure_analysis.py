import json
import numpy as np
import pandas as pd
from src.failure_plan import ROOT,OUT
from src.publication_plan import CACHE
from src.publication_data import digest
from src.publication_metrics import per_sample
from src.acquire_scenarios import save_json

def shift(reference,target):
    reference=np.asarray(reference,dtype=np.float64);target=np.asarray(target,dtype=np.float64)
    scale=np.maximum(reference.std(0),1e-6);z=(target.mean(0)-reference.mean(0))/scale
    outside=(target<reference.min(0))|(target>reference.max(0))
    return dict(mean_abs_standardized_mean_difference=float(np.abs(z).mean()),max_abs_standardized_mean_difference=float(np.abs(z).max()),outside_reference_range_fraction=float(outside.mean()),rows_outside_any_reference_range=float(outside.any(1).mean()),reference_n=len(reference),target_n=len(target))

def correction(geometry,model,profiles,labels):
    a=per_sample(geometry,profiles,labels);b=per_sample(model,profiles,labels)
    ga=np.argsort(-geometry,axis=1,kind='stable')[:,:5];mb=np.argsort(-model,axis=1,kind='stable')[:,:5]
    helped=~a['success']&b['success'];harmed=a['success']&~b['success']
    return dict(n=len(labels),geometry_success=float(a['success'].mean()),model_success=float(b['success'].mean()),helped=int(helped.sum()),harmed=int(harmed.sum()),both_success=int((a['success']&b['success']).sum()),both_fail=int((~a['success']&~b['success']).sum()),top5_set_changed_fraction=float(np.any(np.sort(ga,1)!=np.sort(mb,1),axis=1).mean()),mean_regret_change_db=float((b['regret']-a['regret']).mean()))

def sample(m,split,scenarios):
    chosen=[]
    for s in scenarios:
        ids=np.flatnonzero((m.split==split)&(m.scenario==s))
        chosen.extend(ids[np.linspace(0,len(ids)-1,min(512,len(ids)),dtype=int)])
    return np.asarray(chosen,dtype=int)

def features(m,ids):
    result=np.empty((len(ids),512),np.float32)
    for s in sorted(m.scenario.iloc[ids].unique()):
        loc=np.flatnonzero(m.scenario.to_numpy()[ids]==s);rows=ids[loc]
        path=ROOT/f'data/processed/resolution_v1/s{s}_192.npy'
        receipt=json.loads((path.parent/f's{s}_192.json').read_text())
        if digest(path)!=receipt['sha256']:raise ValueError('Feature hash changed')
        a=np.load(path,mmap_mode='r');x=np.stack(m.x_indices.iloc[rows].to_numpy());v=np.asarray(a[x]).mean(1);a._mmap.close()
        result[loc]=(v-v.mean(1,keepdims=True))/np.sqrt(v.var(1,keepdims=True)+1e-5)
    if not np.isfinite(result).all():raise ValueError('Missing features')
    return result

def audit(plan,folder,progress):
    s=plan['scenario'];m=pd.read_json(CACHE/'manifest.jsonl',lines=True)
    receipt=json.loads((CACHE/'receipt.json').read_text())
    if digest(CACHE/'manifest.jsonl')!=receipt['manifest_sha256'] or digest(CACHE/'arrays.npz')!=receipt['arrays_sha256']:raise ValueError('Data fingerprints changed')
    with np.load(CACHE/'arrays.npz') as a:gps=a['gps']
    other=[x for x in (36,37,38,39) if x!=s]
    ref=sample(m,'train',other);seen=sample(m,'val',other);held=sample(m,'val',[s])
    progress(status='input_shift',model='analysis',batch=1,batches=3)
    fref=features(m,ref);fseen=features(m,seen);fheld=features(m,held)
    report=dict(scenario=s,scope='Descriptive development audit. Equal per-scenario deterministic samples up to 512 windows; no label-based sample selection. Reference statistics fit only to training scenarios. Shift statistics do not prove cause.',sample_ids=dict(reference=ref.tolist(),seen=seen.tolist(),held=held.tolist()),input_shift={})
    for name,a,b,c in [('gps',gps[ref].reshape(len(ref),-1),gps[seen].reshape(len(seen),-1),gps[held].reshape(len(held),-1)),('visual',fref,fseen,fheld)]:report['input_shift'][name]=dict(seen_validation=shift(a,b),held_validation=shift(a,c))
    report['corrections']={}
    for scope,suffix in [('seen','val'),('held','unseen')]:
        with np.load(ROOT/f'results/conservative_v1/stage44/baseline_without{s}_{suffix}.npz') as a:
            ids=a['indices'];geo=a['logits'];power=a['profiles'];labels=a['labels']
        expected=(m.split.iloc[ids]=='val')&((m.scenario.iloc[ids]==s) if scope=='held' else (m.scenario.iloc[ids]!=s))
        if not expected.all():raise ValueError('Audit rows outside declared split')
        values={}
        for arm,stage in [('gps',165+2*(s-36)),('visual',166+2*(s-36))]:
            path=ROOT/f'results/transfer_v1/stage{stage}'/('unseen_predictions.npz' if scope=='held' else 'val_predictions.npz')
            with np.load(path) as a:
                if not np.array_equal(ids,a['indices']):raise ValueError('Prediction IDs differ')
                values[arm]=correction(geo,a['logits'],power,labels)
        report['corrections'][scope]=values
    progress(status='saving_audit',model='analysis',batch=3,batches=3)
    save_json(folder/'audit.json',report)

def report(folder,progress):
    records=[json.loads((OUT/f'stage{i}/audit.json').read_text()) for i in range(175,179)]
    lines=['# Transfer failure audit','', 'Descriptive findings from previously explored validation data; no final-test evaluation or model fitting.','']
    for r in records:
        c=r['corrections']['held']['visual'];v=r['input_shift']['visual'];g=r['input_shift']['gps']
        lines.append(f"- S{r['scenario']}: visual corrections help {c['helped']} windows and harm {c['harmed']}; top-five set changes in {c['top5_set_changed_fraction']:.2%} of windows. Visual standardized mean shift: seen {v['seen_validation']['mean_abs_standardized_mean_difference']:.3f}, held {v['held_validation']['mean_abs_standardized_mean_difference']:.3f}. GPS shift: seen {g['seen_validation']['mean_abs_standardized_mean_difference']:.3f}, held {g['held_validation']['mean_abs_standardized_mean_difference']:.3f}.")
    action='Next candidate: compare image-feature dropout/consistency regularization against the unchanged model using inner scenario splits within each training fold. Choose settings without excluded-scenario labels. Then freeze choices and run outer transfer evaluation. Because these scenarios have already been explored, retain the development-evidence limitation. No post-hoc per-scenario fallback from held labels.'
    lines+=['',action,'','Input shifts are descriptive, not causal attribution. Overlapping windows are correlated; counts are not independent statistical trials. Final calibration/test remain reserved.']
    save_json(folder/'evidence.json',dict(publication_ready=False,calibration_test_used=False,records=records,next_action=action))
    (folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
