import json
import numpy as np
import pandas as pd
import torch
from src.publication_plan import ROOT,OUT,CACHE,PLAN,SEEDS,VARIANTS
from src.publication_data import load,digest,partitions
from src.publication_metrics import metrics,per_sample,calibrate,candidate_outcomes,block_intervals
from src.publication_training import build_model,loader,predict,save_arrays
from src.acquire_scenarios import save_json


def audit(folder,progress):
    manifest,gps,powers,receipt=load();train,val,_=partitions(manifest)
    report={'scope':'Training/validation only; geometric baseline is an adaptation, not official challenge reproduction.',
            'by_scenario':{},'protocol_source':'paper/BENCHMARK_PROTOCOL_LEDGER.md'}
    for s in (36,37,38,39):
        ti=train[manifest.scenario.to_numpy()[train]==s];vi=val[manifest.scenario.to_numpy()[val]==s]
        known=set(manifest.target_beam.iloc[ti]);v=manifest.target_beam.iloc[vi]
        report['by_scenario'][str(s)]=dict(train=len(ti),val=len(vi),train_classes=len(known),
            val_unseen_label_fraction=float((~v.isin(known)).mean()),
            median_horizon_seconds=float(manifest.horizon_seconds.iloc[vi].median()))
    # Extrapolate from last two historical positions, then use relative receiver-frame bearing.
    # Corrected circular angle distances; single pooled offset fitted on training power only.
    angles=np.empty(len(manifest))
    for s in (36,37,38,39):
        frame=pd.read_csv(ROOT/f'data/raw/scenario{s}/scenario{s}.csv')
        idx=np.flatnonzero(manifest.scenario.to_numpy()==s)
        x=np.stack(manifest.x_indices.iloc[idx].to_numpy())
        positions=[]
        for unit in (1,2):
            pos=frame[[f'unit{unit}_gps1_lat',f'unit{unit}_gps1_lon']].to_numpy()[x]
            positions.append(pos[:,-1]+2.5*(pos[:,-1]-pos[:,-2]))
        last=frame[['unit1_gps1_lat','unit1_gps1_lon']].to_numpy()[x[:,-1]]
        travel=positions[0]-last;relative=positions[0]-positions[1]
        heading=np.arctan2(travel[:,0],travel[:,1]);bearing=np.arctan2(relative[:,0],relative[:,1])
        angles[idx]=(-(bearing-heading-np.pi/4)+np.pi)%(2*np.pi)-np.pi
    codebook=np.linspace(-np.pi,np.pi,256)
    distances=np.abs((angles[:,None]-codebook[None,:]+np.pi)%(2*np.pi)-np.pi)
    nearest=distances.argmin(1)
    offsets=np.arange(256)
    loss=powers[train].max(1)[:,None]-np.take_along_axis(powers[train],(nearest[train,None]+offsets)%256,axis=1)
    offset=int(loss.mean(0).argmin())
    logits=np.roll(-distances[val],offset,axis=1)
    labels=manifest.target_beam.to_numpy()[val];scenarios=manifest.scenario.to_numpy()[val]
    report['geometry_offset']=offset
    report['geometry_val']=metrics(logits,powers[val],labels,scenarios)
    majority=np.bincount(manifest.target_beam.iloc[train],minlength=256)
    report['train_frequency_val']=metrics(np.broadcast_to(majority,(len(val),256)),powers[val],labels,scenarios)
    save_json(folder/'report.json',report)
    save_arrays(folder/'geometry_val.npz',dict(logits=logits,profiles=powers[val],labels=labels,scenarios=scenarios,indices=val))


def gate(folder,progress):
    manifest,_,__,receipt=load()
    runs={};vectors={};common_indices=None
    for p in PLAN:
        if p['kind']!='train' or p['holdout'] is not None: continue
        path=ROOT/p['directory'];r=json.loads((path/'result.json').read_text())
        if digest(path/'best.pt')!=r['checkpoint_sha256']: raise ValueError('Checkpoint changed after training')
        with np.load(path/'val_predictions.npz') as a:
            if common_indices is None: common_indices=a['indices']
            if not np.array_equal(a['indices'],common_indices): raise ValueError('Unmatched validation samples')
            vectors.setdefault(p['variant'],[]).append(per_sample(a['logits'],a['profiles'],a['labels'])['success'].astype(float))
        runs.setdefault(p['variant'],[]).append(dict(stage=p['id'],seed=p['seed'],**r))
    means={v:float(np.mean([r['val']['macro_success5'] for r in rs])) for v,rs in runs.items()}
    baseline=max((v for v in VARIANTS if v!='power_aware'),key=lambda v:means[v])
    delta=np.mean(vectors['power_aware'],axis=0)-np.mean(vectors[baseline],axis=0)
    subset=manifest.iloc[common_indices];blocks=subset.block_key.to_numpy();scenarios=subset.scenario.to_numpy()
    rng=np.random.default_rng(8128);draws=[]
    groups={s:np.unique(blocks[scenarios==s]) for s in (36,37,38,39)}
    sums={b:delta[blocks==b].sum() for b in np.unique(blocks)}
    counts={b:int((blocks==b).sum()) for b in sums}
    for _ in range(2000):
        scores=[]
        for s,keys in groups.items():
            sampled=rng.choice(keys,len(keys),replace=True)
            scores.append(sum(sums[b] for b in sampled)/sum(counts[b] for b in sampled))
        draws.append(np.mean(scores))
    ci=np.quantile(draws,[.025,.975]).tolist()
    transfer={}
    for p in PLAN:
        if p['kind']=='train' and p['holdout']:
            r=json.loads((ROOT/p['directory']/'unseen_report.json').read_text())
            transfer.setdefault(str(p['holdout']),{})[p['variant']]=r['macro_success5']
    wins=sum(v['power_aware']>v['fusion_ce'] for v in transfer.values())
    checks=dict(macro_gain_at_least_one_percentage_point=means['power_aware']>=means[baseline]+.01,
                paired_block_interval_positive=ci[0]>0,transfer_wins_at_least_three= wins>=3)
    decision=dict(passed=all(checks.values()),checks=checks,means=means,strongest_baseline=baseline,
        paired_bootstrap_95=ci,transfer=transfer,scope='Predeclared development gate; selection-conditioned uncertainty, not test significance.')
    save_json(folder/'decision.json',decision)
    save_json(folder/'runs.json',runs)
    if decision['passed']:
        frozen=[]
        for p in PLAN:
            if p['kind']=='train' and p['holdout'] is None:
                checkpoint=ROOT/p['directory']/'best.pt'
                frozen.append(dict(stage=p['id'],variant=p['variant'],seed=p['seed'],path=str(checkpoint),sha256=digest(checkpoint)))
        save_json(folder/'freeze.json',dict(models=frozen,data=receipt,
            calibration_target=.9,delta_db=3,code=json.loads((OUT/'stage8/code_manifest.json').read_text()),
            policy='No additional model selection or tuning after this freeze.'))


def frozen_models():
    decision=json.loads((OUT/'stage33/decision.json').read_text())
    if not decision['passed']: return None
    frozen=json.loads((OUT/'stage33/freeze.json').read_text())
    for rel,sha in frozen['code'].items():
        if digest(ROOT/rel)!=sha: raise ValueError('Code changed after experiment declaration')
    _,__,___,receipt=load()
    if receipt!=frozen['data']: raise ValueError('Data changed after freeze')
    for item in frozen['models']:
        from pathlib import Path
        if digest(Path(item['path']))!=item['sha256']: raise ValueError('Frozen model changed')
    return frozen


def heldout(folder,progress,split):
    frozen=frozen_models()
    if frozen is None:
        save_json(folder/'skipped.json',dict(reason='Development quality gate failed; held-out data remains unused.'))
        return
    manifest,gps,profiles,receipt=load();indices=np.flatnonzero(manifest.split==split)
    torch.set_num_threads(4)
    freeze_hash=digest(OUT/'stage33/freeze.json')
    if split=='test':
        calibration=json.loads((OUT/'stage34/calibration.json').read_text())
        if calibration['freeze_sha256']!=freeze_hash: raise ValueError('Calibration/freeze mismatch')
        save_json(folder/'test_lock.json',dict(freeze_sha256=freeze_hash,
            calibration_sha256=digest(OUT/'stage34/calibration.json'),policy='Fixed predictions/metrics only; no test tuning.'))
    parameters={};reports={}
    for item in frozen['models']:
        key=str(item['stage']);pred_file=folder/f'{key}_predictions.npz'
        if not pred_file.exists():
            saved=torch.load(item['path'],map_location='cpu',weights_only=False)
            cfg=saved['config'];scaled=(gps-np.array(saved['scaler_min'],dtype=np.float32))/np.array(saved['scaler_span'],dtype=np.float32)
            model=build_model(cfg);model.load_state_dict(saved['model']);del saved
            predictions=predict(model,loader(manifest,scaled,profiles,indices,cfg),cfg,progress,split)
            save_arrays(pred_file,predictions);del model;torch.cuda.empty_cache()
        with np.load(pred_file) as a:
            logits=a['logits'];power=a['profiles'];labels=a['labels'];scenarios=a['scenarios']
        if split=='calib':
            parameters[key]=calibrate(logits,power)
        else:
            par=calibration['parameters'][key]
            report={'variant':item['variant'],'seed':item['seed'], 'metrics':metrics(logits,power,labels,scenarios),'candidate_sets':{}}
            for adaptive in (False,True):
                counts,regret=candidate_outcomes(logits,power,par,adaptive)
                kind='adaptive_prefix' if adaptive else 'fixed_k'
                report['candidate_sets'][kind]=dict(coverage_3db=float((regret<=3).mean()),mean_candidates=float(counts.mean()),
                    mean_power_loss_db=float(regret.mean()),full_sweep_candidates=256,
                    block_bootstrap_95=dict(zip(('coverage_3db','mean_candidates','mean_power_loss_db'),
                        block_intervals(np.column_stack((regret<=3,counts,regret)),manifest.block_key.to_numpy()[indices],scenarios))),
                    by_scenario={str(s):dict(coverage_3db=float((regret[scenarios==s]<=3).mean()),mean_candidates=float(counts[scenarios==s].mean())) for s in (36,37,38,39)})
            reports[key]=report
        progress(status='calibrating' if split=='calib' else 'locked_testing',model=item['variant'],batch=len(parameters or reports),batches=len(frozen['models']))
    if split=='calib': save_json(folder/'calibration.json',dict(freeze_sha256=freeze_hash,parameters=parameters))
    else: save_json(folder/'test_report.json',dict(freeze_sha256=freeze_hash,models=reports,scope='Locked test; correlated samples, no online control claim.'))


def report(folder,progress):
    decision=json.loads((OUT/'stage33/decision.json').read_text())
    runs=json.loads((OUT/'stage33/runs.json').read_text())
    lines=['# Research evidence package','',
        'This is a generated development report, not a submission-ready manuscript or a state-of-the-art claim.','',
        '## Validation comparisons','']
    for variant,values in runs.items():
        scores=[r['val']['macro_success5'] for r in values]
        lines.append(f'- {variant}: mean macro success within 3 dB at k=5 {np.mean(scores):.2%}; sample SD {np.std(scores,ddof=1):.2%}; three seeds.')
    lines += ['',f"Automatic development gate: {'passed' if decision['passed'] else 'not passed'}.",
        f"Strongest matched baseline: {decision['strongest_baseline']}. Paired validation block-bootstrap interval: {decision['paired_bootstrap_95']}.",
        '', '## Held-out evaluation','']
    lines.append('Calibration and locked test completed; see stage34/calibration.json and stage35/test_report.json.' if decision['passed'] else
                 'Calibration and test were deliberately skipped because the predeclared development gate failed. Further hypotheses are required before opening the test set.')
    lines += ['', '## Claims and outstanding work','',
        '- This is the local 256-beam forecasting protocol. Published headline scores on different splits, codebooks or modalities are not directly comparable.',
        '- Exact reproduction of a stronger published architecture remains outstanding; local fusion baselines are not substitutes for claiming published SOTA.',
        '- Candidate-set power metrics are offline. End-to-end measurement, feedback latency and temporal coverage guarantees have not been established.',
        '- Novelty assessment, failure analysis, authorship, venue selection and manuscript review remain necessary.',
        '- No acceptance or publication-readiness claim follows automatically from passing the development gate.',
        '', 'Protocol: paper/BENCHMARK_PROTOCOL_LEDGER.md. Source snapshots and environment: stage8/.']
    (folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    save_json(folder/'readiness.json',dict(compute_pipeline_complete=True,development_gate_passed=decision['passed'],
        publication_ready=False,reason='Requires scientific novelty, exact published comparisons, and manuscript review.'))
    # Exportable scientific figure, generated only from actual completed runs.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    names=list(runs);means=[np.mean([r['val']['macro_success5'] for r in runs[v]])*100 for v in names]
    std=[np.std([r['val']['macro_success5'] for r in runs[v]],ddof=1)*100 for v in names]
    ax.bar(names,means,yerr=std,color='#315b78',capsize=4)
    ax.set_ylabel('Validation success within 3 dB at k=5 (%)')
    ax.set_xlabel('Model / training objective');ax.set_title('Three-seed means with sample standard deviation')
    fig.tight_layout();fig.savefig(folder/'validation_comparison.png',dpi=180);plt.close(fig)
