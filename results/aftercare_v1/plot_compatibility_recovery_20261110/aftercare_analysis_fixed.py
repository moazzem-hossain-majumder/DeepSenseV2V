"""Follow-up EDA and evidence packaging; no test-driven model selection."""
import json
import shutil
import numpy as np
import pandas as pd
import torch
from src.aftercare_plan import ROOT,OUT,PARENT
from src.publication_plan import CACHE
from src.publication_data import load,digest
from src.publication_metrics import metrics,per_sample,block_intervals
from src.publication_training import build_model,loader,predict
from src.acquire_scenarios import save_json


def image_eda(folder,progress):
    manifest=pd.read_json(CACHE/'manifest.jsonl',lines=True)
    rng=np.random.default_rng(3810);report={};brightness=[];sharpness=[];selected={}
    for scenario in (36,37,38,39):
        subset=manifest[(manifest.scenario==scenario)&manifest.split.isin(['train','val'])]
        rows=np.unique(np.concatenate(subset.x_indices.to_numpy()))
        chosen=np.sort(rng.choice(rows,min(2000,len(rows)),replace=False));selected[str(scenario)]=chosen.tolist()
        images=np.load(CACHE/f'rgb{scenario}.npy',mmap_mode='r')
        values=[]
        for start in range(0,len(chosen),128):
            rgb=np.asarray(images[chosen[start:start+128]],dtype=np.float32)/255
            gray=rgb[...,0]*.299+rgb[...,1]*.587+rgb[...,2]*.114
            lap=gray[:,1:-1,:-2]+gray[:,1:-1,2:]+gray[:,:-2,1:-1]+gray[:,2:,1:-1]-4*gray[:,1:-1,1:-1]
            values.append(np.column_stack((gray.mean((1,2)),gray.std((1,2)),lap.var((1,2)),(gray<.02).mean((1,2)),(gray>.98).mean((1,2)))))
            progress(status='image_quality_eda',model='analysis',scenario=scenario,batch=min(start+128,len(chosen)),batches=len(chosen))
        values=np.concatenate(values);brightness.append(values[:,0]);sharpness.append(values[:,2])
        names=('brightness_0_1','contrast_sd','laplacian_variance_96px','dark_pixel_fraction','bright_pixel_fraction')
        report[str(scenario)]={'sampled_frames':len(chosen),'eligible_unique_frames':len(rows),
            'statistics':{name:dict(zip(('p05','median','p95'),np.quantile(values[:,i],[.05,.5,.95]).tolist())) for i,name in enumerate(names)}}
    save_json(folder/'image_quality.json',dict(scope='Seeded sample of train/validation input frames at resized 96x96 resolution. Not a native-resolution blur metric or semantic day/night label.',scenarios=report,selected_rows=selected))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,data,title,ylabel in zip(axes,(brightness,sharpness),('Image brightness','Resized-image sharpness proxy'),('Mean luminance (0–1)','Laplacian variance at 96 px')):
        ax.boxplot(data,showfliers=False)
        ax.set_xticks([1,2,3,4]);ax.set_xticklabels(['36','37','38','39'])
        ax.set_title(title);ax.set_ylabel(ylabel);ax.set_xlabel('Scenario')
    fig.suptitle('Train/validation input frames; up to 2,000 per scenario, fixed seed')
    fig.tight_layout();fig.savefig(folder/'image_quality.png',dpi=180);plt.close(fig)


def diagnostics(folder,progress):
    manifest=pd.read_json(CACHE/'manifest.jsonl',lines=True)
    with np.load(CACHE/'arrays.npz') as a: gps=a['gps']
    output={}
    for stage in (10,13,16,19,22):
        source=PARENT/f'stage{stage}'
        result=json.loads((source/'result.json').read_text())
        with np.load(source/'val_predictions.npz') as a:
            sample=per_sample(a['logits'],a['profiles'],a['labels']);idx=a['indices'];scenario=a['scenarios']
        distance=gps[idx,-1,2];blocks=manifest.block_key.to_numpy()[idx]
        bins=np.digitize(distance,[10,25,50,100])
        def describe(mask):
            return dict(n=int(mask.sum()),success_3db_at_5=float(sample['success'][mask].mean()),mean_regret_db=float(sample['regret'][mask].mean()))
        output[result['config']['variant']]=dict(
            by_scenario={str(s):describe(scenario==s) for s in np.unique(scenario)},
            by_distance={name:describe(bins==i) for i,name in enumerate(('0-10m','10-25m','25-50m','50-100m','100m+')) if (bins==i).any()},
            by_trajectory={b:describe(blocks==b) for b in np.unique(blocks)})
    geometry=json.loads((PARENT/'stage9/report.json').read_text())['geometry_val']
    decision=json.loads((PARENT/'stage33/decision.json').read_text())
    save_json(folder/'failure_analysis.json',dict(scope='Validation only; seed 42 checkpoint diagnostics, not all-seed confidence.',
        models=output,analytic_geometry=geometry,parent_gate=decision,
        caution='Parent gate compared learned variants. Analytic geometry must also be considered before claiming an improvement.'))


def reproduce(folder,progress):
    source=PARENT/'stage22';saved=torch.load(source/'best.pt',map_location='cpu',weights_only=False)
    manifest,gps,power,receipt=load();cfg=saved['config']
    gps=(gps-np.array(saved['scaler_min'],np.float32))/np.array(saved['scaler_span'],np.float32)
    model=build_model(cfg);model.load_state_dict(saved['model']);del saved
    with np.load(source/'val_predictions.npz') as a: idx=a['indices'];reference=a['logits']
    torch.set_num_threads(4)
    actual=predict(model,loader(manifest,gps,power,idx,cfg),cfg,progress)
    if not np.array_equal(actual['indices'],idx): raise ValueError('Reproduction row alignment failed')
    save_json(folder/'reproducibility.json',dict(checkpoint_sha256=digest(source/'best.pt'),
        max_absolute_logit_difference=float(np.max(np.abs(reference-actual['logits']))),
        top1_agreement=float((reference.argmax(1)==actual['logits'].argmax(1)).mean()),
        exact_prediction_match=bool(np.array_equal(reference,actual['logits'])),
        scope='Inference reproducibility check, not an independent training reproduction.'))


def test_audit(folder,progress):
    test_file=PARENT/'stage35/test_report.json'
    if not test_file.exists():
        save_json(folder/'test_evidence_audit.json',dict(status='incomplete_test_evidence',publication_ready=False,
            reason='Test exposure detected but complete report absent. Do not retrain against this test set.'))
        return
    source=json.loads(test_file.read_text());summary={}
    for row in source['models'].values():
        sets=row['candidate_sets']['adaptive_prefix']
        summary.setdefault(row['variant'],[]).append(dict(seed=row['seed'],coverage=sets['coverage_3db'],
            size=sets['mean_candidates'],worst_scenario_coverage=min(v['coverage_3db'] for v in sets['by_scenario'].values())))
    aggregate={v:dict(mean_coverage=float(np.mean([x['coverage'] for x in rows])),
        mean_candidates=float(np.mean([x['size'] for x in rows])),
        worst_scenario_coverage=float(min(x['worst_scenario_coverage'] for x in rows))) for v,rows in summary.items()}
    candidate=aggregate['power_aware']
    adequate=candidate['mean_coverage']>=.9 and candidate['worst_scenario_coverage']>=.85
    save_json(folder/'test_evidence_audit.json',dict(status='promising_requires_review' if adequate else 'coverage_requires_revision',
        publication_ready=False,source_sha256=digest(test_file),models=aggregate,
        rule='Post-test descriptive adequacy check, not a model-selection or retraining trigger.',
        outstanding=['Exact published-method reproduction','Scientific novelty assessment','Realistic online measurement/feedback cost','Authors, venue and manuscript review']))


def improvement_report(folder,progress):
    manifest=pd.read_json(CACHE/'manifest.jsonl',lines=True);rows=[];success=[];idx=None
    for stage in (40,41,42):
        path=OUT/f'stage{stage}';r=json.loads((path/'result.json').read_text());rows.append(r)
        with np.load(path/'val_predictions.npz') as a:
            if idx is None: idx=a['indices']
            if not np.array_equal(idx,a['indices']): raise ValueError('Mismatched replication samples')
            success.append(per_sample(a['logits'],a['profiles'],a['labels'])['success'].astype(float))
    with np.load(PARENT/'stage9/geometry_val.npz') as a:
        if not np.array_equal(idx,a['indices']): raise ValueError('Geometry validation alignment failed')
        geo=per_sample(a['logits'],a['profiles'],a['labels'])['success']
    delta=np.mean(success,axis=0)-geo
    interval=block_intervals(delta,manifest.block_key.to_numpy()[idx],manifest.scenario.to_numpy()[idx],replicates=2000)
    macro=float(np.mean([r['val']['macro_success5'] for r in rows]))
    geometry=json.loads((PARENT/'stage9/report.json').read_text())['geometry_val']['macro_success5']
    parent_means=json.loads((PARENT/'stage33/decision.json').read_text())['means']
    strongest=max(geometry,max(parent_means.values()))
    report=dict(runs=rows,mean_macro_success5=macro,geometry_macro_success5=geometry,
        parent_learned_means=parent_means,strongest_reference_macro_success5=strongest,
        pooled_paired_gain_interval_95=interval,publication_ready=False,
        next_action='Unseen-scenario and objective-ablation checks before considering calibration/test.' if macro>strongest+.01 and interval[0][0]>0 else 'Investigate sensor geometry and task formulation; do not open test.',
        scope='Exploratory validation; new hypothesis after the parent gate failed. No calibration/test access.')
    save_json(folder/'improvement_report.json',report)
    lines=['# Follow-up development results','',report['scope'],'']
    for r in rows: lines.append(f"- Seed {r['config']['seed']}: macro success within 3 dB at five beams {r['val']['macro_success5']:.2%} (selected epoch {r['best_epoch']}).")
    lines += ['',f'Mean: {macro:.2%}. Analytic geometry baseline: {geometry:.2%}.',
              f'Pooled paired trajectory-bootstrap interval for improvement: {interval}.','',report['next_action'],
              '', 'This evidence does not automatically establish publication readiness.']
    (folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def draft(folder,progress):
    audit=json.loads((OUT/'stage41/test_evidence_audit.json').read_text())
    lines=['# Working manuscript — evidence review required','',
        'Working title: Power-aware GPS–vision beam forecasting under trajectory and scenario shift','',
        '## Abstract outline','',
        'We study 256-beam V2V forecasting from five historical GPS and front-camera observations. The study compares modality-specific and fused models under trajectory-separated splits, with a power-aware objective and calibrated candidate sets. Numerical findings below come from the locked test report; novelty and comparison claims require review.','',
        '## Methods','',
        'Scenarios 36–39; inputs at t-8,t-6,t-4,t-2,t, target t+5. Report actual horizon distributions. Training-only scaling; identical regularization and epoch budgets across three seeds. Validation selects checkpoints, calibration fixes candidate-set parameters, and the locked test evaluates the frozen pipeline.','',
        '## Locked-test results','']
    if 'models' in audit:
        for name,r in audit['models'].items():
            lines.append(f"- {name}: mean empirical 3 dB coverage {r['mean_coverage']:.2%}; mean adaptive candidate count {r['mean_candidates']:.2f}; worst scenario/seed coverage {r['worst_scenario_coverage']:.2%}.")
    else: lines.append('A complete test report was not available; no test values are asserted.')
    lines += ['', '## Limitations and required review','',
        '- Temporal dependence limits exchangeability-based coverage claims. These are empirical reliability results.',
        '- Offline candidate counts do not measure end-to-end latency, feedback availability or actual radio probing cost.',
        '- Compare analytic geometry as well as learned baselines. Published results with different codebooks, modalities or splits are not directly rankable.',
        '- Exact published-model reproduction, a defensible novelty statement, authorship and venue are not yet finalized.',
        '- Review every claim against the source JSON before submission. No automatic submission is configured.','',
        '## Source ledger','',
        'See paper/BENCHMARK_PROTOCOL_LEDGER.md for verified primary-source links and protocol differences.']
    (folder/'MANUSCRIPT_WORKING.md').write_text('\n'.join(lines),encoding='utf-8')


def review_report(folder,progress):
    audit=json.loads((OUT/'stage41/test_evidence_audit.json').read_text())
    reproduction=json.loads((OUT/'stage40/reproducibility.json').read_text())
    save_json(folder/'review_status.json',dict(publication_ready=False,test_evidence=audit,
        reproducibility=reproduction,scope='Evidence branch completed; human scientific/editorial review still required.'))
    (folder/'REPORT.md').write_text('# Evidence review handoff\n\nThe automatic evidence branch is complete.\n\n'
        f"Test-evidence assessment: {audit['status']}.\n\n"
        'Working manuscript: ../stage42/MANUSCRIPT_WORKING.md. Image-quality EDA: ../stage38/. '
        'Failure/baseline analysis: ../stage39/. Reproducibility check: ../stage40/.\n\n'
        'Keep model and test decisions frozen. Novelty, exact published-method comparisons, online cost assumptions, authorship and venue still need review.\n',encoding='utf-8')
