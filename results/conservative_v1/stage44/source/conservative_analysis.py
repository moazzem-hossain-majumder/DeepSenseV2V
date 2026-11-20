import json
import numpy as np
import pandas as pd
from src.conservative_plan import ROOT,OUT,PLAN
from src.publication_plan import CACHE
from src.publication_metrics import per_sample
from src.acquire_scenarios import save_json


def diagnostics(folder,progress):
    with np.load(ROOT/'results/publication_v1/stage9/geometry_val.npz') as a:
        geo=per_sample(a['logits'],a['profiles'],a['labels']);ids=a['indices'];gpanel=a['logits'].argmax(1)//64;target=a['labels']//64
    m=pd.read_json(CACHE/'manifest.jsonl',lines=True)
    context=np.load(OUT/'stage44/context.npy');records={}
    for stage in (40,41,42):
        with np.load(ROOT/f'results/aftercare_v1/stage{stage}/val_predictions.npz') as a:
            if not np.array_equal(ids,a['indices']):raise ValueError('Unaligned older residual results')
            model=per_sample(a['logits'],a['profiles'],a['labels']);mpanel=a['logits'].argmax(1)//64
        help_=model['success']&~geo['success'];harm=geo['success']&~model['success']
        def summary(mask):
            return dict(n=int(mask.sum()),help_fraction=float(help_[mask].mean()),harm_fraction=float(harm[mask].mean()),
                model_panel_accuracy=float((mpanel[mask]==target[mask]).mean()),geometry_panel_accuracy=float((gpanel[mask]==target[mask]).mean()))
        records[str(stage)]=dict(overall=summary(np.ones(len(ids),bool)),
            by_scenario={str(s):summary(m.scenario.to_numpy()[ids]==s) for s in (36,37,38,39)},
            by_slow_heading={str(bool(v)):summary(context[ids,6]==v) for v in (0,1) if (context[ids,6]==v).any()})
    save_json(folder/'help_harm.json',dict(records=records,
        scope='Historical validation comparison; panel label is floor(beam/64). Does not establish physical antenna mounting or image field-of-view calibration.',
        camera_limit='Front RGB5 only. Semantic receiver visibility and front/rear coverage remain unverified; no rear-camera performance claim.'))


def compare(folder,progress):
    m=pd.read_json(CACHE/'manifest.jsonl',lines=True)
    geometry=json.loads((OUT/'stage44/report.json').read_text());pool=geometry['configurations']['pooled']
    with np.load(OUT/'stage44/baseline_pooled_val.npz') as a:
        base=per_sample(a['logits'],a['profiles'],a['labels'])['success'].astype(float);ids=a['indices']
    sc=m.scenario.to_numpy()[ids];blocks=m.block_key.to_numpy()[ids]
    groups={s:np.unique(blocks[sc==s]) for s in (36,37,38,39)}
    records={};rng=np.random.default_rng(6044)
    for method in ('compact_gps','bounded_multimodal'):
        pooled=[p for p in PLAN if p['kind']=='train' and p['method']==method and p['holdout'] is None]
        runs=[];samples=[]
        for p in pooled:
            path=ROOT/p['directory'];selection=json.loads((path/'selection.json').read_text())
            runs.append(dict(seed=p['seed'],**selection))
            with np.load(path/'effective_val.npz') as a:
                if not np.array_equal(ids,a['indices']):raise ValueError('Comparison windows differ')
                samples.append(per_sample(a['logits'],a['profiles'],a['labels'])['success'].astype(float))
        delta=np.mean(samples,axis=0)-base
        sums={b:delta[blocks==b].sum() for b in np.unique(blocks)};counts={b:int((blocks==b).sum()) for b in sums}
        draws=[]
        for _ in range(2000):
            per_s=[]
            for s,keys in groups.items():
                selected=rng.choice(keys,len(keys),replace=True)
                per_s.append(sum(sums[b] for b in selected)/sum(counts[b] for b in selected))
            draws.append(np.mean(per_s))
        transfer={}
        for p in PLAN:
            if p['kind']=='train' and p['method']==method and p['holdout']:
                selected=json.loads((ROOT/p['directory']/'selection.json').read_text());s=p['holdout']
                ref=geometry['configurations'][f'without{s}']['unseen_val']['macro_success5']
                transfer[str(s)]=dict(correction_enabled=selected['correction_enabled'],
                    effective_success=selected['effective_unseen']['macro_success5'],geometry_success=ref)
        records[method]=dict(runs=runs,mean_candidate_success=float(np.mean([r['candidate_val']['macro_success5'] for r in runs])),
            mean_selected_success=float(np.mean([r['effective_val']['macro_success5'] for r in runs])),
            correction_enabled_count=sum(r['correction_enabled'] for r in runs),paired_macro_interval_95=np.quantile(draws,[.025,.975]).tolist(),
            unseen_scenarios=transfer,unseen_wins=sum(r['effective_success']>r['geometry_success']+.0025 for r in transfer.values()))
    save_json(folder/'comparison.json',dict(geometry=pool,methods=records,
        scope='Exploratory selection-validation and unseen-scenario validation. Fallback creates optimistic selection-validation scores; transfer evidence is required.',
        calibration_test_used=False,publication_ready=False))


def report(folder,progress):
    summary=json.loads((OUT/'stage60/comparison.json').read_text());baseline=summary['geometry']['selection_val']['macro_success5']
    lines=['# Conservative-correction research results','',summary['scope'],'',
        f"Selected analytic convention: {summary['geometry']['chosen']}; selection-validation macro success {baseline:.2%}.",'']
    for name,r in summary['methods'].items():
        lines += [f"- {name}: raw learned mean {r['mean_candidate_success']:.2%}; after validation-selected fallback {r['mean_selected_success']:.2%}; corrections enabled in {r['correction_enabled_count']}/3 pooled runs; unseen-scenario wins {r['unseen_wins']}/4.",
                  f"  Paired macro trajectory-bootstrap interval: {r['paired_macro_interval_95']}."]
    qualified=[name for name,r in summary['methods'].items() if r['mean_selected_success']>baseline+.01 and r['paired_macro_interval_95'][0]>0 and r['unseen_wins']>=3]
    action='Review independent baseline reproduction and freeze a new held-out evaluation protocol.' if qualified else 'Do not open test. Investigate geometry/camera observability and whether simpler analytic methods are sufficient.'
    lines += ['',action,'','No calibration/test data was used in this cycle. Publication readiness, novelty, exact published-method comparisons and physical mounting/camera coverage remain separate review requirements.']
    (folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    save_json(folder/'readiness.json',dict(qualified_for_review=qualified,publication_ready=False,next_action=action))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4));names=list(summary['methods']);positions=np.arange(len(names))
    ax.bar(positions-.18,[summary['methods'][n]['mean_candidate_success']*100 for n in names],width=.36,label='Raw learned correction')
    ax.bar(positions+.18,[summary['methods'][n]['mean_selected_success']*100 for n in names],width=.36,label='With validation-selected fallback')
    ax.axhline(baseline*100,color='black',linestyle='--',label='Selected analytic geometry')
    ax.set_xticks(positions);ax.set_xticklabels(names);ax.set_ylabel('Macro success within 3 dB at five beams (%)')
    ax.set_title('Three-seed development validation; selection-biased, not test results');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(folder/'comparison.png',dpi=180);plt.close(fig)
