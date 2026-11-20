import json
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from src.pathway_plan import ROOT,OUT,PLAN
from src.publication_plan import CACHE
from src.diagnostic_analysis import clock_seconds,clock_delta,distribution
from src.acquire_scenarios import save_json

def timing(folder,progress):
    m=pd.read_json(CACHE/'manifest.jsonl',lines=True);records={}
    for s in (36,37,38,39):
        frame=pd.read_csv(ROOT/f'data/raw/scenario{s}/scenario{s}.csv')
        sub=m[(m.scenario==s)&(m.split=='train')];rows=np.unique(np.stack(sub.x_indices))
        f=frame.iloc[rows];rowtime=np.array([clock_seconds(t) for t in f.timestamp])
        image_time=np.array([clock_seconds(t) for t in f.unit1_rgb5]);offset=clock_delta(rowtime,image_time)
        seqs=f.seq_index.to_numpy();byseq={str(q):distribution(offset[seqs==q]) for q in np.unique(seqs)}
        examples=[]
        for i in np.linspace(0,len(f)-1,8,dtype=int):
            path=ROOT/f'data/raw/scenario{s}'/f.unit1_rgb5.iloc[i]
            with Image.open(path) as im:
                exif=im.getexif();examples.append(dict(row=int(rows[i]),image=f.unit1_rgb5.iloc[i],size=list(im.size),
                    exif_time={str(k):str(exif[k]) for k in (306,36867,36868) if k in exif}))
        records[str(s)]=dict(offset=distribution(offset),sequence_offsets=byseq,exif_samples=examples)
        progress(status='checking_clock_provenance',model='analysis',batch=s-35,batches=4)
    shutil.copy2(ROOT/'paper/SENSOR_TIMING_PROVENANCE.md',folder/'PROVENANCE.md')
    save_json(folder/'timing.json',dict(scenarios=records,physical_synchronization_verified=False,row_mapping_changed=False))

def report(folder,progress):
    rows=[]
    for p in PLAN:
        if p['kind']=='probe':rows.append(json.loads((ROOT/p['directory']/'probe.json').read_text()))
    lines=['# RGB learning pathway audit','',
        'Training-only, fixed-checkpoint diagnostic probes. No optimizer steps, validation selection, or calibration/test evaluation.',
        'Timestamp conversion remains unresolved; the distributed CSV mapping is unchanged. See Stage 72/PROVENANCE.md.','']
    saturated=[]
    for r in rows:
        fp=r['summary']['fp32'];bf=r['summary']['bf16']
        lines.append(f"- Checkpoint {r['source_stage']}: saturation FP32 {fp['saturation_fraction']:.2%}, BF16 {bf['saturation_fraction']:.2%}; RGB parameter gradient norm {fp['rgb_gradient_norm']:.4g}/{bf['rgb_gradient_norm']:.4g}; RGB feature variability {fp['rgb_feature_std']:.4g}; mean absolute original-versus-gray correction change {fp['gray_correction_delta']:.4g}.")
        if fp['saturation_fraction']>.5:saturated.append(r['source_stage'])
    action=('Run a controlled training-only pilot comparing the current tanh correction against a smooth non-saturating residual penalty, holding inputs and splits fixed. Verify finite nonzero RGB gradients before any full multi-seed queue.' if saturated else
        'Run a controlled training-only pilot comparing existing fusion with an explicit RGB residual head and modality dropout; establish useful image sensitivity before expanding training.')
    lines+=['',action,'','These probes identify mechanisms to investigate; they do not prove which change will improve generalization. Gradient norms are local diagnostics and depend on parameter count and loss scaling. Zero output-head initialization intentionally blocks upstream gradients on the first step; persistent zero gradients in trained checkpoints require separate explanation.','',
        'Before any shifted-image experiment, obtain timing-conversion provenance. Keep final evaluation reserved and predeclare the next comparison.']
    (folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    save_json(folder/'decision.json',dict(publication_ready=False,saturated_checkpoints=saturated,next_experiment=action,physical_timing_unresolved=True))
