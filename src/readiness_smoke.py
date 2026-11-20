"""Real-data GPU forward/backward/optimizer checks, NOT paper performance estimates.

Uses fresh, explicitly loaded train-only clips from all four scenarios. Does not
consume or overwrite legacy caches, checkpoints, calibration or test predictions.
"""
import gc
import argparse
import json
from pathlib import Path
import pickle
import random
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

from src.acquire_scenarios import ROOT, RAW, save_json
from src.gps_features import extract_sequence_gps_features, GPSFeatureScaler
from src.models import create_model, MultiTaskLoss, count_parameters
from src.multiscenario_windows import build_windows
from src.readiness_audit import describe

OUT=ROOT/'results'/'readiness'


def prepare(scenarios=(36,37,38,39), report_name='window_and_split_audit.json'):
    selected=[]
    summaries={}
    for scenario in scenarios:
        folder=RAW/f'scenario{scenario}'
        frame=pd.read_csv(folder/f'scenario{scenario}.csv')
        with (folder/f'scenario{scenario}.p').open('rb') as f: data=pickle.load(f)
        windows=build_windows(frame,scenario)
        assert windows.groupby('block_key')['split'].nunique().max()==1
        summaries[str(scenario)]={'windows':len(windows),'splits':windows['split'].value_counts().to_dict(),
            'blocks':windows.groupby('split')['block_key'].nunique().to_dict(),
            'actual_horizon_seconds':describe(windows.horizon_seconds)}
        train=windows[windows.split=='train']
        powers=np.concatenate([data[f'unit1_pwr{i}'] for i in range(1,5)],axis=1)
        for index in np.linspace(0,len(train)-1,4,dtype=int):
            row=train.iloc[index]
            selected.append({'scenario':scenario,'block_key':row.block_key,
                'x_indices':row.x_indices,'y_index':int(row.y_index),
                'paths':[folder/frame.unit1_rgb5.iloc[i] for i in row.x_indices],
                'gps':extract_sequence_gps_features(frame,row.x_indices),
                'label':int(row.target_beam),'profile':10*np.log10(np.maximum(powers[row.y_index],1e-12))})
    # Interleave recordings, so every four examples include all four scenarios.
    selected=[selected[s*4+i] for i in range(4) for s in range(len(scenarios))]
    gps=np.stack([x['gps'] for x in selected])
    scaler=GPSFeatureScaler()
    scaler.fit(gps)
    gps=scaler.transform(gps)
    summaries['scope']='Smoke-only train sample scaler; not a production training fit. Front RGB5 view. No test metrics.'
    summaries['selected']=[{k:x[k] for k in ['scenario','block_key','x_indices','y_index']} for x in selected]
    save_json(OUT/report_name,summaries)
    return selected,torch.from_numpy(gps).float()


def load_images(selected,size):
    clips=[]
    for sample in selected:
        frames=[]
        for path in sample['paths']:
            with Image.open(path) as image:
                image.load()
                if image.mode!='RGB': raise ValueError(f'Non-RGB image {path}')
                frames.append(np.array(image.resize((size,size),Image.Resampling.BILINEAR),copy=True))
        clips.append(np.stack(frames))
    rgb=torch.from_numpy(np.stack(clips)).permute(0,1,4,2,3).float()/255
    return (rgb-torch.tensor([.485,.456,.406]).view(1,1,3,1,1))/torch.tensor([.229,.224,.225]).view(1,1,3,1,1)


def configurations():
    base={'model':'P3','size':96,'batch':8,'precision':'bf16','lr':1e-4,
          'profile_weight':.1,'dropout':.12,'seed':42,'steps':3}
    cases=[{**base,'model':model} for model in ['B1','B2','B3','B4','P1','P2','P3']]
    for size in (96,160,224):
        for batch in (8,16):
            for precision in ('fp32','bf16'):
                case={**base,'size':size,'batch':batch,'precision':precision}
                if case not in cases: cases.append(case)
    for key,values in {'lr':[3e-5,3e-4],'profile_weight':[.03,.3],
                       'dropout':[0.,.25],'seed':[7,13]}.items():
        cases.extend({**base,key:value} for value in values)
    return cases


def run(preflight=False):
    if preflight:
        if not all(json.loads((OUT/f'scenario36_{m}_audit.json').read_text())['passed'] for m in ('gps','pwr','rgb180')):
            raise RuntimeError('Scenario 36 must pass its complete file audit first')
    else:
        gate=json.loads((OUT/'integrity_summary.json').read_text())
        if not gate['passed']: raise RuntimeError('Full integrity audit must pass first')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; GPU test not silently moved to CPU')
    torch.set_num_threads(4)
    selected,gps=prepare((36,), 'preflight_windows.json') if preflight else prepare()
    labels=torch.tensor([s['label'] for s in selected],device='cuda')
    profiles=torch.tensor(np.stack([s['profile'] for s in selected]),dtype=torch.float32,device='cuda')
    cases=[{**configurations()[6],'batch':4}] if preflight else configurations()
    images={size:load_images(selected,size) for size in sorted({case['size'] for case in cases})}
    records=[]
    for index,case in enumerate(cases):
        model=optimizer=outputs=loss=None
        random.seed(case['seed']); np.random.seed(case['seed']); torch.manual_seed(case['seed']); torch.cuda.manual_seed_all(case['seed'])
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        result={'configuration':case,'purpose':'training execution test only; no generalization claim'}
        start=time.monotonic()
        try:
            model=create_model(case['model'],d_model=256,gru_layers=3,fusion_heads=8,fusion_layers=3,
                               freeze_until='layer2',dropout=case['dropout'],head_hidden=512).cuda()
            total,trainable=count_parameters(model)
            optimizer=torch.optim.AdamW(model.parameters(),lr=case['lr'],weight_decay=3e-4)
            criterion=MultiTaskLoss(lambda_prof=case['profile_weight'],lambda_smooth=.01,
                                    lambda_rank=.05,use_huber=case['model']=='P2')
            batch=case['batch']; rgb=images[case['size']][:batch].cuda(); g=gps[:batch].cuda()
            losses=[]; grad_norms=[]
            for step in range(case['steps']):
                model.train(); optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=case['precision']=='bf16'):
                    outputs=model(rgb,g)
                    loss,details=criterion(outputs,labels[:batch],profiles[:batch])
                if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                if float(norm)==0: raise ValueError('Zero gradients')
                optimizer.step()
                losses.append(float(loss));grad_norms.append(float(norm))
            torch.cuda.synchronize()
            result.update(status='passed',losses=losses,unclipped_gradient_norms=grad_norms,
                          total_parameters=total,trainable_parameters=trainable,
                          peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                          elapsed_seconds=time.monotonic()-start)
            if case['model']=='P3' and case['size']==96 and case['batch']==8 and case['precision']=='bf16' and case['seed']==42 and case['lr']==1e-4 and case['profile_weight']==.1 and case['dropout']==.12:
                checkpoint=OUT/'smoke_checkpoint.pt'
                tmp=checkpoint.with_suffix('.tmp')
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':3,
                            'configuration':case,'torch_rng':torch.get_rng_state(),
                            'cuda_rng':torch.cuda.get_rng_state_all()},tmp)
                tmp.replace(checkpoint)
                model.eval()
                with torch.no_grad(): expected=model(rgb,g)['pred_profile'].clone()
                state=torch.load(checkpoint,weights_only=False)
                with torch.no_grad(): next(p for p in model.parameters() if p.requires_grad).add_(1.)
                model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
                with torch.no_grad(): restored=model(rgb,g)['pred_profile']
                result['checkpoint_output_max_abs_difference']=float((expected-restored).abs().max())
                if not torch.equal(expected,restored): raise ValueError('Checkpoint restore mismatch')
                del state,expected,restored
        except torch.cuda.OutOfMemoryError:
            result.update(status='out_of_memory',elapsed_seconds=time.monotonic()-start)
        except Exception as error:
            result.update(status='failed',error=f'{type(error).__name__}: {error}')
        records.append(result)
        save_json(OUT/('preflight_scenario36.json' if preflight else 'training_smoke.json'),{'device':torch.cuda.get_device_name(),
            'torch':torch.__version__,'cases':records,'complete':index==len(cases)-1})
        print(f'Smoke {index+1}/{len(cases)}: {case} -> {result["status"]}',flush=True)
        del model,optimizer,outputs,loss
        gc.collect();torch.cuda.empty_cache()
    if any(r['status']=='failed' for r in records): raise SystemExit('Smoke tests failed; inspect report')
    from src.readiness_report import main as report
    report()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-s36',action='store_true')
    args=parser.parse_args()
    run(args.preflight_s36)
