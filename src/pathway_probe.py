import json
from contextlib import nullcontext
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.pathway_plan import PREVIOUS
from src.publication_data import load,partitions,Clips,digest
from src.publication_training import build_model,inputs
from src.conservative_training import BoundedCorrection,ContextLoader,correction_loss
from src.conservative_geometry import key
from src.acquire_scenarios import save_json

def sample_training(m,holdout,per_scenario=32):
    train,_,__=partitions(m,holdout);ids=[]
    for s in sorted(m.scenario.iloc[train].unique()):
        group=train[m.scenario.to_numpy()[train]==s]
        ids.extend(group[np.linspace(0,len(group)-1,min(per_scenario,len(group)),dtype=int)].tolist())
    return np.array(ids,dtype=int)

def grad_norm(module):
    return float(sum((p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None),torch.tensor(0.,device='cuda')).sqrt())

def probe(plan,folder,progress):
    torch.set_num_threads(4)
    # Native GRU permits backward in eval mode without enabling recurrent dropout.
    torch.backends.cudnn.enabled=False
    source=PREVIOUS/f'stage{plan["source"]}'
    saved=torch.load(source/'best.pt',map_location='cpu',weights_only=False);cfg=saved['config']
    m,g,p,receipt=load()
    if saved['data']!=receipt:raise ValueError('Checkpoint data mismatch')
    ids=sample_training(m,cfg['holdout']);train,_,__=partitions(m,cfg['holdout'])
    g=(g-np.array(saved['scaler_min'],np.float32))/np.array(saved['scaler_span'],np.float32)
    model=BoundedCorrection(build_model(cfg)).cuda();model.load_state_dict(saved['model']);del saved
    model.eval();model.disable_correction=False
    context=np.load(PREVIOUS/'stage44/context.npy');lo=context[train].min(0);span=context[train].max(0)-lo;span[span==0]=1
    prior=np.load(PREVIOUS/f'stage44/prior_{key(cfg["holdout"])}.npy',mmap_mode='r')
    captures={};hooks=[]
    for name,mod in [('rgb',model.backbone.rgb_enc),('gps',model.backbone.gps_enc),('fusion',model.backbone.fusion),('head',model.backbone.cls_head),('context',model.context_head)]:
        hooks.append(mod.register_forward_hook(lambda module,args,output,n=name:captures.__setitem__(n,output.detach().float())))
    records=[];feature_chunks={mode:[] for mode in ('fp32','bf16')}
    try:
        batches=ContextLoader(DataLoader(Clips(m,g,p,ids,True),batch_size=8,num_workers=0),(context-lo)/span,prior)
        for bi,(rgb,gps,labels,power,sc,indices) in enumerate(batches):
            for mode in ('fp32','bf16'):
                image,x=inputs(rgb,gps,'fusion_ce');image.requires_grad_(True);x.requires_grad_(True)
                model.zero_grad(set_to_none=True)
                precision=lambda:torch.autocast('cuda',dtype=torch.bfloat16) if mode=='bf16' else nullcontext()
                with precision():
                    output=model(image,x);loss=correction_loss(output,labels.cuda(),power.cuda(),'fusion_ce')
                if not torch.isfinite(loss):raise ValueError('Nonfinite probe loss')
                raw=captures['head']+captures['context'];features=captures['rgb'].mean(1).cpu().numpy();feature_chunks[mode].append(features)
                original=output['correction'].detach().float()
                loss.backward()
                row=dict(precision=mode,batch=bi,n=len(labels),loss=float(loss),
                    saturation_fraction=float((raw.abs()>3).float().mean()),
                    zero_tanh_derivative_fraction=float((1-(original/.25).square()<=1e-7).float().mean()),
                    rgb_gradient_norm=grad_norm(model.backbone.rgb_enc),gps_gradient_norm=grad_norm(model.backbone.gps_enc),
                    fusion_gradient_norm=grad_norm(model.backbone.fusion),head_gradient_norm=grad_norm(model.backbone.cls_head),
                    rgb_input_gradient_norm=float(image.grad.float().norm()),gps_input_gradient_norm=float(x.grad[:,:,:9].float().norm()))
                if not all(np.isfinite(v) for v in row.values() if isinstance(v,(float,int))):raise ValueError('Nonfinite gradient statistic')
                with torch.no_grad(),precision():
                    gray,_=inputs(torch.full_like(rgb,127),gps,'fusion_ce')
                    changed=model(gray,x.detach())['correction'].float()
                row['gray_correction_delta']=float((original-changed).abs().mean());records.append(row)
                progress(status=f'probing_{mode}',model='analysis',batch=2*bi+(1 if mode=='fp32' else 2),batches=2*len(batches))
        summary={}
        for mode in ('fp32','bf16'):
            subset=[r for r in records if r['precision']==mode];weights=[r['n'] for r in subset]
            summary[mode]={k:float(np.average([r[k] for r in subset],weights=weights)) for k in subset[0] if k not in ('precision','batch','n')}
            summary[mode]['rgb_feature_std']=float(np.concatenate(feature_chunks[mode]).std(0).mean())
        save_json(folder/'probe.json',dict(source_stage=plan['source'],checkpoint_sha256=digest(source/'best.pt'),
            sampled_indices=ids.tolist(),training_scenarios=sorted(m.scenario.iloc[ids].unique().tolist()),summary=summary,batches=records,
            scope='Training-only sample, eval-mode gradients; dropout disabled, CuDNN disabled for eval-mode GRU backward, no optimizer steps. BF16 and FP32 use identical inputs and weights.'))
    finally:
        for h in hooks:h.remove()
        prior._mmap.close()
