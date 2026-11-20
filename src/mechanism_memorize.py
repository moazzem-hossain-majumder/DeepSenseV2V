import json
import numpy as np
import torch
from torch.nn import functional as F
from src.mechanism_plan import OUT,ROOT
from src.rerank_training import loss as original_loss
from src.multiscenario_pilot import atomic_checkpoint
from src.publication_data import digest
from src.acquire_scenarios import save_json

def measure(scores,candidates,power):
    order=torch.argsort(scores,descending=True,stable=True)
    chosen=candidates.gather(1,order[:,:5]);selected_power=power.gather(1,chosen).max(1).values
    target=power.gather(1,candidates).argmax(1)
    return dict(success5=float(((power.max(1).values-selected_power)<=3).float().mean()),
        best_candidate_top1=float((order[:,0]==target).float().mean()),loss_db=float((power.max(1).values-selected_power).mean()))

def run(plan,folder,progress,steps=1000):
    torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    source=OUT/'stage117';receipt=json.loads((source/'sample.json').read_text())
    if digest(source/'sample.npz')!=receipt['sample_sha256'] or digest(source/'initial_head.pt')!=receipt['head_sha256']:raise ValueError('Sample changed')
    with np.load(source/'sample.npz') as a:
        x=torch.from_numpy(a['descriptors']).cuda();candidates=torch.from_numpy(a['candidates']).cuda();prior=torch.from_numpy(a['prior']).cuda();power=torch.from_numpy(a['power']).cuda()
    target=power.gather(1,candidates).argmax(1);mode=plan['mode']
    head=torch.nn.Sequential(torch.nn.Linear(569,128),torch.nn.GELU(),torch.nn.Linear(128,1)).cuda()
    head.load_state_dict(torch.load(source/'initial_head.pt',weights_only=True,map_location='cpu'))
    free=torch.nn.Parameter(torch.zeros_like(prior));params=[free] if mode=='per_example_control' else head.parameters()
    optimizer=torch.optim.AdamW(params,lr=.001,weight_decay=0)
    history=[];start=0;latest=folder/'latest.pt'
    if latest.exists():
        state=torch.load(latest,weights_only=False,map_location='cpu')
        if state['mode']!=mode or state['receipt']!=receipt:raise ValueError('Diagnostic checkpoint mismatch')
        head.load_state_dict(state['head']);free.data.copy_(state['free'].cuda());optimizer.load_state_dict(state['optimizer']);history=state['history'];start=state['step']
    initial=measure(prior,candidates,power)
    for step in range(start,steps):
        if (ROOT/'pause.flag').exists():progress(status='paused_by_user',model='analysis');return
        optimizer.zero_grad(set_to_none=True);raw=free if mode=='per_example_control' else head(x).squeeze(-1)
        residual=2*F.softsign(raw) if mode in ('current_loss','bounded_ce') else raw
        scores=prior+residual
        loss=original_loss(dict(candidates=candidates,selected=scores,residual=residual),target,power,'fusion_ce') if mode=='current_loss' else F.cross_entropy(scores,target)
        if not torch.isfinite(loss):raise ValueError('Nonfinite diagnostic loss')
        loss.backward();torch.nn.utils.clip_grad_norm_([free] if mode=='per_example_control' else head.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if (step+1)%50==0 or step+1==steps:
            with torch.no_grad():
                raw=free if mode=='per_example_control' else head(x).squeeze(-1)
                residual=2*F.softsign(raw) if mode in ('current_loss','bounded_ce') else raw
                metrics=measure(prior+residual,candidates,power)
            history.append(dict(step=step+1,pre_update_loss=float(loss),**metrics))
            atomic_checkpoint(latest,dict(mode=mode,receipt=receipt,head=head.state_dict(),free=free.detach(),optimizer=optimizer.state_dict(),step=step+1,history=history))
            save_json(folder/'curve.json',history);progress(status='memorizing_training_sample',model='analysis',batch=step+1,batches=steps)
    save_json(folder/'result.json',dict(mode=mode,initial=initial,final=history[-1],max_success5=max(r['success5'] for r in history),
        scope='Training memorization only, biased fixed sample. 1000 full-batch FP32 AdamW steps at 1e-3; no generalization claim. Per-example control cannot predict unseen data.'))
