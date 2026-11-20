import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.camera_plan import ROOT,OUT,PREVIOUS
from src.camera_data import CameraClips,verify
from src.publication_data import partitions,digest
from src.publication_plan import COMMON
from src.conservative_training import BoundedCorrection,ContextLoader,correction_loss
from src.conservative_geometry import key
from src import publication_training as engine
from src.acquire_scenarios import save_json

class CameraCorrection(BoundedCorrection):
    def __init__(self,backbone,mode):super().__init__(backbone);self.mode=mode
    def forward(self,rgb,gps):
        core=gps[:,:,:9];context=gps[:,-1,9:17];prior=gps[:,-1,17:]
        a=self.backbone(rgb,core)['logits'];b=self.context_head(context)
        raw=a+b if self.mode=='tanh_bf16' else a.float()+b.float()
        correction=.25*raw if self.mode=='linear' else .25*torch.tanh(raw)
        return dict(logits=prior+correction,correction=correction,raw_correction=raw)

def configuration(plan):
    return {**COMMON,'variant':'fusion_ce','batch':8,'epochs':plan['epochs'],'seed':plan['seed'],'holdout':plan['holdout'],
        'camera':plan['camera'],'correction':plan['correction'],'correction_scale':.25,'correction_l2_weight':.1,
        'camera_source_sha256':digest(ROOT/'src/camera_training.py'),'rear_cache_sha256':digest(ROOT/'data/processed/camera_v1/receipt.json'),
        'geometry_report_sha256':digest(PREVIOUS/'stage44/report.json')}

def make_loader(m,g,p,indices,cfg,context,prior,train=False):
    return ContextLoader(DataLoader(CameraClips(m,g,p,indices,cfg['camera']),batch_size=cfg['batch'],shuffle=train,
        num_workers=cfg['workers'],pin_memory=True,persistent_workers=cfg['workers']>0),context,prior)

def run(plan,progress):
    verify();cfg=configuration(plan);context=np.load(PREVIOUS/'stage44/context.npy')
    prior=np.load(PREVIOUS/f'stage44/prior_{key(cfg["holdout"])}.npy',mmap_mode='r')
    original=(engine.loader,engine.build_model,engine.config,engine.objective)
    def loader(m,g,p,indices,c,train=False):
        ti,_,__=partitions(m,c['holdout']);lo=context[ti].min(0);span=context[ti].max(0)-lo;span[span==0]=1
        return make_loader(m,g,p,indices,c,(context-lo)/span,prior,train)
    engine.loader=loader;engine.build_model=lambda c:CameraCorrection(original[1](c),c['correction']).cuda()
    engine.config=lambda _:cfg;engine.objective=correction_loss
    try:engine.train(plan,progress)
    finally:
        engine.loader,engine.build_model,engine.config,engine.objective=original
        prior._mmap.close()

def health(plan):
    # Fixed training sample only; diagnosis must not consume held-scenario or test labels.
    folder=ROOT/plan['directory'];saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);cfg=saved['config']
    m,g,p,_=engine.load();ti,_,__=partitions(m,cfg['holdout'])
    ids=np.concatenate([rows[np.linspace(0,len(rows)-1,8,dtype=int)] for s in sorted(m.scenario.iloc[ti].unique()) for rows in [ti[m.scenario.to_numpy()[ti]==s]]])
    g=(g-np.asarray(saved['scaler_min'],np.float32))/np.asarray(saved['scaler_span'],np.float32)
    model=CameraCorrection(engine.build_model(cfg),cfg['correction']).cuda();model.load_state_dict(saved['model']);del saved
    model.eval();torch.set_num_threads(4);old_cudnn=torch.backends.cudnn.enabled;torch.backends.cudnn.enabled=False
    context=np.load(PREVIOUS/'stage44/context.npy');lo=context[ti].min(0);span=context[ti].max(0)-lo;span[span==0]=1
    prior=np.load(PREVIOUS/f'stage44/prior_{key(cfg["holdout"])}.npy',mmap_mode='r');records=[]
    try:
        for rgb,gps,labels,power,_,__ in make_loader(m,g,p,ids,{**cfg,'workers':0},(context-lo)/span,prior):
            model.zero_grad(set_to_none=True);image,x=engine.inputs(rgb,gps,'fusion_ce')
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(image,x);loss=correction_loss(out,labels.cuda(),power.cuda(),'fusion_ce')
            loss.backward()
            norm=float(sum((q.grad.float().square().sum() for q in model.backbone.rgb_enc.parameters() if q.grad is not None),torch.tensor(0.,device='cuda')).sqrt())
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                gray,_=engine.inputs(torch.full_like(rgb,127),gps,'fusion_ce');other=model(gray,x)['correction'].float()
            records.append(dict(rgb_gradient_norm=norm,gray_delta=float((other-out['correction'].detach().float()).abs().mean()),
                raw_abs_gt3=float((out['raw_correction'].detach().abs()>3).float().mean()),correction_abs_max=float(out['correction'].detach().abs().max())))
        result={k:float(np.mean([r[k] for r in records])) for k in records[0]}
        result.update(finite=all(np.isfinite(v) for r in records for v in r.values()),sample_indices=ids.tolist(),scope='Training-only diagnostic; image sensitivity is not proof of useful visual reasoning.')
        save_json(folder/'health.json',result);return result
    finally:prior._mmap.close();torch.backends.cudnn.enabled=old_cudnn
