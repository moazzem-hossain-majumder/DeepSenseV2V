import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.sensitivity_plan import ROOT,PREVIOUS
from src.publication_data import load,partitions,Clips,digest
from src.publication_training import predict,save_arrays
from src.factorial_training import FactorialReranker
from src.camera_data import CameraClips,verify
from src.publication_metrics import metrics,per_sample
from src.conservative_training import BoundedCorrection,ContextLoader
from src.conservative_geometry import key
from src.acquire_scenarios import save_json

def donors(manifest,indices):
    """Deterministic clip permutation within scenario, never borrowing test/calib."""
    result=np.asarray(indices).copy();rng=np.random.default_rng(6201)
    for s in np.unique(manifest.scenario.to_numpy()[indices]):
        loc=np.flatnonzero(manifest.scenario.to_numpy()[indices]==s)
        order=rng.permutation(loc)
        if len(order)>1:result[order]=np.asarray(indices)[np.roll(order,1)]
    return result

class InterventionClips(CameraClips):
    def __init__(self,m,g,p,indices,mode):
        super().__init__(m,g,p,indices,'both');self.mode=mode
        if mode=='shuffled':self.x[self.indices]=self.x[donors(m,self.indices)].copy()
    def __getitem__(self,i):
        rgb,*other=super().__getitem__(i)
        if self.mode=='gray':rgb=np.full_like(rgb,127)
        return (rgb,*other)

def run(plan,folder,progress):
    torch.set_num_threads(4)
    stage=plan['source'];source=ROOT/plan['source_directory'];verify()
    saved=torch.load(source/'best.pt',map_location='cpu',weights_only=False);cfg=saved['config']
    manifest,gps,power,receipt=load()
    if saved['data']!=receipt:raise ValueError('Checkpoint data changed')
    train,val,held=partitions(manifest,cfg['holdout']);ids=held if cfg['holdout'] else val
    if not (manifest.split.iloc[ids]=='val').all():raise ValueError('Interventions restricted to validation')
    gps=(gps-np.asarray(saved['scaler_min'],np.float32))/np.asarray(saved['scaler_span'],np.float32)
    model=FactorialReranker(cfg['k'],True,cfg['bounded']).cuda();model.load_state_dict(saved['model']);del saved
    context=np.load(PREVIOUS/'stage44/context.npy');minimum=context[train].min(0);span=context[train].max(0)-minimum;span[span==0]=1
    context=(context-minimum)/span
    prior=np.load(PREVIOUS/f'stage44/prior_{key(cfg["holdout"])}.npy',mmap_mode='r')
    result=dict(source_stage=stage,checkpoint_sha256=digest(source/'best.pt'),holdout=cfg['holdout'],
        modes={},scope='Fixed checkpoint sensitivity only, not retraining or independent test evidence.')
    clean=None
    try:
        for j,mode in enumerate(('original','gray','shuffled')):
            path=folder/f'{mode}.npz'
            if path.exists():
                with np.load(path) as a:prediction={k:a[k] for k in a.files}
                if not np.array_equal(prediction['indices'],ids):raise ValueError('Resumed prediction rows changed')
            else:
                batches=DataLoader(InterventionClips(manifest,gps,power,ids,mode),batch_size=cfg['batch'],num_workers=cfg['workers'],pin_memory=True)
                def callback(**kw):
                    kw.update(status=f'intervention_{mode}',model='analysis');progress(**kw)
                prediction=predict(model,ContextLoader(batches,context,prior),cfg,callback)
                save_arrays(path,prediction)
            scores=metrics(prediction['logits'],prediction['profiles'],prediction['labels'],prediction['scenarios'])
            outcomes=per_sample(prediction['logits'],prediction['profiles'],prediction['labels'])['success']
            if mode=='original':clean=outcomes
            result['modes'][mode]=dict(metrics=scores,helped_vs_original=int((outcomes&~clean).sum()),harmed_vs_original=int((clean&~outcomes).sum()))
            progress(status=f'finished_{mode}',model='analysis',batch=j+1,batches=3)
        with np.load(source/'val_predictions.npz') as a,np.load(folder/'original.npz') as b:
            if not np.array_equal(a['indices'],b['indices']):raise ValueError('Reference IDs differ')
            error=float(np.max(np.abs(a['logits']-b['logits'])))
            result['original_reproduction_max_logit_error']=error
            if error>1e-4:raise ValueError(f'Original reproduction mismatch: {error}')
        save_json(folder/'interventions.json',result)
    finally:prior._mmap.close()
