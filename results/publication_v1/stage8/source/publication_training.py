import json
import math
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.publication_plan import ROOT,VARIANTS,config
from src.publication_data import Clips,load,partitions,scale,digest
from src.publication_metrics import objective,metrics
from src.models import create_model
from src.acquire_scenarios import save_json
from src.multiscenario_pilot import atomic_checkpoint


def build_model(cfg):
    model=create_model(VARIANTS[cfg['variant']],d_model=256,gru_layers=3,fusion_heads=8,
                       fusion_layers=3,freeze_until='layer2',dropout=cfg['dropout'],head_hidden=512)
    return model.cuda()


def frozen_bn(model):
    for layer in model.modules():
        if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm): layer.eval()


def inputs(rgb,gps,variant,brightness=0.):
    gps=gps.cuda()
    if variant=='gps': return None,gps
    rgb=rgb.cuda().permute(0,1,4,2,3).float()/255
    if brightness:
        factor=1+(torch.rand((len(rgb),1,1,1,1),device='cuda')*2-1)*brightness
        rgb=(rgb*factor).clamp(0,1)
    mean=torch.tensor([.485,.456,.406],device='cuda').view(1,1,3,1,1)
    std=torch.tensor([.229,.224,.225],device='cuda').view(1,1,3,1,1)
    return (rgb-mean)/std,gps


def loader(manifest,gps,profiles,indices,cfg,train=False):
    return DataLoader(Clips(manifest,gps,profiles,indices,cfg['variant']!='gps'),
        batch_size=64 if cfg['variant']=='gps' else cfg['batch'],shuffle=train,
        num_workers=cfg['workers'],pin_memory=True,persistent_workers=True)


def predict(model,batches,cfg,progress,split='val'):
    model.eval();parts={k:[] for k in ('logits','profiles','labels','scenarios','indices')}
    start=time.monotonic()
    with torch.inference_mode():
        for batch,(rgb,gps,labels,profiles,scenarios,indices) in enumerate(batches):
            rgb,gps=inputs(rgb,gps,cfg['variant'])
            with torch.autocast('cuda',dtype=torch.bfloat16): output=model(rgb,gps)['logits'].float()
            if not torch.isfinite(output).all(): raise ValueError('Nonfinite model output')
            for key,value in zip(parts,(output.cpu().numpy(),profiles.numpy(),labels.numpy(),scenarios.numpy(),indices.numpy())):
                parts[key].append(value)
            if batch%25==0: progress(status='evaluating_'+split,model=cfg['variant'],batch=batch+1,batches=len(batches),elapsed_epoch_seconds=time.monotonic()-start)
    return {k:np.concatenate(v) for k,v in parts.items()}


def save_arrays(path,arrays):
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as f: np.savez_compressed(f,**arrays)
    temp.replace(path)


def train(plan,progress):
    cfg=config(plan);folder=ROOT/plan['directory'];folder.mkdir(parents=True,exist_ok=True)
    manifest,gps,profiles,receipt=load()
    train_idx,val_idx,unseen_idx=partitions(manifest,cfg['holdout'])
    gps,minimum,span=scale(gps,train_idx)
    save_json(folder/'config.json',cfg)
    torch.set_num_threads(4)
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    model=build_model(cfg);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    latest=folder/'latest.pt';history=[];start_epoch=0
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=False)
        if saved['config']!=cfg or saved['data']!=receipt: raise ValueError('Checkpoint configuration/data mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        history=saved['history'];start_epoch=saved['epoch'];del saved
    train_loader=loader(manifest,gps,profiles,train_idx,cfg,True)
    val_loader=loader(manifest,gps,profiles,val_idx,cfg)
    for epoch in range(start_epoch,cfg['epochs']):
        if (ROOT/'pause.flag').exists(): progress(status='paused_by_user');return
        # Per-epoch RNG reset makes interrupted epoch replay independent of worker startup.
        seed=cfg['seed']*1000+epoch
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        lr=cfg['lr']*(.1+.9*.5*(1+math.cos(math.pi*epoch/max(1,cfg['epochs']-1))))
        for group in optimizer.param_groups: group['lr']=lr
        model.train();frozen_bn(model);began=time.monotonic();loss_sum=0.;n=0
        for batch,(rgb,x,labels,power,_,__) in enumerate(train_loader):
            rgb,x=inputs(rgb,x,cfg['variant'],cfg['brightness'])
            labels=labels.cuda();power=power.cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                outputs=model(rgb,x);loss=objective(outputs,labels,power,cfg['variant'])
            if not torch.isfinite(loss): raise ValueError('Nonfinite objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            loss_sum+=float(loss)*len(labels);n+=len(labels)
            if batch%25==0: progress(status='training',model=cfg['variant'],epoch=epoch+1,epochs=cfg['epochs'],batch=batch+1,batches=len(train_loader),elapsed_epoch_seconds=time.monotonic()-began)
        def val_progress(**values): progress(epoch=epoch+1,epochs=cfg['epochs'],**values)
        prediction=predict(model,val_loader,cfg,val_progress)
        val=metrics(prediction['logits'],prediction['profiles'],prediction['labels'],prediction['scenarios'])
        row=dict(epoch=epoch+1,train_loss=loss_sum/n,lr=lr,val=val,seconds=time.monotonic()-began)
        history.append(row)
        state=dict(config=cfg,epoch=epoch+1,model=model.state_dict(),optimizer=optimizer.state_dict(),
                   data=receipt,scaler_min=minimum.tolist(),scaler_span=span.tolist(),history=history)
        atomic_checkpoint(latest,state)
        if val['macro_success5']>=max(h['val']['macro_success5'] for h in history):
            atomic_checkpoint(folder/'best.pt',state);save_arrays(folder/'val_predictions.npz',prediction)
        save_json(folder/(cfg['variant']+'_history.json'),history)
        print(json.dumps(row),flush=True)
    if cfg['holdout']:
        saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);del saved
        held=predict(model,loader(manifest,gps,profiles,unseen_idx,cfg),cfg,progress,'unseen_scenario_val')
        save_arrays(folder/'unseen_predictions.npz',held)
        save_json(folder/'unseen_report.json',metrics(held['logits'],held['profiles'],held['labels'],held['scenarios']))
    best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    save_json(folder/'result.json',dict(config=cfg,best_epoch=best['epoch'],
        val=best['history'][-1]['val'],checkpoint_sha256=digest(folder/'best.pt'),
        scope='Validation-selected development result. No test/calibration evaluation.'))
    save_json(folder/(cfg['variant']+'_finished.json'),dict(reason='fixed_epoch_budget'))
    progress(status='complete',model=cfg['variant'])
