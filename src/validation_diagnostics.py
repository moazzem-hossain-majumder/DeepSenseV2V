"""Frozen best-checkpoint validation diagnostics. Never consumes calib/test inputs.

Zero-input interventions are sensitivity checks, not retrained modality ablations.
"""
import hashlib
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from src import multiscenario_pilot as base

SOURCE = base.ROOT/'results/multiscenario_extended_v1'
OUT = base.ROOT/'results/validation_diagnostics_v1'
TASKS = ['B1','P1','P3','P3_noGPS','P3_noRGB']
KS = (1,3,5,10,20,32,64)


def summarize(logits, profiles, labels):
    order = np.argsort(-logits, axis=1, kind='stable')
    best = profiles.max(axis=1)
    result = {'n':len(labels)}
    for k in KS:
        selected = order[:,:k]
        loss = best-np.take_along_axis(profiles, selected, axis=1).max(axis=1)
        result[f'top{k}'] = float((selected==labels[:,None]).any(axis=1).mean())
        result[f'within_3db_at_{k}'] = float((loss<=3).mean())
        result[f'mean_power_loss_db_at_{k}'] = float(loss.mean())
    result['power_loss_db'] = result['mean_power_loss_db_at_1']
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base.OUT = OUT
    base.CONFIG = {**base.CONFIG, 'epochs':1}
    if json.loads((SOURCE/'status.json').read_text())['status'] != 'complete':
        raise ValueError('Stage 2 not complete')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    samples = base.prepare()
    source_protocol = json.loads((SOURCE/'protocol.json').read_text())
    current_protocol = json.loads((OUT/'protocol.json').read_text())
    if any(current_protocol[k] != source_protocol[k] for k in ('csv_sha256','manifest_sha256','scaler_min','scaler_range')):
        raise ValueError('Data/splits/scaling changed from training')
    del samples['train']
    mean = torch.tensor([.485,.456,.406],device='cuda').view(1,1,3,1,1)
    std = torch.tensor([.229,.224,.225],device='cuda').view(1,1,3,1,1)
    for task in TASKS:
        if (OUT/f'{task}_finished.json').exists(): continue
        name = task.split('_')[0]
        path = SOURCE/f'{name}_best.pt'
        saved = torch.load(path,map_location='cpu',weights_only=False)
        model = base.create_model(name,d_model=256,gru_layers=3,fusion_heads=8,
            fusion_layers=3,freeze_until='layer2',dropout=.12,head_hidden=512).cuda()
        model.load_state_dict(saved['model']); model.eval()
        provenance = dict(checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          selected_epoch=saved['epoch'], selection='validation top-1')
        del saved
        loader = DataLoader(base.Clips(samples['val'],name!='B1'),batch_size=16,
                            num_workers=2,pin_memory=True)
        parts = {k:[] for k in ('logits','profiles','labels','scenarios')}
        start = time.monotonic()
        with torch.inference_mode():
            for batch,(rgb,gps,labels,profiles,scenarios) in enumerate(loader):
                gps = gps.cuda()
                rgb = ((rgb.cuda().permute(0,1,4,2,3).float()/255-mean)/std) if name!='B1' else None
                if task.endswith('noGPS'): gps.zero_()
                if task.endswith('noRGB'): rgb.zero_()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits = model(rgb,gps)['logits'].float()
                if not torch.isfinite(logits).all(): raise ValueError('Nonfinite predictions')
                for key,value in zip(parts,(logits.cpu().numpy(),profiles.numpy(),labels.numpy(),scenarios.numpy())):
                    parts[key].append(value)
                if batch%25==0:
                    base.status(status='validating',model=task,epoch=1,epochs=1,
                        batch=batch+1,batches=len(loader),elapsed_epoch_seconds=time.monotonic()-start)
        arrays = {k:np.concatenate(v) for k,v in parts.items()}
        report = dict(task=task,provenance=provenance,scope='Validation only; zero-input tests are not retrained ablations.',
            val=summarize(arrays['logits'],arrays['profiles'],arrays['labels']),
            by_scenario={str(s):summarize(arrays['logits'][arrays['scenarios']==s],
                arrays['profiles'][arrays['scenarios']==s],arrays['labels'][arrays['scenarios']==s]) for s in (36,37,38,39)},
            seconds=time.monotonic()-start,epoch=1)
        temp = OUT/f'{task}_predictions.tmp'
        with temp.open('wb') as f: np.savez_compressed(f,**arrays)
        temp.replace(OUT/f'{task}_predictions.npz')
        base.save_json(OUT/f'{task}_report.json',report)
        base.save_json(OUT/f'{task}_history.json',[report])
        base.save_json(OUT/f'{task}_finished.json',dict(reason='validation_complete',epoch=1))
        print(json.dumps({'task':task, 'val':report['val']}),flush=True)
        del model,loader,arrays,parts
        torch.cuda.empty_cache()
    reports = [json.loads((OUT/f'{t}_report.json').read_text()) for t in TASKS]
    text = ['# Frozen-checkpoint validation diagnostics','',
            'Validation data only. Test/calibration reserved. Checkpoint selection used validation top-1.',
            'Shortlist results use offline measured powers; they are not an online probing-cost claim.','']
    for r in reports:
        v=r['val']
        text.append(f"- {r['task']} (checkpoint epoch {r['provenance']['selected_epoch']}): Top-1 {v['top1']:.2%}, Top-5 {v['top5']:.2%}, mean top-1 power loss {v['power_loss_db']:.2f} dB; within 3 dB with 5 candidates: {v['within_3db_at_5']:.2%}.")
    text += ['', 'Zeroed normalized images correspond to ImageNet mean pixels. Zeroed scaled GPS is a sensitivity intervention, not a physical missing-GPS model. These distribution shifts do not replace retrained ablations.']
    (OUT/'REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    base.status(status='complete',models=TASKS,epochs_each=1,scope='Validation diagnostics complete; test/calibration untouched.')


if __name__=='__main__':
    try: main()
    except Exception as exc:
        base.OUT=OUT
        base.status(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
