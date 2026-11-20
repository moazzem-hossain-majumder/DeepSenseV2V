"""Resumable pooled training/validation pilot; never evaluates calibration/test.

Run via supervise_work --job pilot. This local blocked protocol is not an
official challenge reproduction. Interrupted epochs restart from their start.
"""
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import pickle
import random
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.acquire_scenarios import ROOT, RAW, save_json
from src.gps_features import extract_sequence_gps_features, GPSFeatureScaler
from src.models import create_model, MultiTaskLoss
from src.multiscenario_windows import build_windows

OUT = ROOT / 'results' / 'multiscenario_pilot_v1'
CONFIG = dict(models=['B1', 'P1', 'P3'], epochs=3, seed=42, size=96,
              lr=1e-4, weight_decay=3e-4, batch_gps=64, batch_rgb=16,
              precision='bf16', workers=2, protocol='blocked-original-sequence-v1')


def status(**kwargs):
    save_json(OUT/'status.json', dict(updated_utc=datetime.now(timezone.utc).isoformat(), **kwargs))


def prepare():
    samples = {'train': [], 'val': []}
    manifests = []
    hashes = {}
    for scenario in (36, 37, 38, 39):
        status(status='preparing', scenario=scenario)
        folder = RAW/f'scenario{scenario}'
        csv = folder/f'scenario{scenario}.csv'
        hashes[str(scenario)] = hashlib.sha256(csv.read_bytes()).hexdigest()
        frame = pd.read_csv(csv)
        with (folder/f'scenario{scenario}.p').open('rb') as f:
            raw = pickle.load(f)
        powers = np.concatenate([raw[f'unit1_pwr{i}'] for i in range(1, 5)], axis=1)
        windows = build_windows(frame, scenario)
        if windows.groupby('block_key')['split'].nunique().max() != 1:
            raise ValueError('Trajectory leakage')
        manifests.append(windows)
        for row in windows[windows.split.isin(samples)].itertuples():
            gps = extract_sequence_gps_features(frame, row.x_indices)
            profile = 10*np.log10(np.maximum(powers[row.y_index], 1e-12))
            if not np.isfinite(gps).all() or not np.isfinite(profile).all():
                raise ValueError('Nonfinite input')
            samples[row.split].append(dict(gps=gps, profile=profile.astype(np.float32),
                label=int(row.target_beam), scenario=scenario,
                paths=[str(folder/frame.unit1_rgb5.iloc[i]) for i in row.x_indices]))
    manifest = pd.concat(manifests, ignore_index=True)
    manifest.to_json(OUT/'split_manifest.jsonl', orient='records', lines=True)
    provenance_path = OUT/'continuation_provenance.json'
    if provenance_path.exists():
        expected = json.loads(provenance_path.read_text())['source_manifest_sha256']
        if hashlib.sha256((OUT/'split_manifest.jsonl').read_bytes()).hexdigest() != expected:
            raise ValueError('Continuation split differs from pilot')
    scaler = GPSFeatureScaler()
    scaler.fit(np.stack([s['gps'] for s in samples['train']]))
    for group in samples.values():
        for sample in group:
            sample['gps'] = scaler.transform(sample['gps'])
    save_json(OUT/'protocol.json', dict(config=CONFIG, csv_sha256=hashes,
        manifest_sha256=hashlib.sha256((OUT/'split_manifest.jsonl').read_bytes()).hexdigest(),
        counts=manifest.groupby(['scenario','split']).size().unstack().to_dict(),
        scaler_min=scaler.min_vals.tolist(), scaler_range=scaler.scale_diff.tolist(),
        features='Existing nine GPS features: velocity fields are displacement proxies, not m/s.',
        scope='All train/val windows; calibration/test reserved. RGB5 front view. No official split claim.'))
    return samples


class Clips(Dataset):
    def __init__(self, samples, images):
        self.samples, self.images = samples, images

    def __len__(self):
        return len(self.samples)

    @staticmethod
    @lru_cache(maxsize=512)
    def image(path):
        with Image.open(path) as im:
            im.load()
            if im.mode != 'RGB':
                raise ValueError(f'Unexpected image mode: {path}')
            return np.array(im.resize((CONFIG['size'], CONFIG['size']), Image.Resampling.BILINEAR), copy=True)

    def __getitem__(self, i):
        s = self.samples[i]
        rgb = np.stack([self.image(p) for p in s['paths']]) if self.images else np.zeros(1, np.uint8)
        return rgb, s['gps'], s['label'], s['profile'], s['scenario']


def atomic_checkpoint(path, value):
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)


def run_model(name, samples):
    random.seed(CONFIG['seed']); np.random.seed(CONFIG['seed']); torch.manual_seed(CONFIG['seed'])
    torch.cuda.manual_seed_all(CONFIG['seed'])
    model = create_model(name, d_model=256, gru_layers=3, fusion_heads=8,
        fusion_layers=3, freeze_until='layer2', dropout=.12, head_hidden=512).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    # Panel boundaries are discontinuities: disable legacy cross-panel smoothing in this pilot.
    criterion = MultiTaskLoss(lambda_prof=.1, lambda_smooth=0, lambda_rank=.05)
    loaders = {split: DataLoader(Clips(group, name!='B1'),
        batch_size=CONFIG['batch_gps'] if name=='B1' else CONFIG['batch_rgb'],
        shuffle=split=='train', num_workers=CONFIG['workers'], pin_memory=True,
        persistent_workers=True) for split, group in samples.items()}
    checkpoint = OUT/f'{name}_latest.pt'
    history, start = [], 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['config'] != CONFIG:
            raise ValueError('Configuration changed; use a new output directory')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
        history, start = saved['history'], saved['epoch']
    if CONFIG.get('patience') and history:
        best_epoch = max(history, key=lambda h: h['val']['top1'])['epoch']
        if start >= CONFIG['min_epochs'] and start-best_epoch >= CONFIG['patience']:
            save_json(OUT/f'{name}_finished.json', dict(reason='early_stopping', epoch=start))
            return True
    mean = torch.tensor([.485,.456,.406], device='cuda').view(1,1,3,1,1)
    std = torch.tensor([.229,.224,.225], device='cuda').view(1,1,3,1,1)
    for epoch in range(start, CONFIG['epochs']):
        if CONFIG.get('lr_after_epoch3'):
            for group in optimizer.param_groups:
                group['lr'] = CONFIG['lr'] if epoch < 3 else CONFIG['lr_after_epoch3']
        if (ROOT/'pause.flag').exists():
            status(status='paused_by_user', model=name)
            return False
        began = time.monotonic()
        result = {'epoch': epoch+1, 'model': name}
        for split, loader in loaders.items():
            model.train(split=='train')
            totals = {s: np.zeros(6, dtype=np.float64) for s in (36,37,38,39)}
            for batch, (rgb, gps, labels, profiles, scenarios) in enumerate(loader):
                gps, labels, profiles = gps.cuda(), labels.cuda(), profiles.cuda()
                rgb = ((rgb.cuda().permute(0,1,4,2,3).float()/255-mean)/std) if name!='B1' else None
                with torch.set_grad_enabled(split=='train'), torch.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(rgb, gps)
                    loss, _ = criterion(outputs, labels, profiles)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss')
                if split=='train':
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                with torch.no_grad():
                    top = outputs['logits'].float().topk(5, dim=1).indices
                    hit = top.eq(labels[:,None])
                    # Positive dB loss: oracle power minus selected power (lower is better).
                    apl = profiles.max(1).values-profiles.gather(1,top[:,:1]).squeeze(1)
                    for scenario in totals:
                        mask = scenarios.to('cuda')==scenario
                        n = int(mask.sum())
                        totals[scenario] += [n, float(loss)*n, int(hit[mask,:1].sum()),
                            int(hit[mask,:3].any(1).sum()), int(hit[mask].any(1).sum()), float(apl[mask].sum())]
                if batch % 25 == 0:
                    status(status='training' if split=='train' else 'validating', model=name,
                        epoch=epoch+1, epochs=CONFIG['epochs'], batch=batch+1, batches=len(loader),
                        elapsed_epoch_seconds=time.monotonic()-began, completed_epochs=history)
            def metrics(t):
                return dict(zip(['n','loss','top1','top3','top5','power_loss_db'],
                    [int(t[0]), *[float(x/t[0]) for x in t[1:]]]))
            result[split] = metrics(sum(totals.values()))
            result[split+'_by_scenario'] = {str(s):metrics(t) for s,t in totals.items()}
        result['seconds'] = time.monotonic()-began
        history.append(result)
        state = dict(config=CONFIG, model=model.state_dict(), optimizer=optimizer.state_dict(),
            epoch=epoch+1, history=history, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        atomic_checkpoint(checkpoint, state)
        if result['val']['top1'] >= max(h['val']['top1'] for h in history):
            atomic_checkpoint(OUT/f'{name}_best.pt', state)
        save_json(OUT/f'{name}_history.json', history)
        print(json.dumps(result), flush=True)
        if CONFIG.get('patience'):
            best_epoch = max(history, key=lambda h: h['val']['top1'])['epoch']
            if epoch+1 >= CONFIG['min_epochs'] and epoch+1-best_epoch >= CONFIG['patience']:
                save_json(OUT/f'{name}_finished.json', dict(reason='early_stopping', epoch=epoch+1))
                return True
    save_json(OUT/f'{name}_finished.json', dict(reason='epoch_limit', epoch=CONFIG['epochs']))
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not json.loads((ROOT/'results/readiness/integrity_summary.json').read_text())['passed']:
        raise ValueError('Integrity gate failed')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    samples = prepare()
    # Train-only majority baseline, evaluated on validation only.
    majority = int(np.bincount([s['label'] for s in samples['train']], minlength=256).argmax())
    save_json(OUT/'majority_baseline.json', dict(beam=majority,
        val_top1=float(np.mean([s['label']==majority for s in samples['val']]))))
    for name in CONFIG['models']:
        if not run_model(name, samples):
            return
    status(status='complete', models=CONFIG['models'], epochs_each=CONFIG['epochs'],
        scope='Training stage complete; no calibration or test evaluation performed.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
