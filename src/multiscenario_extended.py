"""Stage 2: lower-LR continuation of pilot, with validation-only early stopping."""
import hashlib
import json
import shutil
import torch
from src import multiscenario_pilot as pilot

SOURCE = pilot.ROOT/'results'/'multiscenario_pilot_v1'
pilot.OUT = pilot.ROOT/'results'/'multiscenario_extended_v1'
pilot.CONFIG = {**pilot.CONFIG, 'epochs':12, 'lr':3e-5,
                'patience':4, 'min_epochs':6, 'stage':'lower-lr-continuation-v1'}


def main():
    pilot.OUT.mkdir(parents=True, exist_ok=True)
    pilot.status(status='preparing_continuation')
    if json.loads((SOURCE/'status.json').read_text())['status'] != 'complete':
        raise ValueError('Pilot must finish before continuation')
    protocol = json.loads((SOURCE/'protocol.json').read_text())
    for scenario, expected in protocol['csv_sha256'].items():
        path = pilot.RAW/f'scenario{scenario}'/f'scenario{scenario}.csv'
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Input metadata changed since pilot')
    provenance = {}
    for model in pilot.CONFIG['models']:
        for kind in ('best','latest'):
            target = pilot.OUT/f'{model}_{kind}.pt'
            source = SOURCE/f'{model}_{kind}.pt'
            provenance[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
            if not target.exists():
                state = torch.load(source, map_location='cpu', weights_only=False)
                state['config'] = pilot.CONFIG
                for group in state['optimizer']['param_groups']:
                    group['lr'] = pilot.CONFIG['lr']
                pilot.atomic_checkpoint(target, state)
        history = pilot.OUT/f'{model}_history.json'
        if not history.exists():
            shutil.copy2(SOURCE/history.name, history)
    pilot.save_json(pilot.OUT/'continuation_provenance.json', dict(
        source_sha256=provenance, source_manifest_sha256=protocol['manifest_sha256'],
        note='Epochs 1-3 inherited from pilot. Continue latest optimizer/model at lower LR. Best checkpoint retained.'))
    pilot.main()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        pilot.status(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
