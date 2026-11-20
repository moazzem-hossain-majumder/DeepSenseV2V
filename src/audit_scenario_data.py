"""Audit official CSV/pickle agreement and acquisition receipts without training."""
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.acquire_scenarios import RAW, REPORTS, ROOT, MODALITIES, save_json


def audit():
    inventory = {'scope': 'GPS and RGB180 inputs; PWR supervision',
                 'audited_utc': datetime.now(timezone.utc).isoformat(), 'scenarios': {}}
    for scenario in (36, 37, 38, 39):
        folder = RAW / f'scenario{scenario}'
        frame = pd.read_csv(folder / f'scenario{scenario}.csv')
        # These pickles must come from the official transfer, never an unknown sender.
        with (folder / f'scenario{scenario}.p').open('rb') as handle:
            data = pickle.load(handle)
        arrays = [np.asarray(data[f'unit1_pwr{i}']) for i in range(1, 5)]
        powers = np.concatenate(arrays, axis=1)
        labels = np.asarray(data['unit1_overall-beam']).reshape(-1)
        actual = powers.argmax(axis=1)
        ids_match = np.array_equal(frame.abs_index.to_numpy(), np.asarray(data['abs_index']).reshape(-1))
        labels_match = np.array_equal(frame['unit1_overall-beam'].to_numpy(), labels)
        if powers.shape != (len(frame), 256) or not ids_match or not labels_match:
            raise ValueError(f'Scenario {scenario} metadata disagreement')
        if not np.isfinite(powers).all() or np.any(powers < 0) or not np.array_equal(actual, labels):
            raise ValueError(f'Scenario {scenario} power reconstruction failed')
        counts = frame.groupby('seq_index').size()
        modalities = {}
        for modality in MODALITIES:
            receipt = REPORTS / f'scenario{scenario}_{modality}.json'
            modalities[modality] = json.loads(receipt.read_text()) if receipt.exists() else {'status':'pending'}
        inventory['scenarios'][str(scenario)] = {
            'csv_rows': len(frame), 'sequence_count': len(counts),
            'sequence_length_min': int(counts.min()), 'sequence_length_max': int(counts.max()),
            'first_timestamp': str(frame.timestamp.iloc[0]), 'last_timestamp': str(frame.timestamp.iloc[-1]),
            'duplicate_absolute_indices': int(frame.abs_index.duplicated().sum()),
            'csv_pickle_indices_match': ids_match, 'csv_pickle_labels_match': labels_match,
            'power_shape': list(powers.shape), 'power_argmax_match_rate': float(np.mean(actual == labels)),
            'modalities': modalities,
            'complete': all(m['status'] == 'verified_archives_deleted' for m in modalities.values()),
            'day_night': 'not yet established',
        }
    save_json(ROOT / 'results' / 'data_inventory.json', inventory)
    print(json.dumps({s:{k:v for k,v in d.items() if k in ('csv_rows','sequence_count','power_argmax_match_rate','complete')}
                      for s,d in inventory['scenarios'].items()}, indent=2))


if __name__ == '__main__':
    audit()
