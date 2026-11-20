"""Full-file Part 1 integrity audit and descriptive multi-scenario EDA.

Run with Python 3.11: python -m src.readiness_audit --workers 8
Every JPEG is fully decoded (not just header checked). Every GPS/PWR text file is
parsed; referenced numeric data is compared to the official pickle/CSV. Completed
modality reports are checkpointed; --resume trusts those earlier audit snapshots.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
from PIL import Image

from src.acquire_scenarios import ROOT, RAW, REPORTS, MODALITIES, save_json, verify_references, digest

OUT = ROOT / 'results' / 'readiness'


def describe(values):
    x = np.asarray(values, dtype=float).ravel()
    good = x[np.isfinite(x)]
    return {'count': len(x), 'nonfinite': int((~np.isfinite(x)).sum()),
            **({k: float(v) for k,v in zip(['min','p01','p05','median','p95','p99','max'],
               np.quantile(good,[0,.01,.05,.5,.95,.99,1]))} if len(good) else {})}


def seconds(series):
    return pd.to_timedelta(series.str.replace('-', ':', regex=False)).dt.total_seconds().to_numpy()


def inspect_file(item):
    path, expected, kind = item
    try:
        raw = path.read_bytes()
        if not raw: raise ValueError('empty file')
        stats = {}
        if kind == 'rgb180':
            with Image.open(io.BytesIO(raw)) as image:
                image.load()  # Full JPEG entropy decode; truncated images remain errors.
                stats['dimensions'] = f'{image.width}x{image.height}'
                if image.mode != 'RGB': raise ValueError(f'Unexpected image mode {image.mode}')
                image.thumbnail((32,32))
                pixels = np.asarray(image).astype(float)
                stats['brightness'] = float(pixels.mean())
                stats['contrast'] = float(pixels.std())
        else:
            values = np.loadtxt(io.BytesIO(raw)).reshape(-1)
            if values.shape != ((64,) if kind == 'pwr' else (2,)):
                raise ValueError(f'Unexpected numeric shape {values.shape}')
            if not np.isfinite(values).all(): raise ValueError('Non-finite numeric values')
            if kind == 'pwr' and np.any(values < 0): raise ValueError('Negative linear power')
            if kind == 'gps' and (abs(values[0]) > 90 or abs(values[1]) > 180):
                raise ValueError('Invalid GPS coordinates')
            if expected is not None and not np.allclose(values, expected, rtol=1e-6, atol=1e-7):
                raise ValueError(f'Raw/metadata mismatch; max absolute error {np.max(np.abs(values-expected))}')
        return {'name': path.as_posix(), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), **stats}
    except Exception as error:
        return {'name': path.as_posix(), 'error': f'{type(error).__name__}: {error}'}


def scan_modality(scenario, modality, frame, data, workers):
    folder = RAW / f'scenario{scenario}'
    reference_stats = verify_references(scenario, modality)
    jobs = []
    referenced = 0
    for col in MODALITIES[modality]:
        if modality == 'pwr': expected = np.asarray(data[col])
        elif modality == 'gps': expected = frame[[col+'_lat',col+'_lon']].to_numpy()
        else: expected = [None] * len(frame)
        lookup = {Path(name).name: val for name,val in zip(frame[col],expected)}
        referenced += len(lookup)
        paths = sorted((folder / col.replace('_','/')).iterdir())
        jobs.extend((p, lookup.get(p.name), modality) for p in paths if p.is_file())
    receipt = json.loads((REPORTS / f'scenario{scenario}_{modality}.json').read_text())
    errors, dimensions, brightness, contrast = [], Counter(), [], []
    error_count = byte_count = 0
    combined = hashlib.sha256()
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index,result in enumerate(pool.map(inspect_file,jobs),1):
            if 'error' in result:
                error_count += 1
                if len(errors) < 30: errors.append(result)
            else:
                byte_count += result['bytes']
                combined.update((str(Path(result['name']).relative_to(folder)).replace('\\','/') + '\0' + result['sha256'] + '\n').encode())
                if 'dimensions' in result:
                    dimensions[result['dimensions']] += 1
                    brightness.append(result['brightness'])
                    contrast.append(result['contrast'])
            if index % 5000 == 0:
                print(f'S{scenario} {modality}: {index:,}/{len(jobs):,} checked; errors={error_count}',flush=True)
    count_match = len(jobs) == receipt['extracted_files']
    return {'scenario':scenario,'modality':modality, **reference_stats,
            'files_scanned':len(jobs),'unreferenced_files':len(jobs)-referenced,
            'extraction_receipt_count_matches':count_match, 'bytes_read':byte_count,
            'error_count':error_count,'error_examples':errors,
            'passed':error_count==0 and count_match,
            'content_manifest_sha256':combined.hexdigest(),
            'digest_note':'Locally measured fingerprint, not a publisher checksum',
            'image_dimensions':dict(dimensions),'brightness_0_255':describe(brightness),
            'thumbnail_pixel_std_0_255':describe(contrast),
            'elapsed_seconds':time.monotonic()-start,
            'audited_utc':datetime.now(timezone.utc).isoformat()}


def numerical_eda(scenario, frame, data):
    powers = np.concatenate([data[f'unit1_pwr{i}'] for i in range(1,5)],axis=1)
    labels = frame['unit1_overall-beam'].to_numpy(dtype=int)
    times = seconds(frame.timestamp)
    dt = np.diff(times)
    same_seq = np.diff(frame.seq_index.to_numpy()) == 0
    gap = np.diff(frame.abs_index.to_numpy())
    p_db = 10*np.log10(np.maximum(powers,1e-12))
    counts = np.bincount(labels,minlength=256)
    probs = counts[counts>0]/len(labels)
    lat1,lon1,lat2,lon2 = [frame[c].to_numpy() for c in ['unit1_gps1_lat','unit1_gps1_lon','unit2_gps1_lat','unit2_gps1_lon']]
    east = np.radians(lon2-lon1)*6371000*np.cos(np.radians(lat1))
    north = np.radians(lat2-lat1)*6371000
    step_m = np.hypot(np.radians(np.diff(lat1))*6371000,
                     np.radians(np.diff(lon1))*6371000*np.cos(np.radians(lat1[:-1])))
    valid_step = same_seq & (dt>0) & (gap==1)
    blocks = int(1 + (gap > 25).sum())
    required = ['abs_index','seq_index','unit1_overall-beam', 'unit1_gps1_lat','unit1_gps1_lon','unit2_gps1_lat','unit2_gps1_lon']
    metadata_pass = bool(powers.shape == (len(frame),256) and np.isfinite(powers).all()
        and (powers >= 0).all() and np.array_equal(powers.argmax(1),labels)
        and np.array_equal(np.asarray(data['abs_index']),frame.abs_index)
        and np.array_equal(np.asarray(data['unit1_overall-beam']),labels)
        and not frame.abs_index.duplicated().any() and (gap>0).all()
        and np.isfinite(frame[required].to_numpy(dtype=float)).all())
    return {'scenario':scenario,'rows':len(frame),'sequence_count':int(frame.seq_index.nunique()),
        'approx_trajectory_blocks_guard25':blocks,'metadata_pass':metadata_pass,
        'missing_cells_by_column':{k:int(v) for k,v in frame.isna().sum().items() if v},
        'duplicate_absolute_ids':int(frame.abs_index.duplicated().sum()),
        'within_sequence_nonconsecutive_abs_ids':int((same_seq & (gap!=1)).sum()),
        'within_sequence_dt_seconds':describe(dt[same_seq]),'within_sequence_nonpositive_dt':int((same_seq & (dt<=0)).sum()),
        'within_sequence_dt_outside_0_08_to_0_12':int((same_seq & ((dt<.08)|(dt>.12))).sum()),
        'receiver_speed_m_s':describe(step_m[valid_step]/dt[valid_step]),
        'tx_rx_separation_m':describe(np.hypot(east,north)),
        'gps_hdop':{f'unit{i}':describe(frame[f'unit{i}_gps1_hdop']) for i in (1,2)},
        'best_power_db':describe(p_db.max(1)),
        'near_optimal_beams':{str(delta):describe((p_db.max(1,keepdims=True)-p_db<=delta).sum(1)) for delta in (1,3,5)},
        'active_beams':int((counts>0).sum()),'label_entropy_bits':float(-(probs*np.log2(probs)).sum()),
        'descriptive_majority_fraction':float(counts.max()/len(frame)),
        'beam_counts':counts.tolist(),'panel_counts':np.bincount(labels//64,minlength=4).tolist(),
        'csv_sha256':digest(RAW/f'scenario{scenario}'/f'scenario{scenario}.csv'),
        'pickle_sha256':digest(RAW/f'scenario{scenario}'/f'scenario{scenario}.p')}


def figures(eda):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for scenario, report in eda.items():
        counts=np.asarray(report['beam_counts'])
        axes[0,0].plot(np.arange(256),counts/counts.sum(),label=f'S{scenario}',alpha=.8)
        frame=pd.read_csv(RAW/f'scenario{scenario}'/f'scenario{scenario}.csv')
        axes[0,1].scatter(frame.unit1_gps1_lon[::10],frame.unit1_gps1_lat[::10],s=2,label=f'S{scenario}')
    axes[0,0].set(xlabel='Beam index (0–255)',ylabel='Fraction of scenario samples',title='Best-beam distribution')
    axes[0,0].legend()
    axes[0,1].set(xlabel='Longitude (degrees)',ylabel='Latitude (degrees)',title='Receiver trajectories (every tenth sample)')
    axes[0,1].legend(markerscale=3)
    names=list(eda)
    axes[1,0].bar(names,[eda[s]['rows'] for s in names])
    axes[1,0].set(xlabel='Scenario',ylabel='CSV rows',title='Measured sample counts')
    for delta in ('1','3','5'):
        axes[1,1].plot(names,[eda[s]['near_optimal_beams'][delta]['median'] for s in names],marker='o',label=f'{delta} dB')
    axes[1,1].set(xlabel='Scenario',ylabel='Median number of beams',title='Near-optimal candidate counts')
    axes[1,1].legend(title='Power tolerance')
    fig.suptitle('DeepSense 36–39: descriptive EDA of official Part 1 data')
    fig.savefig(OUT/'scenario_overview.png',dpi=170)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    eda={}
    for scenario in (36,37,38,39):
        frame=pd.read_csv(RAW/f'scenario{scenario}'/f'scenario{scenario}.csv')
        with (RAW/f'scenario{scenario}'/f'scenario{scenario}.p').open('rb') as f: data=pickle.load(f)
        eda[str(scenario)]=numerical_eda(scenario,frame,data)
    save_json(OUT/'eda.json',eda)
    figures(eda)
    for scenario in (36,37,38,39):
        frame=pd.read_csv(RAW/f'scenario{scenario}'/f'scenario{scenario}.csv')
        with (RAW/f'scenario{scenario}'/f'scenario{scenario}.p').open('rb') as f: data=pickle.load(f)
        for modality in MODALITIES:
            target=OUT/f'scenario{scenario}_{modality}_audit.json'
            if args.resume and target.exists() and json.loads(target.read_text())['passed']:
                print(f'Reusing completed audit {target.name}',flush=True)
                continue
            result=scan_modality(scenario,modality,frame,data,args.workers)
            save_json(target,result)
            print(f'S{scenario} {modality}: passed={result["passed"]}',flush=True)
    reports=[json.loads((OUT/f'scenario{s}_{m}_audit.json').read_text()) for s in (36,37,38,39) for m in MODALITIES]
    passed=all(r['passed'] for r in reports) and all(e['metadata_pass'] for e in eda.values())
    summary={'passed':passed,'files_scanned':sum(r['files_scanned'] for r in reports),
             'file_errors':sum(r['error_count'] for r in reports),
             'rows':sum(e['rows'] for e in eda.values()),
             'audited_utc':datetime.now(timezone.utc).isoformat(),
             'limitation':'Checks local integrity, extraction counts and official metadata consistency; no publisher per-file hashes available.'}
    save_json(OUT/'integrity_summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    if not passed: raise SystemExit('Integrity gate failed; inspect reports before training')


if __name__=='__main__': main()
