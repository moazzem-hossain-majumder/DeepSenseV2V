"""Versioned cached inputs; no test-dependent preprocessing."""
import hashlib
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from src.publication_plan import ROOT,OUT,CACHE
from src.acquire_scenarios import save_json
from src.multiscenario_windows import build_windows
from src.gps_features import extract_sequence_gps_features


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def prepare(progress):
    CACHE.mkdir(parents=True,exist_ok=True)
    if not json.loads((ROOT/'results/readiness/integrity_summary.json').read_text())['passed']:
        raise ValueError('Raw integrity gate failed')
    pieces=[];gps=[];profiles=[];hashes={};offset=0
    old=json.loads((ROOT/'results/multiscenario_extended_v1/protocol.json').read_text())
    for s in (36,37,38,39):
        folder=ROOT/f'data/raw/scenario{s}';csv=folder/f'scenario{s}.csv'
        hashes[str(s)]=digest(csv)
        if hashes[str(s)]!=old['csv_sha256'][str(s)]: raise ValueError('CSV changed since audited run')
        frame=pd.read_csv(csv); windows=build_windows(frame,s)
        # Memmap is atomic per scenario; interrupted creation restarts that scenario.
        target=CACHE/f'rgb{s}.npy';receipt=CACHE/f'rgb{s}.json'
        valid=target.exists() and receipt.exists() and json.loads(receipt.read_text()).get('csv_sha256')==hashes[str(s)]
        if not valid:
            temp=CACHE/f'rgb{s}.tmp.npy'
            cache=np.lib.format.open_memmap(temp,mode='w+',dtype=np.uint8,shape=(len(frame),96,96,3))
            def decode(path):
                with Image.open(folder/path) as im:
                    im.load()
                    if im.mode!='RGB': raise ValueError('Non-RGB source image')
                    return np.asarray(im.resize((96,96),Image.Resampling.BILINEAR))
            with ThreadPoolExecutor(max_workers=6) as pool:
                for start in range(0,len(frame),256):
                    cache[start:start+256]=np.stack(list(pool.map(decode,frame.unit1_rgb5.iloc[start:start+256])))
                    progress(status='caching',model='analysis',scenario=s,batch=min(start+256,len(frame)),batches=len(frame))
            cache.flush();del cache;temp.replace(target)
            save_json(receipt,dict(csv_sha256=hashes[str(s)],sha256=digest(target),shape=[len(frame),96,96,3]))
        with (folder/f'scenario{s}.p').open('rb') as f: raw=pickle.load(f)
        power=np.concatenate([raw[f'unit1_pwr{i}'] for i in range(1,5)],axis=1)
        for row in windows.itertuples():
            gps.append(extract_sequence_gps_features(frame,row.x_indices))
            profiles.append(10*np.log10(np.maximum(power[row.y_index],1e-12)))
        windows['global_index']=np.arange(offset,offset+len(windows));offset+=len(windows)
        pieces.append(windows)
    manifest=pd.concat(pieces,ignore_index=True)
    if manifest.groupby('block_key').split.nunique().max()!=1: raise ValueError('Block leakage')
    if not np.isfinite(gps).all() or not np.isfinite(profiles).all(): raise ValueError('Nonfinite raw inputs')
    manifest.to_json(CACHE/'manifest.jsonl',orient='records',lines=True)
    temp=CACHE/'arrays.tmp'
    with temp.open('wb') as f: np.savez_compressed(f,gps=np.array(gps,dtype=np.float32),profiles=np.array(profiles,dtype=np.float32))
    temp.replace(CACHE/'arrays.npz')
    # Cross-split raw row identities must be disjoint, not just target labels.
    for s in (36,37,38,39):
        seen={}
        for split in ('train','val','calib','test'):
            subset=manifest[(manifest.scenario==s)&(manifest.split==split)]
            rows=set(subset.y_index)
            for x in subset.x_indices: rows.update(x)
            for prior in seen.values():
                if rows & prior: raise ValueError('Raw input leakage across splits')
            seen[split]=rows
    save_json(CACHE/'receipt.json',dict(csv_sha256=hashes,manifest_sha256=digest(CACHE/'manifest.jsonl'),
        arrays_sha256=digest(CACHE/'arrays.npz'),count=len(manifest),image_size=96,
        note='Cached preprocessing only; test/calibration labels not used for selection.'))


def load():
    receipt=json.loads((CACHE/'receipt.json').read_text())
    if digest(CACHE/'arrays.npz')!=receipt['arrays_sha256'] or digest(CACHE/'manifest.jsonl')!=receipt['manifest_sha256']:
        raise ValueError('Derived data fingerprint changed')
    for s in (36,37,38,39):
        rgb_receipt=json.loads((CACHE/f'rgb{s}.json').read_text())
        if digest(CACHE/f'rgb{s}.npy')!=rgb_receipt['sha256']:
            raise ValueError('Image cache fingerprint changed')
    manifest=pd.read_json(CACHE/'manifest.jsonl',lines=True)
    with np.load(CACHE/'arrays.npz') as a: gps=a['gps'];profiles=a['profiles']
    return manifest,gps,profiles,receipt


def partitions(manifest,holdout=None):
    train=np.flatnonzero((manifest.split=='train') & (manifest.scenario!=holdout))
    val=np.flatnonzero((manifest.split=='val') & (manifest.scenario!=holdout))
    unseen=np.flatnonzero((manifest.split=='val') & (manifest.scenario==holdout)) if holdout else np.array([],int)
    return train,val,unseen


def scale(gps,train):
    minimum=gps[train].min(axis=(0,1));span=gps[train].max(axis=(0,1))-minimum
    span[span==0]=1
    return (gps-minimum)/span,minimum,span


class Clips(Dataset):
    def __init__(self,manifest,gps,profiles,indices,images=True):
        self.indices=np.asarray(indices);self.gps=gps;self.profiles=profiles;self.images=images
        self.scenarios=manifest.scenario.to_numpy();self.labels=manifest.target_beam.to_numpy()
        self.x=np.stack(manifest.x_indices.to_numpy());self.maps={}
    def __len__(self): return len(self.indices)
    def __getitem__(self,i):
        j=self.indices[i];s=int(self.scenarios[j])
        if self.images:
            if s not in self.maps: self.maps[s]=np.load(CACHE/f'rgb{s}.npy',mmap_mode='r')
            rgb=np.array(self.maps[s][self.x[j]],copy=True)
        else: rgb=np.zeros(1,np.uint8)
        return rgb,self.gps[j],self.labels[j],self.profiles[j],s,j
