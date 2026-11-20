import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image
from src.camera_plan import ROOT,CACHE
from src.publication_plan import CACHE as FRONT
from src.publication_data import Clips,digest
from src.acquire_scenarios import save_json

def prepare(progress):
    CACHE.mkdir(parents=True,exist_ok=True)
    parent=json.loads((FRONT/'receipt.json').read_text());receipts={}
    for s in (36,37,38,39):
        csv=ROOT/f'data/raw/scenario{s}/scenario{s}.csv';sha=digest(csv)
        if sha!=parent['csv_sha256'][str(s)]:raise ValueError('Raw CSV changed')
        target=CACHE/f'rear{s}.npy';receipt=CACHE/f'rear{s}.json'
        if target.exists() and receipt.exists():
            info=json.loads(receipt.read_text())
            if info['csv_sha256']!=sha or digest(target)!=info['sha256']:raise ValueError('Rear cache changed')
            receipts[str(s)]=info;continue
        frame=pd.read_csv(csv);temp=CACHE/f'rear{s}.tmp.npy';checkpoint=CACHE/f'rear{s}_checkpoint.json'
        start=0
        if temp.exists() and checkpoint.exists():
            c=json.loads(checkpoint.read_text())
            if c['csv_sha256']!=sha:raise ValueError('Cache resume source changed')
            start=c['rows'];array=np.lib.format.open_memmap(temp,mode='r+')
            if array.shape!=(len(frame),96,96,3):raise ValueError('Cache shape mismatch')
        else:array=np.lib.format.open_memmap(temp,mode='w+',dtype=np.uint8,shape=(len(frame),96,96,3))
        def decode(path):
            with Image.open(ROOT/f'data/raw/scenario{s}'/path) as im:
                im.load()
                if im.mode!='RGB':raise ValueError('Non-RGB rear image')
                return np.asarray(im.resize((96,96),Image.Resampling.BILINEAR))
        with ThreadPoolExecutor(max_workers=6) as pool:
            for first in range(start,len(frame),256):
                stop=min(first+256,len(frame));array[first:stop]=np.stack(list(pool.map(decode,frame.unit1_rgb6.iloc[first:stop])))
                array.flush();save_json(checkpoint,dict(csv_sha256=sha,rows=stop))
                progress(status=f'caching_rear_S{s}',model='analysis',batch=stop,batches=len(frame))
        del array;temp.replace(target)
        info=dict(csv_sha256=sha,sha256=digest(target),shape=[len(frame),96,96,3],column='unit1_rgb6',resize='bilinear',scope='Deterministic image preprocessing only; no label-based selection')
        save_json(receipt,info);receipts[str(s)]=info
    save_json(CACHE/'receipt.json',receipts)

def verify():
    receipts=json.loads((CACHE/'receipt.json').read_text())
    for s,info in receipts.items():
        if digest(CACHE/f'rear{s}.npy')!=info['sha256']:raise ValueError('Rear cache hash mismatch')
    return receipts

class CameraClips(Clips):
    def __init__(self,m,g,p,indices,camera):
        super().__init__(m,g,p,indices,images=camera!='rear');self.camera=camera;self.rear_maps={}
    def __getitem__(self,i):
        front,*rest=super().__getitem__(i)
        if self.camera=='front':return (front,*rest)
        j=self.indices[i];s=int(self.scenarios[j])
        if s not in self.rear_maps:self.rear_maps[s]=np.load(CACHE/f'rear{s}.npy',mmap_mode='r')
        rear=np.array(self.rear_maps[s][self.x[j]],copy=True)
        # Preserve each view's 96x96 pixels and five timestamps; same encoder/parameters.
        image=rear if self.camera=='rear' else np.concatenate((front,rear),axis=2)
        return (image,*rest)
