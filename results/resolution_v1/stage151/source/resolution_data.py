import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.models import resnet18,ResNet18_Weights
from src.resolution_plan import ROOT,CACHE
from src.publication_plan import CACHE as SOURCE
from src.publication_data import digest,Clips
from src.acquire_scenarios import save_json

def prepare(progress):
    CACHE.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    m=pd.read_json(SOURCE/'manifest.jsonl',lines=True);parent=json.loads((SOURCE/'receipt.json').read_text())
    if digest(SOURCE/'manifest.jsonl')!=parent['manifest_sha256']:raise ValueError('Manifest changed')
    model=resnet18(weights=ResNet18_Weights.DEFAULT).cuda().eval();model.fc=torch.nn.Identity()
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    records={}
    for scenario in (36,37,38,39):
        csv=ROOT/f'data/raw/scenario{scenario}/scenario{scenario}.csv'
        if digest(csv)!=parent['csv_sha256'][str(scenario)]:raise ValueError('CSV changed')
        rows=pd.read_csv(csv);subset=m[(m.scenario==scenario)&m.split.isin(['train','val'])]
        ids=np.unique(np.concatenate(subset.x_indices.to_numpy())).astype(int)
        identity=dict(manifest_sha256=parent['manifest_sha256'],csv_sha256=digest(csv),ids=ids.tolist(),encoder='ResNet18 IMAGENET1K_V1',precision='bf16',resize='PIL bilinear square per view, side-by-side front/rear',brightness=0)
        for size in (96,192):
            name=f's{scenario}_{size}';target=CACHE/f'{name}.npy';meta=CACHE/f'{name}.json';checkpoint=CACHE/f'{name}_progress.json'
            if meta.exists():
                info=json.loads(meta.read_text())
                if info['identity']!=identity or digest(target)!=info['sha256']:raise ValueError('Feature cache changed')
                records[name]=info;continue
            tmp=CACHE/f'{name}.tmp.npy';start=0
            if checkpoint.exists() and tmp.exists():
                c=json.loads(checkpoint.read_text())
                if c['identity']!=identity:raise ValueError('Feature resume mismatch')
                start=c['done'];array=np.lib.format.open_memmap(tmp,mode='r+')
            else:
                array=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(len(rows),512));array[:]=np.nan
            def decode(i):
                views=[]
                for col in ('unit1_rgb5','unit1_rgb6'):
                    with Image.open(csv.parent/rows.iloc[i][col]) as im:views.append(np.asarray(im.convert('RGB').resize((size,size),Image.Resampling.BILINEAR)))
                return np.concatenate(views,axis=1)
            with ThreadPoolExecutor(max_workers=6) as pool,torch.inference_mode():
                for first in range(start,len(ids),64):
                    batch_ids=ids[first:first+64];images=np.stack(list(pool.map(decode,batch_ids)))
                    x=torch.from_numpy(images).cuda().permute(0,3,1,2).float()/255
                    with torch.autocast('cuda',dtype=torch.bfloat16):features=model((x-mean)/std).float()
                    if not torch.isfinite(features).all():raise ValueError('Nonfinite image features')
                    array[batch_ids]=features.cpu().numpy();array.flush();save_json(checkpoint,dict(identity=identity,done=first+len(batch_ids)))
                    progress(status=f'features_S{scenario}_{size}px',model='analysis',batch=first+len(batch_ids),batches=len(ids))
            del array;tmp.replace(target)
            info=dict(identity=identity,sha256=digest(target),size=size,rows=len(ids),shape=[len(rows),512]);save_json(meta,info);records[name]=info
    save_json(CACHE/'receipt.json',records)

def verify(size):
    receipt=json.loads((CACHE/'receipt.json').read_text())
    for s in (36,37,38,39):
        name=f's{s}_{size}'
        if digest(CACHE/f'{name}.npy')!=receipt[name]['sha256']:raise ValueError('Feature cache hash mismatch')

class FeatureClips(Clips):
    def __init__(self,m,g,p,indices,size):
        super().__init__(m,g,p,indices,False);self.size=size;self.features={}
        if not m.split.iloc[indices].isin(['train','val']).all():raise ValueError('Only train/validation feature access permitted')
    def __getitem__(self,i):
        _,*rest=super().__getitem__(i);j=self.indices[i];s=int(self.scenarios[j])
        if s not in self.features:self.features[s]=np.load(CACHE/f's{s}_{self.size}.npy',mmap_mode='r')
        value=np.array(self.features[s][self.x[j]],copy=True)
        if not np.isfinite(value).all():raise ValueError('Missing cached features')
        return value,*rest
