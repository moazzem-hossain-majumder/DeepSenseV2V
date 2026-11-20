"""Development-only geometry-assisted residual model; inputs are historical GPS/RGB."""
import json
import numpy as np
import pandas as pd
import torch
from src.aftercare_plan import ROOT,OUT,PARENT
from src.publication_plan import CACHE,COMMON
from src.publication_data import digest
from src.acquire_scenarios import save_json
from src import publication_training as training


def geometry_prior(manifest):
    offset=json.loads((PARENT/'stage9/report.json').read_text())['geometry_offset']
    prior=np.zeros((len(manifest),256),np.float32)
    allowed=manifest.split.isin(['train','val']).to_numpy()
    for scenario in (36,37,38,39):
        ids=np.flatnonzero(allowed & (manifest.scenario.to_numpy()==scenario))
        x=np.stack(manifest.x_indices.iloc[ids].to_numpy())
        frame=pd.read_csv(ROOT/f'data/raw/scenario{scenario}/scenario{scenario}.csv')
        estimated=[]
        for unit in (1,2):
            pos=frame[[f'unit{unit}_gps1_lat',f'unit{unit}_gps1_lon']].to_numpy()[x]
            estimated.append(pos[:,-1]+2.5*(pos[:,-1]-pos[:,-2]))
        last=frame[['unit1_gps1_lat','unit1_gps1_lon']].to_numpy()[x[:,-1]]
        travel=estimated[0]-last;relative=estimated[0]-estimated[1]
        heading=np.arctan2(travel[:,0],travel[:,1]);bearing=np.arctan2(relative[:,0],relative[:,1])
        angle=(-(bearing-heading-np.pi/4)+np.pi)%(2*np.pi)-np.pi
        distance=np.abs((angle[:,None]-np.linspace(-np.pi,np.pi,256)+np.pi)%(2*np.pi)-np.pi)
        prior[ids]=np.roll(np.maximum(-.5*(distance/.35)**2,-20),offset,axis=1)
    return prior


class GeometryResidual(torch.nn.Module):
    def __init__(self,backbone):
        super().__init__();self.backbone=backbone
        self.geometry_log_gain=torch.nn.Parameter(torch.tensor(0.))
    def forward(self,rgb,gps):
        output=self.backbone(rgb,gps[:,:,:9])
        output['logits']=output['logits']+self.geometry_log_gain.exp().clamp(max=10)*gps[:,-1,9:]
        return output


class PriorLoader:
    def __init__(self,base,prior): self.base,self.prior=base,prior
    def __len__(self): return len(self.base)
    def __iter__(self):
        for rgb,gps,labels,power,scenarios,indices in self.base:
            prior=torch.from_numpy(self.prior[indices.numpy()]).unsqueeze(1).expand(-1,gps.shape[1],-1)
            yield rgb,torch.cat((gps,prior),dim=2),labels,power,scenarios,indices


def run(plan,progress):
    route=json.loads((OUT/'route.json').read_text())
    if route['test_exposed'] or route['branch']!='improve_validation': raise ValueError('Retraining forbidden after test exposure')
    if (PARENT/'stage35/test_lock.json').exists(): raise ValueError('Test exposed after routing; refuse retraining')
    manifest=pd.read_json(CACHE/'manifest.jsonl',lines=True)
    prior=geometry_prior(manifest)
    # Ensure prior replicates the train-fitted analytic baseline on validation.
    with np.load(PARENT/'stage9/geometry_val.npz') as a:
        agreement=float((prior[a['indices']].argmax(1)==a['logits'].argmax(1)).mean())
    if agreement<.999: raise ValueError('Geometry-prior baseline alignment failed')
    cfg={**COMMON,'variant':'power_aware','seed':plan['seed'],'holdout':None,
        'geometry_sigma_radians':.35,'geometry_gain_initial':1.,'geometry_offset_source_sha256':digest(PARENT/'stage9/report.json'),
        'aftercare_source_sha256':digest(ROOT/'src/aftercare_geometry.py'),
        'hypothesis':'Learn visual/GPS residual corrections around a training-only analytic geometry prior.'}
    folder=ROOT/plan['directory'];folder.mkdir(parents=True,exist_ok=True)
    save_json(folder/'geometry_provenance.json',dict(agreement=agreement,
        scope='Observed historical positions only; analytic offset fitted on original training partition. No calibration/test evaluation.'))
    original_build,original_loader,original_config=training.build_model,training.loader,training.config
    training.build_model=lambda config:GeometryResidual(original_build(config)).cuda()
    training.loader=lambda m,g,p,i,c,train=False:PriorLoader(original_loader(m,g,p,i,c,train),prior)
    training.config=lambda p:cfg
    try: training.train(plan,progress)
    finally: training.build_model,training.loader,training.config=original_build,original_loader,original_config
