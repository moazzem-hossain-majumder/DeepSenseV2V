import json
import numpy as np
import torch
from torch.nn import functional as F
from src.conservative_plan import ROOT,OUT
from src.conservative_geometry import key
from src.publication_data import partitions,digest
from src.publication_plan import COMMON
from src.publication_training import save_arrays
from src.acquire_scenarios import save_json
from src import publication_training as engine


class BoundedCorrection(torch.nn.Module):
    def __init__(self,backbone=None):
        super().__init__();self.backbone=backbone;self.disable_correction=False
        if backbone is None:
            self.head=torch.nn.Sequential(torch.nn.Linear(53,64),torch.nn.GELU(),torch.nn.Dropout(.25),torch.nn.Linear(64,64),torch.nn.GELU(),torch.nn.Linear(64,256))
            final=self.head[-1]
        else:
            self.context_head=torch.nn.Linear(8,256)
            torch.nn.init.zeros_(self.context_head.weight);torch.nn.init.zeros_(self.context_head.bias)
            final=backbone.cls_head[-1]
        torch.nn.init.zeros_(final.weight);torch.nn.init.zeros_(final.bias)
    def forward(self,rgb,gps):
        core=gps[:,:,:9];context=gps[:,-1,9:17];prior=gps[:,-1,17:]
        if self.backbone is None: raw=self.head(torch.cat((core.reshape(len(core),-1),context),1))
        else: raw=self.backbone(rgb,core)['logits']+self.context_head(context)
        correction=.25*torch.tanh(raw)
        if self.disable_correction: correction=correction*0
        return {'logits':prior+correction,'correction':correction}


def correction_loss(outputs,labels,profiles,variant):
    relative=(profiles-profiles.max(1,keepdim=True).values).clamp(-40,0)
    soft=F.softmax(relative/3,dim=1);logits=outputs['logits'].float()
    return -(soft*F.log_softmax(logits,dim=1)).sum(1).mean()+.05*(F.softmax(logits,dim=1)*(-relative)).sum(1).mean()+.1*outputs['correction'].float().square().mean()


class ContextLoader:
    def __init__(self,base,context,prior):self.base,self.context,self.prior=base,context,prior
    def __len__(self):return len(self.base)
    def __iter__(self):
        for rgb,gps,label,power,scenario,indices in self.base:
            rows=indices.numpy();extra=np.concatenate((self.context[rows],self.prior[rows]),axis=1)
            extra=torch.from_numpy(np.asarray(extra,dtype=np.float32)).unsqueeze(1).expand(-1,gps.shape[1],-1)
            yield rgb,torch.cat((gps,extra),2),label,power,scenario,indices


def use_correction(candidate_score,baseline_score):
    return candidate_score>=baseline_score+.0025


def run(plan,progress):
    folder=ROOT/plan['directory'];folder.mkdir(parents=True,exist_ok=True)
    geometry=json.loads((OUT/'stage44/report.json').read_text());baseline=geometry['configurations'][key(plan['holdout'])]
    prior=np.load(OUT/f"stage44/prior_{key(plan['holdout'])}.npy",mmap_mode='r')
    context=np.load(OUT/'stage44/context.npy')
    original_loader,original_build,original_config,original_loss,original_predict=engine.loader,engine.build_model,engine.config,engine.objective,engine.predict
    cfg={**COMMON,'variant':'gps' if plan['method']=='compact_gps' else 'fusion_ce',
        'seed':plan['seed'],'holdout':plan['holdout'],'method':plan['method'],
        'correction_bound':.25,'correction_l2_weight':.1,'fallback_min_gain':.0025,
        'coordinate_decision':baseline['chosen'],'geometry_report_sha256':digest(OUT/'stage44/report.json'),
        'conservative_source_sha256':digest(ROOT/'src/conservative_training.py')}
    context_parameters={}
    def enhanced_loader(m,g,p,indices,c,train=False):
        ti,_,__=partitions(m,c['holdout']);minimum=context[ti].min(0);span=context[ti].max(0)-minimum;span[span==0]=1
        context_parameters.update(minimum=minimum.tolist(),span=span.tolist())
        return ContextLoader(original_loader(m,g,p,indices,c,train),(context-minimum)/span,prior)
    def predict(model,batches,c,callback,split='val'):
        if split=='unseen_scenario_val':
            saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
            model.disable_correction=not use_correction(saved['history'][-1]['val']['macro_success5'],baseline['selection_val']['macro_success5'])
            del saved
        return original_predict(model,batches,c,callback,split)
    engine.loader=enhanced_loader
    engine.build_model=lambda c:BoundedCorrection(None if plan['method']=='compact_gps' else original_build(c)).cuda()
    engine.config=lambda p:cfg;engine.objective=correction_loss;engine.predict=predict
    try:engine.train(plan,progress)
    finally:
        engine.loader,engine.build_model,engine.config,engine.objective,engine.predict=original_loader,original_build,original_config,original_loss,original_predict
        prior._mmap.close()
    if json.loads((folder/'status.json').read_text())['status']=='paused_by_user':return
    result=json.loads((folder/'result.json').read_text());enabled=use_correction(result['val']['macro_success5'],baseline['selection_val']['macro_success5'])
    selection=dict(correction_enabled=enabled,candidate_val=result['val'],geometry_val=baseline['selection_val'],
        effective_val=result['val'] if enabled else baseline['selection_val'],context_scaler=context_parameters,
        scope='Fallback chosen on selection validation only. Validation performance is selection-biased; unseen-scenario checks are required.')
    source=folder/'val_predictions.npz' if enabled else OUT/f"stage44/baseline_{key(plan['holdout'])}_val.npz"
    with np.load(source) as a:save_arrays(folder/'effective_val.npz',{k:a[k] for k in a.files})
    if plan['holdout']:selection['effective_unseen']=json.loads((folder/'unseen_report.json').read_text())
    save_json(folder/'selection.json',selection)
