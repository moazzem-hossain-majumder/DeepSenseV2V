import numpy as np
import torch
from torch.nn import functional as F
from src.factorial_plan import ROOT,PREVIOUS
from src.rerank_training import Reranker
from src.camera_data import verify
from src.camera_training import make_loader
from src.publication_plan import COMMON
from src.publication_data import partitions,digest
from src.conservative_geometry import key
from src import publication_training as engine

class FactorialReranker(Reranker):
    def __init__(self,k,vision,bounded,encoder=None):
        super().__init__(k,vision,encoder);self.bounded=bounded
    def forward(self,rgb,gps):
        n=len(gps);prior=gps[:,-1,17:].float();candidate=torch.argsort(prior,dim=1,descending=True,stable=True)[:,:self.k]
        if self.vision:
            self.encoder.eval()
            with torch.no_grad():features=self.encoder(rgb.reshape(-1,*rgb.shape[2:])).reshape(n,5,512).mean(1).float()
            features=F.layer_norm(features,(512,))
        else:features=torch.zeros(n,512,device=gps.device)
        historical=torch.cat((gps[:,:,:9].reshape(n,-1),gps[:,-1,9:17],features),1).float()
        base=prior.gather(1,candidate);angle=candidate.float()*(2*torch.pi/256)
        rank=torch.arange(self.k,device=gps.device).float().expand(n,-1)/self.k
        descriptors=torch.stack((torch.sin(angle),torch.cos(angle),base,rank),2)
        data=torch.cat((historical[:,None,:].expand(-1,self.k,-1),descriptors),2)
        with torch.autocast(device_type=gps.device.type,enabled=False):
            raw=self.head(data.float()).squeeze(-1)
            residual=2*F.softsign(raw) if self.bounded else raw
            selected=base+residual
        # Row-wise translation preserves ranking while keeping the engine's finite-output guard.
        ranked=selected-selected.detach().amin(1,keepdim=True)
        logits=torch.full_like(prior,-1000.).scatter(1,candidate,ranked)
        return dict(logits=logits,candidates=candidate,selected=selected,residual=residual)

def objective(outputs,labels,power,kind):
    values=power.float().gather(1,outputs['candidates']);relative=(values-values.max(1,keepdim=True).values).clamp(-40,0);scores=outputs['selected'].float()
    if kind=='hard':primary=F.cross_entropy(scores,values.argmax(1))
    elif kind=='soft':primary=-(F.softmax(relative/3,dim=1)*F.log_softmax(scores,dim=1)).sum(1).mean()
    else:raise ValueError('Unknown objective')
    return primary+.05*(F.softmax(scores,dim=1)*(-relative)).sum(1).mean()+.1*outputs['residual'].square().mean()

def configuration(plan):
    return {**COMMON,'variant':'fusion_ce','batch':16,'epochs':plan['epochs'],'seed':plan['seed'],'holdout':plan['holdout'],'camera':'both','k':plan['k'],
        'vision':plan['vision'],'bounded':plan['bounded'],'objective':plan['objective'],'head_precision':'fp32','residual_penalty':.1,'expected_regret_weight':.05,
        'factorial_source_sha256':digest(ROOT/'src/factorial_training.py'),'rear_receipt_sha256':digest(ROOT/'data/processed/camera_v1/receipt.json'),
        'geometry_report_sha256':digest(PREVIOUS/'stage44/report.json')}

def run(plan,progress):
    verify();cfg=configuration(plan);context=np.load(PREVIOUS/'stage44/context.npy');prior=np.load(PREVIOUS/f'stage44/prior_{key(cfg["holdout"])}.npy',mmap_mode='r')
    original=engine.loader,engine.build_model,engine.config,engine.objective
    def loader(m,g,p,ids,c,train=False):
        ti,_,__=partitions(m,c['holdout']);lo=context[ti].min(0);span=context[ti].max(0)-lo;span[span==0]=1
        return make_loader(m,g,p,ids,c,(context-lo)/span,prior,train)
    engine.loader=loader;engine.build_model=lambda c:FactorialReranker(c['k'],c['vision'],c['bounded']).cuda()
    engine.config=lambda _:cfg;engine.objective=lambda out,label,power,variant:objective(out,label,power,cfg['objective'])
    try:engine.train(plan,progress)
    finally:
        engine.loader,engine.build_model,engine.config,engine.objective=original;prior._mmap.close()
