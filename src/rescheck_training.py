import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from src.rerank_training import Reranker
from src import factorial_training as parent
from src import publication_training as engine
from src.penalty_training import objective
from src.conservative_training import ContextLoader
from src.resolution_data import FeatureClips,verify
from src.resolution_plan import ROOT,CACHE
from src.publication_data import digest
class FeatureReranker(Reranker):
    def __init__(self,k,vision,bounded,encoder=None):
        super().__init__(k,vision,torch.nn.Identity());self.bounded=bounded
    def forward(self,rgb,gps):
        n=len(gps);prior=gps[:,-1,17:].float();candidate=torch.argsort(prior,dim=1,descending=True,stable=True)[:,:self.k]
        features=F.layer_norm(rgb.float().mean(1),(512,)) if self.vision else torch.zeros(n,512,device=gps.device)
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

def run(plan,progress):
    verify(plan['resolution'])
    original=(parent.configuration,parent.objective,parent.make_loader,parent.FactorialReranker,engine.inputs)
    def config(p):
        return {**original[0](p),'resolution':p['resolution'],'brightness':0.,'residual_penalty':0.,'rescheck_source_sha256':digest(ROOT/'src/rescheck_training.py'),'features_receipt_sha256':digest(CACHE/'receipt.json')}
    def loader(m,g,p,ids,c,context,prior,train=False):
        return ContextLoader(DataLoader(FeatureClips(m,g,p,ids,c['resolution']),batch_size=c['batch'],shuffle=train,num_workers=c['workers'],pin_memory=True,persistent_workers=c['workers']>0),context,prior)
    parent.configuration=config;parent.objective=lambda out,l,p,kind:objective(out,l,p,kind,0.)
    parent.make_loader=loader;parent.FactorialReranker=FeatureReranker
    engine.inputs=lambda rgb,gps,variant,brightness=0.:(rgb.cuda(),gps.cuda())
    try:parent.run(plan,progress)
    finally:parent.configuration,parent.objective,parent.make_loader,parent.FactorialReranker,engine.inputs=original
