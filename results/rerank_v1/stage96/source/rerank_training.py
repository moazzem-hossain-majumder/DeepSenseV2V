import numpy as np
import torch
from torch.nn import functional as F
from torchvision.models import resnet18,ResNet18_Weights
from src.rerank_plan import ROOT,PREVIOUS
from src.camera_data import verify
from src.camera_training import make_loader
from src.publication_plan import COMMON
from src.publication_data import partitions,digest
from src.conservative_geometry import key
from src import publication_training as engine

class Reranker(torch.nn.Module):
    def __init__(self,k,vision,encoder=None):
        super().__init__();self.k=k;self.vision=vision
        self.encoder=resnet18(weights=ResNet18_Weights.DEFAULT) if encoder is None else encoder
        self.encoder.fc=torch.nn.Identity()
        for p in self.encoder.parameters():p.requires_grad=False
        self.head=torch.nn.Sequential(torch.nn.Linear(569,128),torch.nn.GELU(),torch.nn.Linear(128,1))
        torch.nn.init.zeros_(self.head[-1].weight);torch.nn.init.zeros_(self.head[-1].bias)
    def forward(self,rgb,gps):
        n=len(gps);prior=gps[:,-1,17:].float();candidates=torch.argsort(prior,dim=1,descending=True,stable=True)[:,:self.k]
        if self.vision:
            self.encoder.eval()
            with torch.no_grad():features=self.encoder(rgb.reshape(-1,*rgb.shape[2:])).reshape(n,5,512).mean(1).float()
            features=F.layer_norm(features,(512,))
        else:features=torch.zeros(n,512,device=gps.device)
        historical=torch.cat((gps[:,:,:9].reshape(n,-1),gps[:,-1,9:17],features),1).float()
        score=prior.gather(1,candidates);angle=candidates.float()*(2*torch.pi/256)
        rank=torch.arange(self.k,device=gps.device).float().expand(n,-1)/self.k
        descriptor=torch.stack((torch.sin(angle),torch.cos(angle),score,rank),2)
        inputs=torch.cat((historical[:,None,:].expand(-1,self.k,-1),descriptor),2)
        raw=self.head(inputs).squeeze(-1).float();residual=2*F.softsign(raw)
        selected=score+residual
        logits=(prior-10000).scatter(1,candidates,selected)
        return dict(logits=logits,candidates=candidates,selected=selected,residual=residual)

def loss(outputs,labels,power,variant):
    candidate_power=power.float().gather(1,outputs['candidates'])
    relative=(candidate_power-candidate_power.max(1,keepdim=True).values).clamp(-40,0)
    soft=F.softmax(relative/3,dim=1);scores=outputs['selected'].float()
    return -(soft*F.log_softmax(scores,dim=1)).sum(1).mean()+.05*(F.softmax(scores,dim=1)*(-relative)).sum(1).mean()+.1*outputs['residual'].square().mean()

def configuration(plan):
    return {**COMMON,'variant':'fusion_ce','batch':16,'epochs':plan['epochs'],'seed':plan['seed'],'holdout':plan['holdout'],
        'camera':'both','k':plan['k'],'vision':plan['vision'],'frozen_visual_features':True,
        'rerank_source_sha256':digest(ROOT/'src/rerank_training.py'),
        'rear_receipt_sha256':digest(ROOT/'data/processed/camera_v1/receipt.json'),
        'geometry_report_sha256':digest(PREVIOUS/'stage44/report.json')}

def run(plan,progress):
    verify();cfg=configuration(plan);context=np.load(PREVIOUS/'stage44/context.npy')
    prior=np.load(PREVIOUS/f'stage44/prior_{key(cfg["holdout"])}.npy',mmap_mode='r')
    original=engine.loader,engine.build_model,engine.config,engine.objective
    def loader(m,g,p,indices,c,train=False):
        ti,_,__=partitions(m,c['holdout']);lo=context[ti].min(0);span=context[ti].max(0)-lo;span[span==0]=1
        return make_loader(m,g,p,indices,c,(context-lo)/span,prior,train)
    engine.loader=loader;engine.build_model=lambda c:Reranker(c['k'],c['vision']).cuda()
    engine.config=lambda _:cfg;engine.objective=loss
    try:engine.train(plan,progress)
    finally:
        engine.loader,engine.build_model,engine.config,engine.objective=original
        prior._mmap.close()
