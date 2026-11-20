import torch
from torch.nn import functional as F
from src.rescheck_training import FeatureReranker
from src import rescheck_training as base
from src.publication_data import digest
from src.robust_plan import ROOT

class RobustReranker(FeatureReranker):
    def __init__(self,k,vision,bounded,dropout=.5,consistency=0.):
        super().__init__(k,vision,bounded);self.image_dropout=dropout;self.consistency=consistency
    def forward(self,rgb,gps):
        if not self.training:return super().forward(rgb,gps)
        mask=(torch.rand((len(rgb),1,1),device=rgb.device)>=self.image_dropout).to(rgb.dtype)
        out=super().forward(rgb*mask,gps)
        if self.consistency:
            out['clean_scores']=super().forward(rgb,gps)['selected']
            out['missing_scores']=super().forward(torch.zeros_like(rgb),gps)['selected']
        return out

def consistency_loss(out):
    a=out['clean_scores'].float();b=out['missing_scores'].float()
    return .5*(F.kl_div(F.log_softmax(a,1),F.softmax(b,1),reduction='batchmean')+F.kl_div(F.log_softmax(b,1),F.softmax(a,1),reduction='batchmean'))

def run(plan,progress):
    original_model,original_loss,original_config=base.FeatureReranker,base.objective,base.parent.configuration
    def config(p):
        return {**original_config(p),'image_dropout':p['image_dropout'],'consistency_weight':p['consistency'],'robust_source_sha256':digest(ROOT/'src/robust_training.py')}
    base.FeatureReranker=lambda k,v,b:RobustReranker(k,v,b,plan['image_dropout'],plan['consistency'])
    base.objective=lambda out,l,p,kind,penalty:original_loss(out,l,p,kind,penalty)+(plan['consistency']*consistency_loss(out) if 'clean_scores' in out else 0.)
    base.parent.configuration=config
    try:base.run(plan,progress)
    finally:base.FeatureReranker,base.objective,base.parent.configuration=original_model,original_loss,original_config
