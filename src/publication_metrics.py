import math
import numpy as np
import torch
from torch.nn import functional as F


def objective(outputs,labels,profiles,variant):
    logits=outputs['logits'].float()
    if variant=='power_aware':
        relative=(profiles-profiles.max(1,keepdim=True).values).clamp(-40,0)
        soft=F.softmax(relative/3.,dim=1)
        loss=-(soft*F.log_softmax(logits,dim=1)).sum(1).mean()
        loss=loss+.05*(F.softmax(logits,dim=1)*(-relative)).sum(1).mean()
        return loss+.1*F.smooth_l1_loss(outputs['pred_profile'].float(),relative)
    loss=F.cross_entropy(logits,labels)
    if variant=='profile_original':
        from src.models import pairwise_ranking_loss
        loss=loss+.1*F.mse_loss(outputs['pred_profile'].float(),profiles)+.05*pairwise_ranking_loss(outputs['pred_profile'],profiles)
    return loss


def per_sample(logits,profiles,labels,k=5):
    order=np.argsort(-logits,axis=1,kind='stable')
    loss=profiles.max(1)-np.take_along_axis(profiles,order[:,:k],axis=1).max(1)
    return dict(hit=(order[:,:k]==labels[:,None]).any(1),success=loss<=3,regret=loss)


def metrics(logits,profiles,labels,scenarios):
    result={'n':len(labels)}
    for k in (1,3,5,10,15,20,32,64):
        m=per_sample(logits,profiles,labels,k)
        result[f'top{k}']=float(m['hit'].mean())
        result[f'within_3db_at_{k}']=float(m['success'].mean())
        result[f'power_loss_db_at_{k}']=float(m['regret'].mean())
        if k==5:
            result['macro_success5']=float(np.mean([m['success'][scenarios==s].mean() for s in np.unique(scenarios)]))
    result['power_loss_db']=result['power_loss_db_at_1']
    return result


def ordered_probabilities(logits):
    x=logits-logits.max(1,keepdims=True);p=np.exp(x);p/=p.sum(1,keepdims=True)
    order=np.argsort(-p,axis=1,kind='stable');ordered=np.take_along_axis(p,order,axis=1)
    before=np.cumsum(ordered,axis=1)-ordered
    before[:,0]=0
    return order,before


def calibrate(logits,profiles,coverage=.9):
    order,before=ordered_probabilities(logits)
    good=profiles>=profiles.max(1,keepdims=True)-3
    rank=np.take_along_axis(good,order,axis=1).argmax(1)
    score=before[np.arange(len(rank)),rank]
    index=math.ceil((len(score)+1)*coverage)-1
    q=1. if index>=len(score) else float(np.sort(score)[index])
    k=256 if index>=len(score) else int(np.sort(rank+1)[index])
    return dict(q=q,fixed_k=k,target_coverage=coverage,n=len(score),
                scope='Empirical calibration under dependence; no distribution-free guarantee claimed.')


def candidate_outcomes(logits,profiles,parameters,adaptive=True):
    order,before=ordered_probabilities(logits)
    count=(before<=parameters['q']).sum(1) if adaptive else np.full(len(logits),parameters['fixed_k'])
    count=np.clip(count,1,logits.shape[1])
    selected=np.arange(logits.shape[1])[None,:]<count[:,None]
    power=np.take_along_axis(profiles,order,axis=1)
    loss=profiles.max(1)-np.where(selected,power,-np.inf).max(1)
    return count,loss


def block_intervals(values,blocks,scenarios,replicates=1000,seed=1729):
    """Scenario-stratified trajectory bootstrap of pooled means, not IID errors."""
    values=np.asarray(values);blocks=np.asarray(blocks);scenarios=np.asarray(scenarios)
    if values.ndim==1: values=values[:,None]
    groups=[np.unique(blocks[scenarios==s]) for s in np.unique(scenarios)]
    sums={b:values[blocks==b].sum(0) for b in np.unique(blocks)}
    counts={b:int((blocks==b).sum()) for b in sums}
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(replicates):
        sampled=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        draws.append(sum(sums[b] for b in sampled)/sum(counts[b] for b in sampled))
    return np.quantile(draws,[.025,.975],axis=0).T.tolist()
