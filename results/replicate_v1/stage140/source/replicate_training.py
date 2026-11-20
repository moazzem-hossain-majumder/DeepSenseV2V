from torch.nn import functional as F
from src import factorial_training as parent
from src.replicate_plan import ROOT
from src.publication_data import digest

def objective(outputs,labels,power,kind,penalty):
    values=power.float().gather(1,outputs['candidates'])
    relative=(values-values.max(1,keepdim=True).values).clamp(-40,0)
    scores=outputs['selected'].float()
    return F.cross_entropy(scores,values.argmax(1))+.05*(F.softmax(scores,dim=1)*(-relative)).sum(1).mean()+penalty*outputs['residual'].square().mean()

def run(plan,progress):
    original_config,original_objective=parent.configuration,parent.objective
    def configuration(p):
        return {**original_config(p),'residual_penalty':p['penalty'],'replicate_source_sha256':digest(ROOT/'src/replicate_training.py')}
    parent.configuration=configuration
    parent.objective=lambda out,label,power,kind:objective(out,label,power,kind,plan['penalty'])
    try:parent.run(plan,progress)
    finally:parent.configuration,parent.objective=original_config,original_objective
