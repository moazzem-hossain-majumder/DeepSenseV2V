import numpy as np
import pandas as pd
import torch
from PIL import Image,ImageDraw,ImageOps
from src.mechanism_plan import ROOT,OUT,PREVIOUS
from src.publication_data import load,partitions,scale,digest
from src.publication_metrics import per_sample
from src.publication_training import inputs,save_arrays
from src.camera_training import make_loader
from src.camera_data import verify
from src.rerank_training import Reranker
from src.acquire_scenarios import save_json

def choose(m,power,prior):
    train,_,__=partitions(m);sc=m.scenario.to_numpy();labels=m.target_beam.to_numpy();ids=[]
    for s in (36,37,38,39):
        rows=train[sc[train]==s]
        good=per_sample(prior[rows],power[rows],labels[rows],20)['success']
        bad=~per_sample(prior[rows],power[rows],labels[rows],5)['success']
        eligible=rows[good&bad];blocks=m.block_key.to_numpy()[eligible];unique=np.unique(blocks)
        if len(unique)<4:raise ValueError(f'Insufficient independent eligible training blocks for S{s}')
        for block in unique[np.linspace(0,len(unique)-1,4,dtype=int)]:ids.append(int(eligible[np.flatnonzero(blocks==block)[0]]))
    return np.array(ids)

def sample(folder,progress):
    verify();m,g,p,_=load();prior=np.load(PREVIOUS/'stage44/prior_pooled.npy',mmap_mode='r')
    try:
        ids=choose(m,p,prior);train,_,__=partitions(m);g,_,__=scale(g,train)
        context=np.load(PREVIOUS/'stage44/context.npy');lo=context[train].min(0);span=context[train].max(0)-lo;span[span==0]=1
        torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(4)
        model=Reranker(20,True).cuda().eval();captured=[]
        hook=model.head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().cpu().numpy()))
        parts={k:[] for k in ('descriptors','candidates','prior','power','indices')}
        cfg=dict(batch=4,workers=0,camera='both')
        try:
            with torch.inference_mode():
                for j,(rgb,gps,labels,power,sc,indices) in enumerate(make_loader(m,g,p,ids,cfg,(context-lo)/span,prior)):
                    image,x=inputs(rgb,gps,'fusion_ce');out=model(image,x)
                    parts['descriptors'].append(captured.pop());parts['candidates'].append(out['candidates'].cpu().numpy());parts['prior'].append(out['selected'].cpu().numpy())
                    parts['power'].append(power.numpy());parts['indices'].append(indices.numpy())
                    progress(status='extracting_fixed_training_features',model='analysis',batch=j+1,batches=4)
        finally:hook.remove()
        arrays={k:np.concatenate(v) for k,v in parts.items()};save_arrays(folder/'sample.npz',arrays)
        torch.save(model.head.state_dict(),folder/'initial_head.pt')
        save_json(folder/'sample.json',dict(indices=ids.tolist(),sample_sha256=digest(folder/'sample.npz'),head_sha256=digest(folder/'initial_head.pt'),
            scope='16 deliberately selected training-only recoverable failures, one per distinct block, four per scenario. Biased diagnostic sample, not evaluation data.'))
        for s in (36,37,38,39):
            frame=pd.read_csv(ROOT/f'data/raw/scenario{s}/scenario{s}.csv');selected=ids[m.scenario.to_numpy()[ids]==s]
            sheet=Image.new('RGB',(1440,4*284+36),'white');draw=ImageDraw.Draw(sheet)
            draw.text((8,8),f'S{s}: training failures | front / rear (512px previews) | front / rear (96px inputs enlarged)',fill='black')
            for j,i in enumerate(selected):
                row=int(m.x_indices.iloc[i][-1]);y=36+j*284
                draw.text((8,y),f'global {i}, CSV row {row}, block {m.block_key.iloc[i]}',fill='black')
                for v,col in enumerate(('unit1_rgb5','unit1_rgb6')):
                    with Image.open(ROOT/f'data/raw/scenario{s}'/frame[col].iloc[row]) as im:
                        im=im.convert('RGB');sheet.paste(ImageOps.contain(im,(512,256)),(v*512,y+20))
                        small=im.resize((96,96),Image.Resampling.BILINEAR).resize((192,192),Image.Resampling.NEAREST)
                        sheet.paste(small,(1024+v*208,y+20))
            sheet.save(folder/f'visual_S{s}.png')
    finally:prior._mmap.close()
