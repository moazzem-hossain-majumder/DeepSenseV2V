"""Historical-input coordinate audit; fit offsets on training, conventions on validation."""
import json
import numpy as np
import pandas as pd
from src.conservative_plan import ROOT,OUT
from src.publication_data import load,partitions
from src.publication_metrics import metrics
from src.publication_training import save_arrays
from src.acquire_scenarios import save_json


def key(holdout): return 'pooled' if holdout is None else f'without{holdout}'


def wrap(angle): return (angle+np.pi)%(2*np.pi)-np.pi


def observed_geometry(p1,p2,times):
    """p1/p2: N,5,(latitude,longitude); times: N,5 observed seconds."""
    delta=np.diff(times,axis=1)
    if np.any(delta<=0): raise ValueError('Nonpositive historical time increments')
    # Legacy candidate reproduces the previous analytic construction.
    e1=p1[:,-1]+2.5*(p1[:,-1]-p1[:,-2]);e2=p2[:,-1]+2.5*(p2[:,-1]-p2[:,-2])
    travel=e1-p1[:,-1];rel=e1-e2
    legacy=wrap(-(np.arctan2(rel[:,0],rel[:,1])-np.arctan2(travel[:,0],travel[:,1])-np.pi/4))
    # Local east/north displacements in meters, expressed about last observed receiver.
    origin=p1[:,-1:];coslat=np.cos(np.deg2rad(origin[:,:,0]));radius=6371000.
    def enu(p):
        d=np.deg2rad(p-origin)
        return np.stack((d[:,:,1]*radius*coslat,d[:,:,0]*radius),axis=-1)
    r1,r2=enu(p1),enu(p2)
    centered=times-times.mean(1,keepdims=True)
    denom=(centered**2).sum(1)
    v1=(r1*centered[:,:,None]).sum(1)/denom[:,None]
    v2=(r2*centered[:,:,None]).sum(1)/denom[:,None]
    # Fixed 0.5 s horizon: do not use the future sample's timestamp as an input.
    predicted1=r1[:,-1]+.5*v1;predicted2=r2[:,-1]+.5*v2
    relative=predicted1-predicted2
    heading=np.arctan2(v1[:,1],v1[:,0]);bearing=np.arctan2(relative[:,1],relative[:,0])
    physical=wrap(-(bearing-heading-np.pi/4))
    speed1=np.linalg.norm(v1,axis=1);speed2=np.linalg.norm(v2,axis=1)
    context=np.column_stack((np.sin(heading),np.cos(heading),np.sin(bearing-heading),np.cos(bearing-heading),
        speed1,speed2,(speed1<1).astype(float),delta[:,-1]))
    return legacy,physical,context.astype(np.float32)


def fit_offset(distances,profiles,train):
    nearest=distances[train].argmin(1)
    offsets=np.arange(256)
    power=np.take_along_axis(profiles[train],(nearest[:,None]+offsets)%256,axis=1)
    return int(power.mean(0).argmax())  # Equivalent to minimizing mean dB regret.


def prepare(folder,progress):
    m,_,power,receipt=load();allowed=m.split.isin(['train','val']).to_numpy()
    angles=np.zeros((len(m),2),np.float32);context=np.zeros((len(m),8),np.float32)
    for scenario in (36,37,38,39):
        ids=np.flatnonzero(allowed&(m.scenario.to_numpy()==scenario))
        frame=pd.read_csv(ROOT/f'data/raw/scenario{scenario}/scenario{scenario}.csv');x=np.stack(m.x_indices.iloc[ids].to_numpy())
        positions=[frame[[f'unit{u}_gps1_lat',f'unit{u}_gps1_lon']].to_numpy()[x] for u in (1,2)]
        times=pd.to_timedelta(frame.timestamp.str.replace('-',':',regex=False)).dt.total_seconds().to_numpy()[x]
        legacy,physical,features=observed_geometry(*positions,times)
        angles[ids,0]=legacy;angles[ids,1]=physical;context[ids]=features
        progress(status='auditing_coordinates',model='analysis',batch=scenario-35,batches=4)
    np.save(folder/'context.npy',context)
    diagnostics={str(s):dict(train_val_windows=int((allowed&(m.scenario.to_numpy()==s)).sum()),
        median_heading_change_radians=float(np.median(np.abs(wrap(angles[allowed&(m.scenario.to_numpy()==s),1]-angles[allowed&(m.scenario.to_numpy()==s),0])))),
        slow_heading_fraction=float((context[allowed&(m.scenario.to_numpy()==s),6]>0).mean())) for s in (36,37,38,39)}
    distances=[np.abs(wrap(angles[:,j,None]-np.linspace(-np.pi,np.pi,256))).astype(np.float32) for j in (0,1)]
    report=dict(scope='Two analytic candidates; train-only offset fit, selection-validation convention choice. No mounting-calibration or semantic camera-coverage claim.',
        horizon='Fixed nominal 0.5 seconds; ENU candidate uses observed timestamp spacing only.',data=receipt,scenarios=diagnostics,configurations={})
    for holdout in (None,36,37,38,39):
        train,val,unseen=partitions(m,holdout);candidates=[]
        for name,distance in zip(('legacy_latlon','enu_timeaware'),distances):
            offset=fit_offset(distance,power,train);score=np.roll(-distance[val],offset,axis=1)
            result=metrics(score,power[val],m.target_beam.to_numpy()[val],m.scenario.to_numpy()[val])
            candidates.append(dict(name=name,offset=offset,val=result))
        chosen=max(range(2),key=lambda j:candidates[j]['val']['macro_success5'])
        decision=candidates[chosen];k=key(holdout)
        score=np.roll(-distances[chosen],decision['offset'],axis=1)
        prior=np.maximum(-.5*(score/.35)**2,-20).astype(np.float32);prior[~allowed]=0
        np.save(folder/f'prior_{k}.npy',prior)
        # Float32 rank ties are reflected in the exact prior baseline used by learning.
        selection_metrics=metrics(prior[val],power[val],m.target_beam.to_numpy()[val],m.scenario.to_numpy()[val])
        record=dict(candidates=candidates,chosen=decision['name'],offset=decision['offset'],selection_val=selection_metrics,
            training_scenarios=sorted(m.scenario.iloc[train].unique().tolist()),selection_scenarios=sorted(m.scenario.iloc[val].unique().tolist()))
        save_arrays(folder/f'baseline_{k}_val.npz',dict(logits=prior[val],profiles=power[val],labels=m.target_beam.to_numpy()[val],scenarios=m.scenario.to_numpy()[val],indices=val))
        if holdout:
            record['unseen_val']=metrics(prior[unseen],power[unseen],m.target_beam.to_numpy()[unseen],m.scenario.to_numpy()[unseen])
            save_arrays(folder/f'baseline_{k}_unseen.npz',dict(logits=prior[unseen],profiles=power[unseen],labels=m.target_beam.to_numpy()[unseen],scenarios=m.scenario.to_numpy()[unseen],indices=unseen))
        report['configurations'][k]=record
    save_json(folder/'report.json',report)
