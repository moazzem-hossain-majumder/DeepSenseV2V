"""Scenario-qualified, sequence-contained future windows for readiness experiments.

Five inputs t-8,t-6,t-4,t-2,t and target t+5 by sample index. Actual timestamp
horizons are recorded rather than assumed exactly 0.5 seconds. This is not yet a
reproduction of the challenge's published split.
"""
import numpy as np
import pandas as pd


def build_windows(frame, scenario, spacing=2, horizon=5, history=5, guard_abs=25):
    frame=frame.reset_index(drop=True)
    if min(spacing,horizon,history)<1: raise ValueError('Window parameters must be positive')
    ids=frame.abs_index.to_numpy()
    if np.any(np.diff(ids)<=0): raise ValueError('Absolute sample IDs must increase strictly')
    blocks=np.cumsum(np.r_[0,np.diff(ids)>guard_abs])
    n=int(blocks.max())+1
    if n<7: raise ValueError('Too few independent trajectory blocks for four splits')
    cuts=np.cumsum([round(.55*n),round(.15*n),round(.15*n)])
    split_labels=np.array(['train','val','calib','test'])
    split_by_row=split_labels[np.searchsorted(cuts,blocks,side='right')]
    times=pd.to_timedelta(frame.timestamp.str.replace('-',':',regex=False)).dt.total_seconds().to_numpy()
    if not np.isfinite(times).all(): raise ValueError('Timestamps must be finite')
    records=[]
    span=spacing*(history-1)+horizon
    for _,group in frame.groupby('seq_index',sort=False):
        indices=group.index.to_numpy()
        for start in range(len(indices)-span):
            complete=indices[start:start+span+1]
            if np.any(np.diff(complete)!=1) or np.any(np.diff(ids[complete])!=1): continue
            if np.any(np.diff(times[complete])<=0): continue
            if len(set(split_by_row[complete]))!=1: continue
            x=complete[np.arange(history)*spacing]
            y=int(complete[-1])
            records.append({'scenario':scenario,'seq_index':int(frame.seq_index.iloc[y]),
                'block_key':f'{scenario}:{blocks[y]}','split':split_by_row[y],
                'x_indices':x.tolist(),'y_index':y,'target_beam':int(frame['unit1_overall-beam'].iloc[y]),
                'horizon_seconds':float(times[y]-times[x[-1]])})
    return pd.DataFrame(records)
