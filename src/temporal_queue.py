import sys
from src.temporal_plan import ROOT,OUT,PLAN
from src.unattended_queue import run,completed
if __name__=='__main__':
    if not completed(ROOT/'results/motion_v1/stage227'):raise ValueError('Stage 227 incomplete')
    OUT.mkdir(parents=True,exist_ok=True)
    sys.exit(run(PLAN,'src.temporal_worker','temporal',OUT/'queue.json'))
