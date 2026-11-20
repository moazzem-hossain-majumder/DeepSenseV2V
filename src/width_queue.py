import sys
from src.width_plan import ROOT,OUT,PLAN
from src.unattended_queue import run,completed
if __name__=='__main__':
    if not completed(ROOT/'results/temporal_v1/stage239'):raise ValueError('Stage 239 incomplete')
    OUT.mkdir(parents=True,exist_ok=True)
    sys.exit(run(PLAN,'src.width_worker','width',OUT/'queue.json'))
