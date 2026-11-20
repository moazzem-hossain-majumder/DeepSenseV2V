import sys
from src.steady_plan import ROOT,OUT,PLAN
from src.unattended_queue import run,completed
if __name__=='__main__':
    if not completed(ROOT/'results/robust_v1/stage189'):raise ValueError('Stage 189 incomplete')
    OUT.mkdir(parents=True,exist_ok=True)
    sys.exit(run(PLAN,'src.steady_worker','steady',OUT/'queue.json'))
