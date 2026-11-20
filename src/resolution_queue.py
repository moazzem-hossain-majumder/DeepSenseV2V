import sys
from src.resolution_plan import ROOT,OUT,PLAN
from src.unattended_queue import run,completed
if __name__=='__main__':
    if not completed(ROOT/'results/sensitivity_v1/stage150'):raise ValueError('Stage 150 incomplete')
    OUT.mkdir(parents=True,exist_ok=True)
    sys.exit(run(PLAN,'src.resolution_worker','resolution',OUT/'queue.json'))
