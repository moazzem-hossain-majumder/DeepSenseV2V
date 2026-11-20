import sys
from src.rescheck_plan import ROOT,OUT,PLAN
from src.unattended_queue import run,completed
if __name__=='__main__':
    if not completed(ROOT/'results/resolution_v1/stage154'):raise ValueError('Stage 154 incomplete')
    OUT.mkdir(parents=True,exist_ok=True)
    sys.exit(run(PLAN,'src.rescheck_worker','rescheck',OUT/'queue.json'))
