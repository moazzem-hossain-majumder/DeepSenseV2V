import sys
from src.factorial_plan import ROOT,OUT,PLAN
from src.unattended_queue import run,completed
if __name__=='__main__':
    if not completed(ROOT/'results/mechanism_v1/stage123'):raise ValueError('Stage 123 incomplete')
    OUT.mkdir(parents=True,exist_ok=True)
    sys.exit(run(PLAN,'src.factorial_worker','factorial',OUT/'queue.json'))
