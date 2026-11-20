import sys
from src.diagnostic_plan import ROOT,OUT,PLAN,PREVIOUS
from src.unattended_queue import run,completed
if __name__=='__main__':
    if not completed(PREVIOUS/'stage61'):raise ValueError('Stage 61 incomplete')
    OUT.mkdir(parents=True,exist_ok=True)
    sys.exit(run(PLAN,'src.diagnostic_worker','diagnostic',OUT/'queue.json'))
