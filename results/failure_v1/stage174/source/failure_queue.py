import sys
from src.failure_plan import ROOT,OUT,PLAN
from src.unattended_queue import run,completed
if __name__=='__main__':
    if not completed(ROOT/'results/transfer_v1/stage173'):raise ValueError('Stage 173 incomplete')
    OUT.mkdir(parents=True,exist_ok=True)
    sys.exit(run(PLAN,'src.failure_worker','failure',OUT/'queue.json'))
