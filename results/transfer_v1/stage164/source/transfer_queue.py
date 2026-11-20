import sys
from src.transfer_plan import ROOT,OUT,PLAN
from src.unattended_queue import run,completed
if __name__=='__main__':
    if not completed(ROOT/'results/rescheck_v1/stage163'):raise ValueError('Stage 163 incomplete')
    OUT.mkdir(parents=True,exist_ok=True)
    sys.exit(run(PLAN,'src.transfer_worker','transfer',OUT/'queue.json'))
