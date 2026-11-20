import sys
from src.widecheck_plan import ROOT,OUT,PLAN
from src.unattended_queue import run,completed
if __name__=='__main__':
    if not completed(ROOT/'results/width_v1/stage249'):raise ValueError('Stage 249 incomplete')
    OUT.mkdir(parents=True,exist_ok=True)
    sys.exit(run(PLAN,'src.widecheck_worker','widecheck',OUT/'queue.json'))
