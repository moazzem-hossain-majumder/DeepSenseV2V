import sys
from src.replicate_plan import ROOT,OUT,PLAN
from src.unattended_queue import run,completed
if __name__=='__main__':
    if not completed(ROOT/'results/penalty_v1/stage139'):raise ValueError('Stage 139 incomplete')
    OUT.mkdir(parents=True,exist_ok=True)
    sys.exit(run(PLAN,'src.replicate_worker','replicate',OUT/'queue.json'))
