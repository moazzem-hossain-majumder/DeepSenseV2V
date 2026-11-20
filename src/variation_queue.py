import sys
from src.variation_plan import ROOT,OUT,PLAN
from src.unattended_queue import run,completed
if __name__=='__main__':
    if not completed(ROOT/'results/anchor_v1/stage217'):raise ValueError('Stage 217 incomplete')
    OUT.mkdir(parents=True,exist_ok=True)
    sys.exit(run(PLAN,'src.variation_worker','variation',OUT/'queue.json'))
