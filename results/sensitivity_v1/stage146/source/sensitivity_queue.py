import sys
from src.sensitivity_plan import ROOT,OUT,PLAN
from src.unattended_queue import run,completed
if __name__=='__main__':
    if not completed(ROOT/'results/replicate_v1/stage145'):raise ValueError('Stage 145 incomplete')
    OUT.mkdir(parents=True,exist_ok=True)
    sys.exit(run(PLAN,'src.sensitivity_worker','sensitivity',OUT/'queue.json'))
