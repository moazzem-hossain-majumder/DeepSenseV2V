import sys
from src.penalty_plan import ROOT,OUT,PLAN
from src.unattended_queue import run,completed
if __name__=='__main__':
    if not completed(ROOT/'results/factorial_v1/stage133'):raise ValueError('Stage 133 incomplete')
    OUT.mkdir(parents=True,exist_ok=True)
    sys.exit(run(PLAN,'src.penalty_worker','penalty',OUT/'queue.json'))
