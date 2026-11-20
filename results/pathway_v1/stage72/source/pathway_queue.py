import sys
from src.pathway_plan import ROOT,OUT,PLAN,PREVIOUS
from src.unattended_queue import run,completed
if __name__=='__main__':
    if not completed(ROOT/'results/diagnostic_v1/stage71'):raise ValueError('Stage 71 incomplete')
    OUT.mkdir(parents=True,exist_ok=True)
    sys.exit(run(PLAN,'src.pathway_worker','pathway',OUT/'queue.json'))

