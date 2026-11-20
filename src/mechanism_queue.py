import sys
from src.mechanism_plan import ROOT,OUT,PLAN
from src.unattended_queue import run,completed
if __name__=='__main__':
    if not completed(ROOT/'results/rerank_v1/stage116'):raise ValueError('Stage 116 incomplete')
    OUT.mkdir(parents=True,exist_ok=True)
    sys.exit(run(PLAN,'src.mechanism_worker','mechanism',OUT/'queue.json'))
