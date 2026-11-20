import sys
from src.rerank_plan import ROOT,OUT,PLAN
from src.unattended_queue import run,completed
if __name__=='__main__':
    if not completed(ROOT/'results/camera_v1/stage95'):raise ValueError('Stage 95 incomplete')
    OUT.mkdir(parents=True,exist_ok=True)
    sys.exit(run(PLAN,'src.rerank_worker','rerank',OUT/'queue.json'))
