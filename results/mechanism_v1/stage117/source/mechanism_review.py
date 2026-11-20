import ast
import json
import numpy as np
from src.mechanism_plan import ROOT,OUT
from src.publication_data import digest
from src.acquire_scenarios import save_json

def benchmark(folder,progress):
    source=OUT/'reference/competition_benchmark.py.txt';receipt=json.loads((OUT/'reference/receipt.json').read_text())
    if digest(source)!=receipt['sha256']:raise ValueError('Official source snapshot changed')
    tree=ast.parse(source.read_text(encoding='utf-8-sig'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_ori_from_pos_delta')
    # Execute only the inspected pure numerical function, never the file's top-level I/O.
    if function.decorator_list or function.args.defaults:raise ValueError('Unexpected function decoration')
    for node in ast.walk(function):
        if isinstance(node,(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal,ast.With,ast.Try)):raise ValueError('Unexpected function statement')
        if isinstance(node,ast.Attribute) and not (isinstance(node.value,ast.Name) and node.value.id=='np' and node.attr in ('zeros','arctan','pi')):raise ValueError('Unexpected function attribute')
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id not in ('len','range'):raise ValueError('Unexpected function call')
    scope={'np':np,'__builtins__':{'len':len,'range':range}}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'official_orientation','exec'),scope)
    orientation=scope[function.name];lat=np.array([0.,1.,0.,-1.]);lon=np.array([1.,0.,-1.,0.])
    original=orientation(lat,lon);changed=lat.copy();changed[-1]=1.;perturbed=orientation(changed,lon)
    wrapped=np.arange(256)+1;wrapped[wrapped>255]-=255
    result=dict(source=receipt,orientation_first_before=float(original[0]),orientation_first_after_last_input_change=float(perturbed[0]),
        previous_sample_dependency_detected=bool(original[0]!=perturbed[0]),
        shifted_index_unique_count=int(np.unique(wrapped).size),expected_unique_count=256,
        scope='Literal orientation function tested on synthetic arrays; index-wrap arithmetic reproduced independently. Not a dataset score or official challenge reproduction.',
        remaining=['Official challenge split unavailable locally','Physical timestamp conversion unresolved','Matched published learned-baseline reproduction incomplete'])
    save_json(folder/'benchmark_checks.json',result)

def report(folder,progress):
    results={}
    for i in (118,119,120,121):
        r=json.loads((OUT/f'stage{i}/result.json').read_text());results[r['mode']]=r
    control=results['per_example_control']['final']['success5']
    current=results['current_loss']['final']['success5'];bounded=results['bounded_ce']['final']['success5'];unbounded=results['unbounded_ce']['final']['success5']
    if control<.95:action='Investigate optimizer/target/candidate consistency before further model training; positive control did not reach 95%.'
    elif unbounded>=.95 and current<.95:action='A memorization-capable head exists on this sample. Compare objective and correction constraints in a predeclared pilot before generalization claims.'
    elif unbounded<.95:action='The shared scoring head did not memorize this sample sufficiently; investigate representation/capacity and optimization before more large runs.'
    else:action='The current setup can memorize the small sample. Prioritize generalization and visual observability; memorization is not evidence of predictive value.'
    lines=['# Focused mechanism audit','',
        'Sixteen selected training failures, recoverable within geometry top 20. Results below are memorization, not validation/test accuracy.','']
    for mode,r in results.items():lines.append(f"- {mode}: initial success {r['initial']['success5']:.2%}; final {r['final']['success5']:.2%}; best observed {r['max_success5']:.2%}; exact best-candidate top-1 {r['final']['best_candidate_top1']:.2%}.")
    lines+=['',action,'','Visual contact sheets are in stage117. The agent visual review, if completed, is recorded separately in visual_review.md; contact sheet creation alone is not an annotation or transmitter-identification result.',
        'Stage122 records synthetic implementation checks of the snapshotted official source. It does not provide an official challenge score. Timing and benchmark comparability remain unresolved.','',
        'No calibration/test evaluation or automatic full-scale training was performed.']
    (folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    save_json(folder/'decision.json',dict(publication_ready=False,next_action=action,training_memorization=results))
