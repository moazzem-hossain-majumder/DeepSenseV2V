"""Conditional next cycle, separate from the frozen Stages 8–36 protocol."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/aftercare_v1'
PARENT=ROOT/'results/publication_v1'


def choose_branch(development_passed,test_exposed):
    # Never optimize against a test set that has already been inspected.
    if test_exposed: return 'evidence_review'
    if development_passed: return 'inconsistent_holdout_state'
    return 'improve_validation'


def plan_for(branch):
    if branch not in ('evidence_review','improve_validation'): raise ValueError('Parent evidence needs attention')
    def stage(sid,title,kind,seed=None):
        return dict(id=sid,title=f'Stage {sid} - {title}',kind=kind,seed=seed,branch=branch,
            directory=f'results/aftercare_v1/stage{sid}',models=['power_aware' if kind=='train' else 'analysis'],
            epochs=8 if kind=='train' else 1,unit='epochs' if kind=='train' else 'analysis passes')
    common=[stage(37,'branch and provenance','route'),stage(38,'image-quality EDA','images'),
            stage(39,'failure and baseline audit','diagnostics')]
    if branch=='improve_validation':
        return common+[stage(40,'geometry-assisted P3 seed 42','train',42),
                       stage(41,'geometry-assisted P3 seed 7','train',7),
                       stage(42,'geometry-assisted P3 seed 13','train',13),
                       stage(43,'next-cycle evidence report','improvement_report')]
    return common+[stage(40,'checkpoint reproducibility','reproduce'),
                   stage(41,'locked-test evidence audit','test_audit'),
                   stage(42,'evidence-based manuscript draft','draft'),
                   stage(43,'review package','review_report')]
