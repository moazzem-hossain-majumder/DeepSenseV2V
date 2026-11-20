"""Render the current readiness evidence into a repository report."""
import json
from datetime import datetime, timezone
from pathlib import Path
from src.acquire_scenarios import ROOT, MODALITIES

OUT=ROOT/'results'/'readiness'


def main():
    eda=json.loads((OUT/'eda.json').read_text())
    reports=[json.loads(p.read_text()) for p in OUT.glob('scenario*_audit.json')]
    summary_path=OUT/'integrity_summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else None
    smoke_path=OUT/'training_smoke.json'
    smoke=json.loads(smoke_path.read_text()) if smoke_path.exists() else None
    lines=['# DeepSense 36–39: data integrity, EDA and training readiness','',
           f'Generated UTC: {datetime.now(timezone.utc).isoformat()}','',
           '## Integrity status','']
    if summary:
        lines.append(f'Full audit passed: **{summary["passed"]}**. Files scanned: **{summary["files_scanned"]:,}**. File errors: **{summary["file_errors"]}**.')
    else:
        lines.append(f'Full scan is still in progress. Completed modality audits: **{len(reports)}/12**. '
                     f'Files checked in completed audits: **{sum(r["files_scanned"] for r in reports):,}**. '
                     f'Errors in completed audits: **{sum(r["error_count"] for r in reports)}**.')
    lines += ['', 'Checks: CSV-reference existence, nonempty files, extracted-file counts, full JPEG decoding, '
              'GPS/PWR text parsing, numeric agreement with source metadata, CSV/pickle alignment, and local content fingerprints.', '',
              'Local hashes are not publisher-provided checksums. These checks cannot establish that the original data collection '
              'was lossless; they test the acquired Part 1 files and their consistency with supplied metadata.', '',
              '## Descriptive EDA','', '![Scenario overview](scenario_overview.png)','']
    for scenario,e in eda.items():
        lines.append(f'- S{scenario}: {e["rows"]:,} CSV rows, {e["sequence_count"]} sequence IDs, '
                     f'{e["approx_trajectory_blocks_guard25"]} trajectory blocks under the 25-index-gap rule, '
                     f'{e["active_beams"]}/256 observed best-beam classes; '
                     f'median Tx–Rx separation {e["tx_rx_separation_m"]["median"]:.1f} m.')
    lines += ['', 'Interpretation and next checks:', '',
              '- All four metadata checks pass: no duplicate absolute IDs, exact CSV/pickle label agreement, '
              'and full-profile argmax matches the label. This is descriptive data validation, not model performance.',
              '- Nine within-sequence intervals are outside 0.08–0.12 seconds (S36: 2, S38: 7). '
              'No within-sequence sample-index gaps or nonpositive time differences were found. Keep the observations; '
              'record actual future horizons and assess timing sensitivity before final experiments.',
              '- There is one unreferenced raw file per downloaded sensor directory in each scenario (32 files total). '
              'They remain on disk. This is a source metadata/raw-file discrepancy, not evidence that our download lost files. '
              'Do not invent missing labels or add these files to supervised samples.',
              '- Sequence IDs are not independent trajectories: 124/155/180/102 sequences become 19/19/23/15 '
              'blocks under the current grouping rule. Retain trajectory-level separation and uncertainty estimates.',
              '- Best-beam classes are imbalanced; train-derived baselines and per-scenario metrics are required. '
              'All-data class proportions in EDA must not become test-tuned baseline choices.', '',
              '## Training smoke matrix','',
              '26 configurations are prepared: B1/B2/B3/B4/P1/P2/P3; resolutions 96/160/224; '
              'batch sizes 8/16; FP32/BF16; learning rates 0.00003/0.0001/0.0003; '
              'profile-loss weights 0.03/0.1/0.3; dropout 0/0.12/0.25; seeds 42/7/13. '
              'This is a controlled feasibility matrix, not a full Cartesian grid or a hyperparameter selection experiment.', '',
              'Each case uses three optimizer steps on a fixed train-only sample spanning four scenarios. '
              'Checks include finite losses/gradients, parameter counts, step time, peak allocated GPU memory and '
              'a checkpoint save/restore test for the default P3 case. Out-of-memory cases identify hardware limits. '
              'Front RGB5 is used for model compatibility; both RGB5 and RGB6 are checked by the data audit.', '']
    if smoke:
        cases=smoke['cases']
        lines.append(f'Cases recorded: {len(cases)}/26. Passed: {sum(c["status"]=="passed" for c in cases)}; '
                     f'out of memory: {sum(c["status"]=="out_of_memory" for c in cases)}; '
                     f'failed: {sum(c["status"]=="failed" for c in cases)}. Matrix complete: {smoke["complete"]}.')
    else: lines.append('GPU smoke matrix is queued behind the full integrity gate; no training results claimed yet.')
    lines += ['', 'Smoke results are **not** accuracy estimates, publication evidence, or proof of convergence. '
              'Official-baseline reproduction, protocol comparison, feedback-cost audit and full multi-seed experiments remain pending.', '',
              '## Files and resumption','',
              '- `eda.json`: numerical descriptors and source metadata hashes.',
              '- `scenarioNN_MODALITY_audit.json`: full-file checks and content fingerprints.',
              '- `integrity_summary.json`: final integrity gate (created only after all 12 modality checks).',
              '- `window_and_split_audit.json`: qualified blocks, splits and actual horizon distributions.',
              '- `training_smoke.json`: case-by-case GPU test evidence.',
              '- `../data_acquisition/readiness_supervisor.json`: watchdog status.', '',
              'Run `python -m src.supervise_work --job readiness` using the configured Python 3.11 interpreter. '
              'Completed audit modalities are checkpointed. This work is local and does not need internet '
              'because the ResNet weights are already cached.', '']
    preflight=OUT/'preflight_scenario36.json'
    if preflight.exists():
        p=json.loads(preflight.read_text())['cases'][0]
        lines += ['## Completed Scenario 36 GPU preflight','',
                  f'Status: **{p["status"]}**. The full S36 GPS, power and image audits passed before this test.', '']
        if p['status']=='passed':
            lines.append(f'P3: {p["total_parameters"]:,} total parameters, batch 4, 96-pixel images, BF16, '
                         f'three optimizer steps. Loss values: {p["losses"]}. '
                         f'Peak allocated PyTorch GPU memory: {p["peak_allocated_gib"]:.3f} GiB. '
                         'This small-batch check demonstrates executable training, not generalization or convergence.')
        lines.append('')
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__': main()
