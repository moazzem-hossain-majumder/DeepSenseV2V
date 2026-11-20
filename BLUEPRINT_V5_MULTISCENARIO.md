# DeepSenseV2V — Blueprint V5: Multi-Scenario (36–39) Paper Plan & Phase Tracker

## Current execution authority — 9 November 2026

Stages 1–7 are complete. The new automatic publication-development queue is defined in `src/publication_plan.py` (Stages 8–36). Read `paper/EXPERIMENT_TRACKER.md` and `paper/BENCHMARK_PROTOCOL_LEDGER.md` for current scope, completed results, fixed hypotheses, claim limitations and quality gates. Historical readiness notes below are dated records, not current job status.

## Execution update — 7 November 2026 (authoritative over older planning notes below)

### Readiness work — 8 November 2026

**Readiness phase finished at 12:29 Bangladesh time.** The full audit scanned
892,474 files with zero detected file errors; all 12 modality audits passed.
All 26 GPU smoke configurations passed, including 224-pixel inputs and batch 16.
The supervisor exited successfully. No full multi-epoch training, validation-based
model selection, calibration evaluation or held-out test evaluation has run yet.
Next: verify comparable published protocols/baselines, then implement the full
multi-scenario training runner and execute controlled train/validation pilots.
The progress notes below describe how this completed phase was carried out.

- Multi-scenario descriptive EDA is saved in `results/readiness/eda.json` and
  `scenario_overview.png`; `results/readiness/REPORT.md` explains scope and limitations.
- Full-file audit is running: parse every GPS/PWR file, compare referenced numeric data
  to official metadata, fully decode every JPEG, count extracted files, and fingerprint
  contents. S36 has passed all three modality audits, including 49,600 JPEGs.
  Do not claim the four-scenario deep scan is complete until `integrity_summary.json` says so.
- Metadata checks pass for all scenarios. Nine timing outliers (2 in S36, 7 in S38)
  are retained; sample IDs remain consecutive within sequences. Use actual horizons.
- Each scenario has one raw file per sensor directory not referenced by the official
  CSV (32 files total). They are preserved and listed in `unreferenced_raw_files.json`.
  This is a source metadata discrepancy, not missing downloaded files. Earlier references
  to rounded website sample counts should not be taken as the established explanation.
- Fixed the legacy loader's silent black-image substitution: failed loads now raise.
- Added scenario-qualified trajectory blocks and sequence-contained five-input windows
  with stride 2 and horizon 5; separate train/validation/calibration/test partitions.
  This does not claim reproduction of the official challenge split.
- Real-data S36 P3 GPU preflight passed: 17,113,408 model parameters, 96-pixel input,
  batch 4, BF16, three optimizer steps with finite gradients and losses
  127.8861 → 126.9642 → 126.4511. These are smoke checks, not convergence evidence.
- A 26-case multi-scenario GPU matrix is queued behind the full integrity gate:
  seven models, 96/160/224 image sizes, batch 8/16, FP32/BF16, selected learning rates,
  profile-loss weights, dropout and seeds. Test calibration/test performance is not used
  for tuning here. Record hardware limits rather than forcing every large configuration.
- `python -m src.supervise_work --job readiness` supervises audit → GPU smoke checks.
  It adopts the active audit, resumes completed modality checkpoints after interruption,
  and runs independently of chat. `DeepSense Readiness Resume.lnk` starts it on Windows sign-in.
  Follow `results/data_acquisition/readiness_supervisor.json` and `results/readiness/pipeline_status.json`.
  Integrity failures need attention; they do not trigger training automatically.


### Outage recovery update

**Acquisition complete — 8 November 2026, 09:59 Bangladesh time.** All four scenarios
passed GPS/PWR/RGB180 extraction CRC checks and CSV-reference verification. All 12
modality receipts report `verified_archives_deleted`; no split ZIPs or partial ZIPs
remain under `data/raw/_zips`. The final inventory marks all scenarios complete,
with 112,196 rows total and 100% power-argmax/label agreement. The supervisor finished
successfully. Training and the research credibility gates remain pending.

The initial worker stopped on DNS failure during an internet outage. Its approximately
13.3 GiB of partial RGB files were retained. The replacement worker retries network
outages indefinitely and probes Dropbox connectivity every 15 seconds after detecting
an interruption. This applies to metadata, signed-link refresh and file transfers.
Active HTTP requests still have their own timeouts before an outage is detected.

`src/supervise_work.py` watches the downloader independently of chat availability and
restarts an unexpectedly exited worker after 15 seconds. Current evidence is in
`results/data_acquisition/download_supervisor.json` and the append-only
`results/data_acquisition/download_worker.log`; older `acquisition.log` is historical.
A per-job process lock prevents duplicate supervisors. The Windows Startup shortcut
`DeepSense Download Resume.lnk` resumes the queue at user sign-in after reboot.
No transfers can run while the PC is powered off or asleep. Storage, integrity and
invalid-path failures remain explicit needs-attention conditions.

Training is not running. The same supervisor supports explicitly configured training
jobs and always adds `--resume`; `train.py` already saves latest/pause checkpoints.
Offline training does not wait for internet. A training crash can lose work since the
last saved checkpoint. Respect `pause.flag`; do not launch training before the research gates.
Ten tests cover archive safety, exact resume, prolonged outages, 15-second checks,
mid-stream interruption, worker restart and checkpoint-resume command construction.

**Objective:** Build a defensible GPS + RGB paper across measured Scenarios 36–39.
No submission deadline; venue and author list remain undecided. Strong results mean
reproducible, fair evidence, including negative findings; acceptance cannot be promised.

### Verified setup and scope

- Workspace: `D:\DeepSenseV2V`. Keep changes local; preserve earlier S36 results as legacy artifacts.
- GPU: NVIDIA RTX 4060, 8 GB; CUDA verified with Python 3.11 and PyTorch 2.5.1+cu121.
- Training interpreter: `C:\Users\MITHIL\AppData\Local\Programs\Python\Python311\python.exe`.
- Python 3.12 also exists but has no PyTorch; neither interpreter is currently on PATH.
- D: initially had 283.71 GiB free. Selected Part 1 downloads total approximately 126.5 GiB.
- Inputs: GPS and RGB180 only. PWR is ground truth, never an extra predictor input.
  Download CSV and `.p`; skip redundant MATLAB `.mat`, RGB90, LiDAR and radar.
- Official source: https://www.deepsense6g.net/scenarios/Scenarios%2030-39/scenarios-36-39
- User supplied working Part 1 transfer links for all four scenarios; access is resolved.
- S36 CSV and pickle: 24,799 rows, 124 sequence IDs. Reconstructed four-array power
  argmax agrees with all pickle labels. Site headline counts are rounded; audit actual rows.

### Current execution tracker

- [x] Inspect existing V5 plan, raw files, GPU, disk and Python environments.
- [x] Resolve all four Part 1 sources and save exact filename/size manifests.
- [x] Add resumable downloads directly to `data/raw`, split-volume extraction without
  an intermediate joined ZIP, CRC checks and CSV reference verification.
- [x] Protect every raw scenario and split archive from accidental Git inclusion.
- [x] Test split-volume reading and refusal of paths outside the extraction directory.
- [x] Finish metadata downloads for all four scenarios.
- [x] Check CSV/pickle index and label agreement plus full 256-power argmax for all scenarios:
  S36 24,799 / S37 30,999 / S38 35,999 / S39 20,399 rows; all agree 100%.
- [x] Extract and CRC-check S36 GPS/PWR; verify 49,598 GPS and 99,196 power references;
  remove both verified archive groups.
- [x] Launch four-worker acquisition for all four scenarios (see live status/log files).
- [x] Finish GPS, PWR and RGB180 for each scenario; verification receipts govern completion.
- [x] Delete each archive group only after CRC and CSV-reference checks pass.
- [ ] Build final data inventory, sequence/time checks and per-scenario power-label audit.
- [ ] Reproduce the official GPS baseline with its exact data/protocol; then evaluate
  that same implementation on our blocked splits separately.
- [ ] Implement and test scenario-aware windows, splits and caches before scaling training.
- [ ] Pilot all baselines and proposed models; freeze evaluation design, then run five seeds.
- [ ] Complete E1–E5, uncertainty estimates, ablations and the paper evidence audit.

Machine-readable evidence lives in `results/data_acquisition/`: `scenarioNN_manifest.json`
lists selected source files; `scenarioNN_MODALITY.json` records verification and archive cleanup.
`verified_archives_deleted` means that modality passed extraction CRC and all CSV-referenced
paths existed and were nonempty before its archives were removed. Missing receipts mean pending.
Raw files and temporary download URLs stay under the ignored `data/raw/` tree.
`run_status.json` records the worker PID and running/failed/complete state; confirm that
the PID is still alive before interpreting a stale “running” record after a reboot.
`results/data_inventory.json` records metadata checks and acquisition status at the last
audit; it is refreshed automatically when the full acquisition run finishes.
The worker logs are `results/data_acquisition/acquisition.log` and
`results/data_acquisition/acquisition_errors.log`. The PC must stay awake and online;
after interruption, rerun the resume command. Do not mark all scenarios complete just
because the worker started. Five tests cover split archives, CRC failure, path safety,
exact download resume and refusal of ignored range requests. The worker uses bounded
32 MiB HTTP ranges and refreshes expired signed links while retaining partial files.

Resume command (safe to rerun after a stopped process; do not run two copies together):

```powershell
& 'C:\Users\MITHIL\AppData\Local\Programs\Python\Python311\python.exe' -u src/supervise_work.py
```

### Publication upgrades and credibility gates

1. **Protocol audit:** Verify challenge horizon, input spacing, split and APL formula against
   the official executable baseline. Published numbers and our blocked-split numbers are
   separate comparisons; matching an input horizon does not make different splits equivalent.
   A fixed ±2 percentage-point tolerance is not sufficient proof of reproduction.
   Verified reference: the official `competition_benchmark.py` computes
   `mean(10 * log10(predicted_power / optimal_power))`, so larger (closer to zero) is better.
   Its beam-offset estimate uses training labels; fit offsets on training only in our
   held-out experiments. Record the exact reference commit before numerical reproduction.
   Source: https://github.com/DeepSense6G/Multi-Modal-V2V-Beam-Prediction-Challenge-2023-Baseline
2. **Fair baselines:** Fix or explain B0, include GPS-only, image-only, fusion,
   classification-only and persistence/geometric baselines. Compute majority baselines
   from training data. Compare identical seeds, splits, horizons and tuning budgets.
   Do not compare a model's Top-5 to a baseline's Top-1 as evidence of superiority.
3. **Feedback cost audit:** Current online controllers read the complete true 256-beam
   profile at each step. This is an oracle-feedback simulation, not yet evidence of
   deployable probing savings. Add periodic full-sweep audits, delayed feedback and total
   beam-measurement cost; report oracle feedback separately. Choose sets before seeing
   the target frame's powers. The global-optimum miss indicator is unobservable from
   candidate-only measurements without extra assumptions or audits.
4. **Guarantee audit:** Clipped thresholds and this PID implementation do not automatically
   inherit the cited methods' theorems. Until assumptions and boundary behavior are proved,
   report empirical risk control. Include threshold saturation and feasibility checks.
5. **Leakage audit:** Fit transforms on training only; keep window inputs and targets within
   one permitted sequence/split. Preserve trajectory grouping and temporal guard intervals.
   More short sequences do not automatically imply more independent trajectories.
6. **Shift design:** E1 per-scenario, E2 pooled, E3 leave-one-scenario-out, E4 inter-city/urban,
   E5 ordered recording replay. Label E5 as concatenated recordings, not one physical drive.
   Day/night is optional only after verified acquisition context; clock time alone is insufficient.
7. **Uncertainty:** Five fixed seeds (42, 7, 13, 21, 101), paired trajectory-level comparisons,
   explicit seed and recording variability, per-scenario results and macro averages.
   Do not tune on held-out scenarios or select the best test seed.
8. **Practical value:** Near-optimal miss risk at 1/3 dB, total probes including audits,
   APL with stated sign/aggregation, Top-1/3/5, profile error, latency and peak GPU memory.
   Ablate profile supervision, RGB, history, resolution, controller and feedback frequency.
9. **Literature:** Earlier citations, numerical claims and categorical novelty statements below
   are research leads, not a completed source audit. Verify papers, dates, code and protocols;
   maintain a source ledger before stating a research gap. Do not claim “first” from this plan.

**Phase order:** Data → credibility/protocol/feedback gates → generalized implementation →
pilot → frozen five-seed E1–E5 experiments → paper. Advance based on correctness and evidence,
not on requiring the proposed model to win. If it loses, investigate and report honestly.

---

## Earlier V5 planning notes (retain for context; qualifications above take precedence)

**Created:** 2026-11-07 · **Supersedes for planning:** `DEEPSENSE_PROJECT_BLUEPRINT_V4.md` (V4 stays the reference for title, novelty wording and method definitions)
**Local root:** `D:\DeepSenseV2V` · **Git:** local branch only, nothing pushed (owner: moazzem-hossain-majumder)
**Hardware:** RTX 4060 8 GB, Python 3.12, ~300 GB free on D:

---

## 0. Goal in one paragraph

Turn the Scenario-36 course project into a publishable paper by (a) fixing the
evaluation so it is comparable with published DeepSense V2V numbers, (b) extending
every experiment to Scenarios 36–39, and (c) testing the online trajectory-adaptive
controller where it should matter most: **under shift between scenarios**
(inter-city → urban, day → night, recording → recording). The core claim does not
change: full 256-beam power-profile prediction + online utility-risk control.

---

## 1. Findings that change the plan (read first)

| # | Finding | Why it matters | Action |
|---|---|---|---|
| F1 | **The official 2023 challenge protocol** uses inputs at t−8, t−6, t−4, t−2, t (0.2 s apart) and predicts **t+5 (0.5 s ahead)**, inside 200-frame sequences (≈23,187 windows for S36). Our repo uses 5 consecutive frames → next frame (0.1 s ahead). | A 0.1 s horizon is easier and not comparable with any published V2V number. | Implement the official windowing as the **primary** protocol; keep 0.1 s as a secondary result. |
| F2 | **Published non-learned GPS baseline (challenge page) beats our learned models on S36**: Top-1/3/5 = 31.2 / 62.6 / 70.4 %, APL −4.81 dB. Ours: B3 Top-5 31 %, P3 Top-5 47 %, B0 Top-5 0.88 %. | A reviewer will put these tables side by side. Our B0 at 0.88 % almost certainly has a bug (array offset, beam ordering, or quadrant mapping). | **Phase 0 gate:** reproduce the challenge baseline exactly; find out why B0/B1 are so far below it. No scaling until fixed. |
| F3 | APL definitions differ: challenge reports APL in negative dB from a linear power ratio; our README reports positive dB. | Inconsistent metric ⇒ incomparable numbers. | Adopt the challenge APL formula, report sign convention explicitly. |
| F4 | **v2vtrack** (S. Hossain, Texas Tech; code v0.2.1, 7 Nov 2026) evaluates **online** tracking on S36–39 as streams, shows the receiver array offset **differs by a few degrees between recordings and drifts within one**, and runs a continual 36→39 stream. | Closest prior work to our "online" claim on the same data. Also explains why GPS models break across scenarios. | Cite and differentiate (theirs: GPS-geometry Kalman offset tracker using sweep feedback; ours: learned RGB+GPS full power profile + conformal risk control on candidate-set size). Consider their offset tracker as a baseline. |
| F5 | Our test set is **3 trajectory blocks** ⇒ P3 Top-5 swings 22–47 % across seeds. | Unpublishable variance. | With 4 scenarios, use more blocks per split, ≥5 seeds, block-bootstrap CIs. |
| F6 | `D:\DeepSense_V2V_Dhaka_24H` is **synthetic** (100 samples per scenario, Dhaka coordinates, Oct 2026 timestamps). | Cannot appear in the paper as measured data. | Leave untouched; never mix into `data/raw`. |
| F7 | The real Scenario-36 raw data is **not** on this PC. | Must re-download 36 too. | Download Part 1 for 36, 37, 38, 39. |

---

## 2. How published work evaluates Scenarios 36–39

| Work | Scenarios | Split / protocol | Metrics | Notes for us |
|---|---|---|---|---|
| DeepSense V2V Challenge 2023 (official baseline) | 36–39, per scenario | Official windows x = t−8…t, y = t+5 | Top-1/3/5, APL | **Main comparison table.** GPS geometric map + constant beam shift. |
| Morais et al., DeepSense-V2V dataset paper (IEEE TVT 2025) | 36–39 | Descriptive | — | Defines inter-city (36, 37) vs urban (38, 39). Cite for data. |
| Osman, Charan, Alkhateeb 2023 (vision V2V) | V2V (36) | Future beam from 360° camera | Top-k | ≈85 % Top-5. Vision baseline reference. |
| Mollah et al. 2024 / 2025 (arXiv 2402.01259, 2504.06173, 2509.11112) | 36–39 (+31–34) | **Random 6:2:2 per scenario** | Top-M, power ratio, APL | Random split leaks adjacent frames; our blocked split is the stricter protocol — say so. |
| v2vtrack (Hossain 2026) | 36–39 | Streams; source 36 → targets 37–39; continual 36→39 | Top-k, APL | Cross-recording shift + online tracking. |
| To read before writing | — | — | — | BeamWorld (trajectory-aware world model, V2V vision, Jan 2026); Target-in-the-Loop GPS+LSTM V2V (May 2026); Transition-aware conformal UAV beam tracking (Jan 2026); BeamGuard risk-aware V2I (Aug 2026); SCAN-BEST. |

**Takeaway:** nobody has reported a *blocked, leakage-safe* multi-scenario study with
*calibrated candidate sets* and an *online controller under cross-scenario shift* on
36–39. That is the gap to own, provided F2 is fixed.

---

## 3. Experiment design (derived from §2)

| ID | Experiment | Train → Test | Answers |
|---|---|---|---|
| **E1** | Per-scenario | each of 36/37/38/39, blocked split | Comparable to challenge table; RQ1, RQ3 per scenario |
| **E2** | Pooled | 36–39 together, blocked split, scenario-stratified | Does more data stabilise P3 vs P1? |
| **E3** | Leave-one-scenario-out | 3 scenarios → held-out 1 (×4) | RQ5 shift; where static CRC should fail and ACI/PID should hold |
| **E4** | Regime shift | inter-city (36, 37) → urban (38, 39), and reverse; day → night (confirm which scenarios are night) | Named, interpretable shifts for the paper |
| **E5** | Continual stream | 36 → 37 → 38 → 39 online | Online controller's rolling miss rate over time (headline figure) |

Fixed across all: official windowing (F1) as primary; ≥5 seeds; block-bootstrap 95 % CI;
metrics = Top-1/3/5, APL (challenge formula), profile MAE, Δ = 1/3 dB miss rate, average
probes, search reduction, rolling miss rate. Baselines: challenge geometric map, B1
GPS-only, B2 RGB-only, B3 fusion, P1, P3, static CRC, ACI, PID, (optional) v2vtrack offset tracker.

---

## 4. Phases and tracker

Status: ☐ not started · ◐ in progress · ☑ done

### Phase A — Data (this week)
- ◐ A1 Folder `D:\DeepSenseV2V` + clone ☑
- ☐ A2 Sign in to deepsense6g.net (user does this; Claude cannot type passwords)
- ☐ A3 Download **Part 1 only** (GPS, PWR, RGB180, CSV + `.p`) for 36, 37, 38, 39 → `data/raw/_zips/scenarioNN/`
- ☐ A4 Install 7-Zip; extract each scenario to `data/raw/scenarioNN/`; move CSV into scenario folder
- ☐ A5 Verify extraction (file counts vs CSV rows, sample counts 24,800 / 31,000 / 36,000 / 20,400), **then** delete zips
- ☐ A6 Record per-scenario disk size, day/night, sequence count in `results/data_inventory.json`

### Phase 0 — Credibility gate (before any new training)
- ☐ 0.1 Reproduce the challenge baseline on S36 within ±2 pp of published Top-1/3/5 and APL
- ☐ 0.2 Find the root cause of B0 = 0.88 % Top-5 (beam ordering of the 4×64 arrays, array offset, rgb5/rgb6 front mapping)
- ☐ 0.3 Implement official windowing (t−8…t → t+5) alongside current 0.1 s windowing
- ☐ 0.4 Switch APL to the challenge formula; regenerate S36 tables
- ☐ 0.5 Re-check: does P3 now beat the challenge baseline on S36? If not, rethink before scaling

### Phase B — Code generalisation
- ☐ B1 `src/paths.py`, `src/dataset.py`: remove `scenario36` hard-coding → `scenarios: [36,37,38,39]` in `config.yaml`
- ☐ B2 Per-scenario RGB cache files (96×96 uint8) so 4 scenarios fit in RAM / on disk
- ☐ B3 `partitioning.py`: scenario-aware blocked splits; support E1–E5 split modes
- ☐ B4 Verify 256-vector reconstruction and argmax = label **per scenario**
- ☐ B5 Experiment runner: `python run_experiment.py --exp E3 --holdout 38 --seed 7`
- ☐ B6 Unit tests for windowing, splits (no block in two splits), APL

### Phase C — Experiments
- ☐ C1 E1 per-scenario (all models, 5 seeds)
- ☐ C2 E2 pooled
- ☐ C3 E3 leave-one-scenario-out
- ☐ C4 E4 regime shift
- ☐ C5 E5 continual stream
- ☐ C6 Significance tests + CIs; ablations (profile head, RGB, controller η)

### Phase D — Paper
- ☐ D1 Mandatory novelty re-check (V4 rule: 2–4 weeks before submission), incl. §2 "to read" list
- ☐ D2 Choose venue (decide after Phase C: conference ≈6 pp vs journal)
- ☐ D3 Rewrite `paper/draft.md` for 36–39; new figures; data-availability & licence statement
- ☐ D4 Internal review pass (independent reviewer agent reads paper vs results JSON)

---

## 5. Data layout and download procedure

```
D:\DeepSenseV2V\data\raw\
  _zips\scenario36\ ...           # temporary, deleted after verified extraction
  scenario36\  scenario36.csv  scenario36.p  unit1\{gps1,pwr1..4,rgb5,rgb6}  unit2\gps1
  scenario37\  ...
  scenario38\  ...   (Part 1 may be split across several links)
  scenario39\  ...
```
- Source: deepsense6g.net → Scenarios 36–39 → "Download Dataset" (sign-in required).
- Part 1 = GPS, PWR, RGB180 + CSV & pre-loaded dictionaries. Part 2+ = RGB90, LiDAR, radar (**not downloaded**).
- Archives are split zips → extract with 7-Zip ("Extract here" on the `.001`).
- Delete zips only after A5 verification passes.
- `data/raw` is git-ignored; never commit raw data (licence: cite DeepSense 6G, Alkhateeb et al., IEEE Commun. Mag. 2023, and Morais et al., IEEE TVT 2025).

---

## Execution tracker (2026-11-08)

See [paper/EXPERIMENT_TRACKER.md](paper/EXPERIMENT_TRACKER.md) for the current pilot queue, completed integrity/EDA work, evaluation gates, recovery behavior, and provisional time budget. Live execution is recorded in `results/multiscenario_pilot_v1/status.json`. Pilot results must not be presented as converged or official benchmark results.

## 6. Open questions / decisions log

| Date | Question | Decision |
|---|---|---|
| 2026-11-07 | Which parts to download? | Part 1 only |
| 2026-11-07 | Git handling? | Local only |
| 2026-11-07 | Venue? | Not decided — revisit after Phase C |
| 2026-11-07 | Eval design? | Derived from literature (§2–3) |
| open | Is the moazzem-hossain-majumder repo co-owned? Author list / order? | — |
| open | Which scenarios are night-time? (37 reported as night in one 2026 paper; verify from CSV timestamps) | — |
| open | Keep RGB at 96×96, or test 160/224 now that data is 4× larger? | — |
| open | Include v2vtrack offset tracker as a baseline? | — |
