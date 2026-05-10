# ES RL Agent — Project Plan

This document tracks milestone progress, completion criteria, and next steps for the ES RL Agent evaluation and review workflow.

---

## Overall Status

### Completed
- Milestone 1 — Stabilize evaluation artifacts
- Milestone 2 — Build deterministic verifier
- Milestone 3 — Build LLM input packet

### In Progress
- Milestone 5 — Build orchestration
- Milestone 6 — Real-run validation

### Not Started
- Milestone 4 — Build LLM review layer

---

## Milestone 1 — Stabilize evaluation artifacts

### Status
**DONE**

### Tasks
- [x] Ensure `evaluate.py` always writes `eval_summary.json`
- [x] Ensure `evaluate.py` always writes `steps.csv`
- [x] Ensure `evaluate.py` always writes `trades.csv`
- [x] Ensure `evaluate.py` always writes `trade_breakdown.csv`
- [x] Ensure `eval_summary.json` contains required eligibility diagnostics
- [x] Ensure `steps.csv` contains required columns

### Completed Notes
- `evaluate.py` now consistently writes all required report artifacts.
- `steps.csv` is created with required columns even when step logs are empty.
- `trades.csv` is created with stable columns even when no completed trades exist.
- `trade_breakdown.csv` is created with stable headers even when no trades exist.
- `eval_summary.json` includes required `eligibility_diagnostics` fields.

### Done Criteria
- [x] A real run folder consistently contains all required files
- [x] Required diagnostics/columns are present without manual patching

---

## Milestone 2 — Build deterministic verifier

### Status
**DONE**

### Tasks
- [x] Create `scripts/verify_eval_output.py`
- [x] Validate required files exist
- [x] Validate JSON/CSV parsing
- [x] Validate required step columns
- [x] Compute behavior checks
- [x] Implement diagnosis classification
- [x] Implement verdict classification
- [x] Write `verification.json`

### Completed Notes
- Deterministic verifier script created.
- Verifier checks structural integrity of evaluation outputs.
- Verifier computes behavioral signals from run artifacts.
- Verifier writes `verification.json` into the run folder.
- Diagnosis and verdict classification are available for downstream review.

### Done Criteria
- [x] Running verifier on a known dead run produces:
  - `verdict = FAIL`
  - `diagnosis = inactive_policy`
- [x] Running verifier writes `verification.json` successfully

---

## Milestone 3 — Build LLM input packet

### Status
**DONE**

### Tasks
- [x] Create `scripts/build_llm_input_packet.py`
- [x] Define compact packet schema
- [x] Include verdict and diagnosis
- [x] Include key metrics
- [x] Include blocked reason counts
- [x] Include filtered step samples
- [x] Exclude full raw `steps.csv`
- [x] Write `llm_input_packet.json`

### Completed Notes
- Packet builder created and working.
- Packet schema is compact and readable.
- Packet includes key metrics, diagnosis context, and evidence summaries.
- Raw full `steps.csv` is intentionally excluded.
- Output is suitable for downstream LLM review or manual inspection.

### Done Criteria
- [x] Packet is generated from a real run
- [x] Packet is compact and readable
- [x] Packet includes enough evidence for diagnosis without raw full logs

---

## Milestone 4 — Build LLM review layer

### Status
**NOT STARTED**

### Tasks
- [ ] Create `prompts/eval_review_prompt.txt`
- [ ] Create `scripts/review_with_llm.py`
- [ ] Load packet and prompt
- [ ] Call LLM
- [ ] Save `llm_review.md`

### Done Criteria
- [ ] A real run generates `llm_review.md`
- [ ] Review includes:
  - Summary
  - Evidence
  - Interpretation
  - Top recommended changes
  - What to verify next run
  - Confidence

### Notes
- This is the next major missing layer in the workflow.
- Once completed, this milestone can be integrated into the unified pipeline.

---

## Milestone 5 — Build orchestration

### Status
**IN PROGRESS**

### Original Tasks
- [ ] Create `scripts/run_train_eval_review.py`
- [x] Run `train.py`
- [x] Run `evaluate.py`
- [x] Run verifier
- [x] Run packet builder
- [ ] Run LLM review
- [x] Print/save final output paths

### Current Implementation Notes
- Instead of `scripts/run_train_eval_review.py`, the project now uses:
  - `scripts/run_post_eval.py`
  - `scripts/run_pipeline.py`
- `scripts/run_post_eval.py` runs:
  - `scripts/verify_eval_output.py`
  - `scripts/build_llm_input_packet.py`
- `scripts/run_pipeline.py` supports:
  - default evaluate + post-eval
  - `--train`
  - `--compare`
  - `--all`
  - `--evaluate-only`
  - `--post-eval-only`
  - `--compare-only`

### Current Status Summary
- Core orchestration exists and works for train/evaluate/verifier/packet/compare.
- LLM review stage is not yet wired in because Milestone 4 is not complete.

### Done Criteria
- [ ] One command runs the whole workflow end-to-end
- [x] Outputs are saved into the report folder

### Remaining Work
- [ ] Add LLM review stage into unified pipeline
- [ ] Optionally print final paths for review artifact as well

---

## Milestone 6 — Real-run validation

### Status
**IN PROGRESS**

### Tasks
- [ ] Test on known zero-trade run
- [ ] Test on a run with valid setup bars
- [ ] Confirm diagnoses are sensible
- [ ] Confirm review recommendations are grounded in metrics

### Current Notes
- Artifact generation and deterministic verification are in place.
- Informal validation has likely occurred during iteration, but formal milestone signoff is still pending.
- This milestone depends partly on Milestone 4 for full review-output validation.

### Done Criteria
- [ ] Dead runs are classified as dead
- [ ] Active runs are not falsely labeled dead
- [ ] Review outputs are useful and consistent

---

## Current Implemented Scripts

### Root scripts
- `train.py`
- `evaluate.py`

### Post-eval scripts
- `scripts/verify_eval_output.py`
- `scripts/build_llm_input_packet.py`
- `scripts/run_post_eval.py`

### Comparison / orchestration scripts
- `scripts/compare_runs.py`
- `scripts/run_pipeline.py`

---

## Current Workflow

### Basic evaluation workflow
```powershell
python evaluate.py
python scripts/run_post_eval.py
```

### Unified default workflow
```powershell
python scripts/run_pipeline.py
```

Runs:
- `evaluate.py`
- `scripts/run_post_eval.py`

### Full workflow
```powershell
python scripts/run_pipeline.py --all
```

Runs:
- `train.py`
- `evaluate.py`
- `scripts/run_post_eval.py`
- `scripts/compare_runs.py`

---

## Current Artifacts

### Model pointers
- `models/latest_model.txt`

### Run pointers
- `reports/latest_run.txt`
- `reports/best_run.txt` *(planned / optional)*

### Per-run artifacts
- `equity_curve.png`
- `steps.csv`
- `trades.csv`
- `trade_breakdown.csv`
- `eval_summary.json`
- `verification.json`
- `llm_input_packet.json`

### Comparison artifacts
- `reports/comparisons/run_comparison_YYYYMMDD_HHMMSS.csv`

---

## Next Recommended Milestones

### Immediate Next
1. Complete Milestone 4 — Build LLM review layer
2. Finish Milestone 5 by wiring review into pipeline
3. Finish Milestone 6 with formal real-run validation

### After That
4. Add best-run promotion workflow
   - `scripts/pick_best_run.py`
   - `reports/best_run.txt`
   - optional `models/best_model.txt`

5. Expand diagnostics
   - blocked-reason analysis
   - setup-level performance
   - session/day breakdown
   - drawdown and equity diagnostics

---

## Definition of “Current Success”

The project now has a stable and repeatable evaluation backbone:

- evaluation artifacts are consistently written
- deterministic verification is available
- compact LLM-ready packets are available
- post-eval automation exists
- unified pipeline exists
- comparison workflow exists

The main missing capability is automated LLM review generation and formal validation signoff.

---

## Next Action Items

- [ ] Create `prompts/eval_review_prompt.txt`
- [ ] Create `scripts/review_with_llm.py`
- [ ] Save `llm_review.md` into run folder
- [ ] Add review stage into `scripts/run_pipeline.py`
- [ ] Run formal validation against dead and active runs
- [ ] Optionally add `scripts/pick_best_run.py`
