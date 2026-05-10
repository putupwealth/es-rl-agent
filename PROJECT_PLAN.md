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



# RL Research Loop

This project is no longer just a training script.

It is becoming an **RL research loop**.

## Core idea

- **PPO learns the policy**
- **The LLM helps design better training conditions for PPO**

The LLM does **not** replace reinforcement learning.  
It improves the **decision-making around PPO experiments**.

---

## Research loop

The workflow is:

1. **Train model**
2. **Evaluate behavior**
3. **Verify behavior deterministically**
4. **Summarize evidence compactly**
5. **Use LLM to interpret evidence**
6. **Choose next reward / feature / rule change**
7. **Retrain**

Then repeat.

---

## What PPO does

PPO is responsible for learning from:

- observations
- actions
- rewards
- episode transitions

PPO updates the policy weights and learns which actions produce better long-term reward under the current environment design.

---

## What the LLM does

The LLM is not the policy learner.

Its role is to act as a **research reviewer** that reads run artifacts and helps answer:

- What is the policy doing?
- What is it doing wrong?
- Why is it likely happening?
- What should be changed next?
- What should be verified in the next run?
- Is the current reward system helping or hurting learning?
- Should the reward system stay the same, be simplified, or be adjusted?

The LLM improves the experimental loop by turning run evidence into structured next-step recommendations.

---

## Why this matters

Without this layer, RL iteration can become random trial and error:

- tweak reward
- retrain
- inspect results manually
- guess next change
- repeat

With an LLM review layer, the process becomes more systematic:

- verify what happened
- summarize the evidence
- interpret likely causes
- propose targeted changes
- define what success looks like next run

This makes the project more disciplined and easier to scale.

---

## What the LLM should help improve

The LLM should help guide decisions about:

- reward shaping
- whether the current reward system is good enough to keep
- when a reward system should stay unchanged
- when a reward term should be softened or removed
- when one small new shaping term is justified
- action gating
- entry and exit rules
- observation features
- time-of-day constraints
- overtrading controls
- drawdown handling
- validation priorities
- next-run experiment design

Examples:

- reduce invalid entry attempts
- discourage overtrading
- penalize holding losers
- restrict entries to stronger time windows
- add time-window features to observation
- keep the current reward system unchanged for another validation run
- simplify reward if penalties are stacking too aggressively
- test one reward change at a time

---

## Reward system assessment

A key role of the LLM is to evaluate whether the current reward system is producing the intended behavior.

The LLM should help answer:

- Is the current reward design encouraging good trades?
- Is it causing overtrading?
- Is it allowing random entries?
- Is it failing to discourage holding losers?
- Is it creating too much inactivity?
- Is it managing drawdown well enough?
- Should the reward system stay as it is, or should it change?

The goal is **not** to change reward every run.

The goal is to determine whether the current reward system is:

- working well enough to keep
- slightly too harsh
- slightly too weak
- overly complex
- sending mixed signals
- causing unintended behavior

A strong LLM review should say whether the next best step is:

- **keep the current reward system**
- **simplify the reward system**
- **soften an existing penalty**
- **add one small shaping term**
- **remove one harmful shaping term**

This makes the LLM part of the **reward-design feedback loop**, not part of PPO training itself.

---

## What “good” means in this project

A “good” PPO policy is **not** just one that trades more.

A good policy should show behavior like:

- trades only when conditions are favorable
- avoids random or low-quality entries
- does not overtrade
- cuts losing trades reasonably
- does not hold losers too long
- respects intended market/session constraints
- produces cleaner, more explainable decisions
- improves risk-adjusted behavior, not just raw activity

In this project, “good” means the policy becomes:

### 1. More selective
It should trade when valid opportunity exists, not just when action noise happens.

### 2. Less random
Entries should align with:
- valid breakout context
- directional bias
- intended session/time filters
- other explicit setup logic

### 3. Less overactive
The agent should avoid churning many weak trades just to stay active.

### 4. Better at risk control
It should:
- avoid excessive drawdown
- avoid holding losing trades too long
- exit bad trades more cleanly
- preserve capital better

### 5. More explainable
A good run should be understandable from artifacts:
- why entries happened
- why trades were blocked
- why exits happened
- what the main failure mode was

### 6. More consistent
Improvement should not rely on one lucky run.
A good policy should show repeatable improvement across comparable evaluations.

---

## How to judge whether a run is getting better

A run is getting better if it shows some combination of:

- more participation on valid setup bars
- fewer invalid entry attempts
- fewer blocked low-quality actions
- lower overtrading / churn
- shorter loser hold duration
- better drawdown behavior
- more stable trade quality
- clearer alignment with intended trading windows
- better consistency across repeated runs

Important:
A run is **not automatically better** just because:
- it took more trades
- it had one large winner
- raw PnL increased once

Behavior quality matters, not just one headline metric.

---

## What the LLM should not do

The LLM should **not**:

- claim it trained the PPO model
- replace deterministic verification
- invent metrics not present in the packet
- make vague recommendations without evidence
- assume the reward system must change every run
- act as financial advice

Its role is:

- interpret
- prioritize
- recommend
- help design better experiments

---

## Project architecture meaning

This project now has two different intelligence layers:

### 1. Learning layer
**PPO**
- learns trading behavior from environment feedback

### 2. Research layer
**LLM**
- reviews outputs
- diagnoses likely failure modes
- recommends the next best environment/reward/feature changes
- evaluates whether the current reward system should be kept or changed

Together they form a repeatable research workflow.

---

## Practical interpretation

A good way to think about it is:

> PPO learns the current game.  
> The LLM helps redesign the game so PPO can learn better behavior.

Or more concretely:

> PPO optimizes the policy.  
> The LLM improves the experiment design.

Also:

> PPO learns under the current reward system.  
> The LLM helps judge whether that reward system is good enough to keep.

---

## Current loop in this project

The intended loop is:

1. `train.py`
2. `evaluate.py`
3. `scripts/verify_eval_output.py`
4. `scripts/build_llm_input_packet.py`
5. `scripts/review_with_llm.py`
6. decide next environment/reward/feature change
7. retrain with `train.py`

Over time, this turns the repository into a structured RL experimentation system rather than a single-model training project.

---

## Goal of this approach

The goal is not to build a magic trading bot.

The goal is to build a disciplined system that can:

- test ideas objectively
- detect failure modes faster
- improve environment design systematically
- reduce random tuning
- support repeatable RL experimentation
- decide when a reward system is good enough to keep stable

That is the real value of the RL research loop.


