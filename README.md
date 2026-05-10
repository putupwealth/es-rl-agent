# ES RL Agent

A reinforcement learning trading research project for ES futures using key market levels, higher-timeframe bias, and disciplined reward design.

## Project Goal

Build a structure-first RL trading agent that learns to trade fewer, higher-quality setups around important levels.

Current Version 2 focus:

- ES futures
- PDH / PDL breakout behavior
- RTH-only entries
- 1H + 4H trend bias
- PPO reinforcement learning
- Simulated environment only
- No live trading

## Strategy Concept

The agent should learn how price behaves around previous day levels:

- Clean PDH breakout → possible long
- Clean PDL breakdown → possible short
- Fake breakout → avoid or exit
- Range day → stay flat
- Sweep and reversal → learn through reward/penalty later

This is not a scalping system. The goal is quality over quantity.

## Architecture

```text
Databento ES 1m data
        ↓
Data validation
        ↓
PDH / PDL level engine
        ↓
Breakout event detection
        ↓
1H / 4H bias features
        ↓
Gymnasium RL environment
        ↓
PPO training
        ↓
Evaluation + verification + packet build
        ↓
Run comparison + reward/feature tuning
```

## Folder Structure

```text
es-rl-agent/
│
├── data/                          # Local market data, ignored by Git
├── models/                        # Trained RL models, ignored by Git
│   ├── latest_model.txt           # Pointer to latest trained model
│   └── *.zip
│
├── reports/                       # Evaluation artifacts, ignored by Git
│   ├── latest_run.txt             # Pointer to latest evaluation run
│   ├── best_run.txt               # Optional pointer to best run
│   ├── comparisons/               # Run comparison CSV outputs
│   └── <run_id>/
│       ├── equity_curve.png
│       ├── trades.csv
│       ├── steps.csv
│       ├── trade_breakdown.csv
│       ├── eval_summary.json
│       ├── verification.json
│       └── llm_input_packet.json
│
├── scripts/
│   ├── verify_eval_output.py      # Deterministic verifier
│   ├── build_llm_input_packet.py  # Compact packet builder for LLM review
│   ├── run_post_eval.py           # Wrapper: verify + packet build
│   └── compare_runs.py            # Compare evaluation runs side by side
│
├── src/
│   ├── __init__.py
│   ├── levels.py                  # PDH/PDL and breakout feature logic
│   ├── features.py                # Higher-timeframe trend/bias features
│   └── env.py                     # Gymnasium RL trading environment
│
├── download_databento.py          # Downloads ES data from Databento
├── validate_data.py               # Validates downloaded data
├── test_levels.py                 # Tests PDH/PDL logic
├── test_bias.py                   # Tests 1H/4H bias features
├── test_env.py                    # Validates RL environment
├── train.py                       # Trains PPO model
├── evaluate.py                    # Evaluates trained model
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Current Data Source

Data is downloaded from Databento.

Current dataset:

- Dataset: `GLBX.MDP3`
- Symbol: `ES.v.0`
- Schema: `ohlcv-1m`
- Sessions: `ETH + RTH`
- Timezone: `US/Eastern`

The downloaded CSV includes:

- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `symbol`
- `session`
- `is_rth`
- `is_eth`
- `is_roll_period`

Data files are not committed to GitHub.

## Environment Setup

Create and activate virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

If adding new packages:

```powershell
python -m pip freeze > requirements.txt
```

## Required Environment Variables

Create a local `.env` file:

```env
DATABENTO_API_KEY=your_key_here
```

Never commit `.env` to GitHub.

## Data Download

Run:

```powershell
python download_databento.py
```

This downloads ES 1-minute data and saves it under:

```text
data/ES_1min_all_sessions.csv
```

## Data Validation

Run:

```powershell
python validate_data.py
```

Validation checks:

- required columns
- timestamp parsing
- duplicate timestamps
- null values
- session counts
- price sanity
- roll-period flags

## Feature Pipeline

### `src/levels.py`

Adds:

- `PDH`
- `PDL`
- `PDC`
- `near_PDH`
- `near_PDL`
- `break_above_PDH`
- `break_below_PDL`
- `first_break_above_PDH`
- `first_break_below_PDL`
- `retest_PDH`
- `retest_PDL`

Important distinction:

- `break_above_PDH` = price is above PDH
- `first_break_above_PDH` = price crossed above PDH on this candle

### `src/features.py`

Adds higher-timeframe context:

- `trend_1h_up`
- `trend_1h_down`
- `trend_4h_up`
- `trend_4h_down`
- `bias_long`
- `bias_short`

Bias logic:

- `bias_long = 1H trend up AND 4H trend up`
- `bias_short = 1H trend down AND 4H trend down`

## RL Environment

The environment is defined in:

```text
src/env.py
```

Agent actions:

- `0 = HOLD`
- `1 = LONG`
- `2 = SHORT`
- `3 = EXIT`

Current V2 risk controls:

- RTH-only entries
- max trades per episode
- max hold bars
- stop loss
- take profit
- forced exit outside RTH
- commission

Observation includes:

- PDH/PDL breakout features (binary)
- near-level flags (binary)
- retest-level flags (binary)
- bars since first long/short break (normalised continuous)
- distance to PDH/PDL in points (normalised continuous)
- 1H/4H bias
- RTH/ETH flags
- roll-period flag
- current position
- unrealized PnL
- bars held

## Training

Run:

```powershell
python train.py
```

Other examples:

```powershell
python train.py --version 2 --seed 42
python train.py --run-id es_pdh_pdl_v2_s42_20260510_154210
python train.py --model-dir models --latest-pointer models/latest_model.txt
```

Current model setup:

- Algorithm: `PPO`
- Policy: `MlpPolicy`
- Library: `Stable-Baselines3`
- Training data: RTH-only bars (`is_rth == 1`, `is_roll_period == 0`)
- Episode length: ~390 bars (one RTH session)
- Entropy coefficient: `0.01` (explicit exploration pressure)

Model output:

```text
models/{run_id}.zip
```

Default run ID format:

```text
es_pdh_pdl_v2_s42_YYYYMMDD_HHMMSS
```

Latest trained model pointer:

```text
models/latest_model.txt
```

That pointer should contain a repo-relative path, for example:

```text
models/es_pdh_pdl_v2_s42_20260509_201830.zip
```

## Evaluation

Run:

```powershell
python evaluate.py
```

Other examples:

```powershell
python evaluate.py --model-file models/es_pdh_pdl_v2_s42_20260509_201830.zip
python evaluate.py --latest-model-pointer models/latest_model.txt
python evaluate.py --reports-dir reports
python evaluate.py --latest-run-pointer reports/latest_run.txt
python evaluate.py --run-id custom_run_name
```

Evaluation creates:

```text
reports/{run_id}/equity_curve.png
reports/{run_id}/trades.csv
reports/{run_id}/steps.csv
reports/{run_id}/trade_breakdown.csv
reports/{run_id}/eval_summary.json
```

Latest evaluation pointer:

```text
reports/latest_run.txt
```

That pointer contains the latest real run directory, for example:

```text
reports/es_pdh_pdl_v2_s42_20260509_201830
```

Evaluation checks:

- final equity
- action counts
- logged trades
- entry time
- exit time
- direction
- entry price
- exit price
- PnL
- entry confluence
- eligibility diagnostics (valid-zone frequency, entry-attempt rate on valid vs invalid bars)

## Post-Evaluation Pipeline

After evaluation, run:

```powershell
python scripts/run_post_eval.py
```

This wrapper runs:

```powershell
python scripts/verify_eval_output.py reports/latest_run.txt
python scripts/build_llm_input_packet.py reports/latest_run.txt
```

### Verifier

Run directly if needed:

```powershell
python scripts/verify_eval_output.py reports/latest_run.txt
python scripts/verify_eval_output.py reports/<run_id>
```

Writes:

```text
reports/<run_id>/verification.json
```

Possible high-level outputs include:

- `PASS`
- `WARN`
- `FAIL`

Possible diagnoses include:

- `behaviorally_alive`
- `active_but_unprofitable`
- `active_but_blocked`
- `invalid_action_heavy`
- `inactive_policy`
- `no_setup_opportunity`
- `missing_or_invalid_outputs`

### LLM Input Packet

Run directly if needed:

```powershell
python scripts/build_llm_input_packet.py reports/latest_run.txt
python scripts/build_llm_input_packet.py reports/<run_id>
```

Writes:

```text
reports/<run_id>/llm_input_packet.json
```

This packet is a compact summary artifact for downstream review and analysis.

## Run Comparison

Run:

```powershell
python scripts/compare_runs.py
```

Common examples:

```powershell
python scripts/compare_runs.py --reports-dir reports
python scripts/compare_runs.py --reports-dir reports --latest 5
python scripts/compare_runs.py --reports-dir reports --rank-by
python scripts/compare_runs.py --reports-dir reports --only-pass
python scripts/compare_runs.py --reports-dir reports --csv-dir reports/comparisons
python scripts/compare_runs.py --reports-dir reports --latest 10 --csv-dir reports/comparisons
python scripts/compare_runs.py reports/latest_run.txt
python scripts/compare_runs.py reports/run_a reports/run_b
```

By default, comparison CSV output is written under:

```text
reports/comparisons/
```

with a timestamped filename such as:

```text
reports/comparisons/run_comparison_YYYYMMDD_HHMMSS.csv
```

Comparison helps review:

- verdict
- diagnosis
- total PnL
- total trades
- valid vs invalid attempts
- win rate
- final realized equity

## Current Known Behavior

Earlier versions often chose action `0` (hold) or action `3` (exit, a no-op while flat), resulting in zero trades.

Root causes diagnosed and fixed in this version:

1. Duplicate no-op while flat: action `3` now carries a `-0.5` penalty when `position == 0`.
2. ETH training noise: training restricted to RTH-only bars.
3. Entry gate penalties too harsh: RTH-gate penalty `-2 → -0.5`; zone-gate penalty `-3 → -1`.
4. Ad hoc reward shaping removed: stop/TP/hold exit bonuses, bias entry bonuses, manual-exit tiered bonuses, and drawdown penalty all removed; mark-to-market is now the primary signal.
5. Low exploration: `ent_coef=0.01` added to PPO.
6. Richer observation: `retest_PDH/PDL` flags, `bars_since_long/short_break` (normalised), and `dist_to_PDH/PDL` (normalised) added to the observation vector.
7. Shorter episodes: `max_steps` capped at `390` (~one RTH session) to improve credit assignment.
8. Deterministic post-eval verification added for faster behavioral diagnosis.
9. Compact LLM input packet added for structured downstream review.
10. Run comparison workflow added for side-by-side evaluation review.

## Planned Expansion

Later expansion may include:

- IBH / IBL levels
- ORH / ORL levels
- VWAP
- volume profile
- order-flow features
- LLM evaluation coach

## Important Rules

Do not commit:

- `data/`
- `models/`
- `reports/`
- `.env`
- `venv/`

These are intentionally ignored by Git.

## Development Workflow

Recommended workflow:

```powershell
venv\Scripts\activate
python validate_data.py
python test_levels.py
python test_bias.py
python test_env.py
python train.py
python evaluate.py
python scripts/run_post_eval.py
python scripts/compare_runs.py
```

After code changes:

```powershell
git status
git add .
git commit -m "Describe change"
git push
```

## Suggested `.gitignore`

```gitignore
venv/
__pycache__/
*.pyc

data/
models/
reports/

.env
.DS_Store
```

## Project Philosophy

This project is not trying to build a magic trading bot.

The goal is to build a disciplined research system that can:

- test trading ideas objectively
- learn behavior around key levels
- avoid emotional human mistakes
- prefer fewer, higher-quality trades
- improve through repeated evaluation and reward tuning

## Disclaimer

This is a research and simulation project only.

- It is not financial advice.
- It is not ready for live trading.
- Do not connect to a broker until the model is stable in unseen test data and paper trading.
