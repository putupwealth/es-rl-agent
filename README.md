# ES RL Agent

A reinforcement learning trading research project for ES futures using key market levels, higher-timeframe bias, and disciplined reward design.

## Project Goal

Build a structure-first RL trading agent that learns to trade fewer, higher-quality setups around important levels.

Current Version 1 focus:

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
Evaluation + trade logs
        ↓
Reward/feature tuning

Folder Structure

es-rl-agent/
│
├── data/                  # Local market data, ignored by Git
├── models/                # Trained RL models, ignored by Git
├── reports/               # Evaluation charts/trade logs, ignored by Git
│
├── src/
│   ├── __init__.py
│   ├── levels.py          # PDH/PDL and breakout feature logic
│   ├── features.py        # Higher-timeframe trend/bias features
│   └── env.py             # Gymnasium RL trading environment
│
├── download_databento.py  # Downloads ES data from Databento
├── validate_data.py       # Validates downloaded data
├── test_levels.py         # Tests PDH/PDL logic
├── test_bias.py           # Tests 1H/4H bias features
├── test_env.py            # Validates RL environment
├── train.py               # Trains PPO model
├── evaluate.py            # Evaluates trained model
│
├── requirements.txt
├── .gitignore
└── README.md

Current Data Source

Data is downloaded from Databento.

Current dataset:

Dataset: GLBX.MDP3
Symbol: ES.v.0
Schema: ohlcv-1m
Sessions: ETH + RTH
Timezone: US/Eastern

The downloaded CSV includes:

timestamp
open
high
low
close
volume
symbol
session
is_rth
is_eth
is_roll_period

Data files are not committed to GitHub.

Environment Setup

Create and activate virtual environment:

python -m venv venv
venv\Scripts\activate

Install dependencies:

python -m pip install -r requirements.txt

If adding new packages:

python -m pip freeze > requirements.txt

Required Environment Variables

Create a local .env file:

DATABENTO_API_KEY=your_key_here

Never commit .env to GitHub.

Data Download

Run:
python download_databento.py

This downloads ES 1-minute data and saves it under:

data/ES_1min_all_sessions.csv

Data Validation

Run:

python validate_data.py

Validation checks:

Required columns
Timestamp parsing
Duplicate timestamps
Null values
Session counts
Price sanity
Roll-period flags
Feature Pipeline
src/levels.py

Adds:

PDH
PDL
PDC
near_PDH
near_PDL
break_above_PDH
break_below_PDL
first_break_above_PDH
first_break_below_PDL
retest_PDH
retest_PDL

Important distinction:

break_above_PDH = price is above PDH
first_break_above_PDH = price crossed above PDH on this candle
src/features.py

Adds higher-timeframe context:

trend_1h_up
trend_1h_down
trend_4h_up
trend_4h_down
bias_long
bias_short

Bias logic:

bias_long = 1H trend up AND 4H trend up
bias_short = 1H trend down AND 4H trend down
RL Environment

The environment is defined in:

src/env.py

Agent actions:

0 = HOLD
1 = LONG
2 = SHORT
3 = EXIT

Current V1 risk controls:

RTH-only entries
max trades per episode
max hold bars
stop loss
take profit
forced exit outside RTH
commission
drawdown penalty

Observation includes:

PDH/PDL breakout features
near-level flags
1H/4H bias
RTH/ETH flags
roll-period flag
current position
unrealized PnL
bars held
Training

Run:

python train.py

Current model:

Algorithm: PPO
Policy: MlpPolicy
Library: Stable-Baselines3

Model output:

models/es_pdh_pdl_ppo_v1.zip
Evaluation

Run:

python evaluate.py

Evaluation creates:

reports/es_pdh_pdl_equity_curve_v1.png
reports/es_pdh_pdl_trades_v1.csv

Evaluation checks:

final equity
action counts
logged trades
entry time
exit time
direction
entry price
exit price
PnL
entry confluence
Current Known Behavior

The first working model is not profitable yet.

Current observations:

Model can train successfully
Model can evaluate successfully
Environment risk controls prevent multi-day disasters
Agent currently tends to short too often
Entry rules need to be made stricter
Next improvement should restrict long/short actions by directional breakout validity
Next Planned Improvement

Current issue:

Agent can short near PDL even without a clean PDL breakdown.

Next fix:

Long entries only allowed near/above PDH breakout context.
Short entries only allowed near/below PDL breakdown context.

Stricter V2 logic:

valid_long_zone =
    first_break_above_PDH == 1
    OR break_above_PDH == 1

valid_short_zone =
    first_break_below_PDL == 1
    OR break_below_PDL == 1

Later expansion:

IBH / IBL levels
ORH / ORL levels
VWAP
Volume profile
Order-flow features
LLM evaluation coach
Important Rules

Do not commit:

data/
models/
reports/
.env
venv/

These are intentionally ignored by Git.

Development Workflow

Recommended workflow:

venv\Scripts\activate
python validate_data.py
python test_levels.py
python test_bias.py
python test_env.py
python train.py
python evaluate.py

After code changes:

git status
git add .
git commit -m "Describe change"
git push
Project Philosophy

This project is not trying to build a magic trading bot.

The goal is to build a disciplined research system that can:

Test trading ideas objectively
Learn behavior around key levels
Avoid emotional human mistakes
Prefer fewer, higher-quality trades
Improve through repeated evaluation and reward tuning
Disclaimer

This is a research and simulation project only.

It is not financial advice.
It is not ready for live trading.
Do not connect to a broker until the model is stable in unseen test data and paper trading.


Also update `.gitignore` to this:

```gitignore
venv/
__pycache__/
*.pyc

data/
models/
reports/

.env
.DS_Store

Then commit:

git add README.md .gitignore
git commit -m "Add project documentation"
git push