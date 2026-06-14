# Generative DRO Reproduction

Two standalone examples:

- `examples/outage`: flow-based scenario generation for county outage counts.
- `examples/portfolio`: flow-generated return scenarios followed by DRO portfolio optimization.

## Setup

```bash
cd generative_dro_repro
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CUDA is optional but recommended for full reruns.

## Data

- Outage: PowerOutage.us outage-count records in `examples/outage/data/outage_counts.csv`.
- Portfolio: Yahoo Finance adjusted daily close prices in `examples/portfolio/data/yfinance`.

## Run

```bash
cd examples/outage
python train.py
python plot.py

cd ../portfolio
python train.py
python plot.py
```

Useful options:

```bash
python train.py --device cuda --output-dir results/default
python plot.py --output-dir results/default
```

## Experiment Settings

Outage:

- Data: 10 Atlanta-area counties; 15-minute outage-count vectors.
- Window: `2018-01-01T00:00:00` to `2024-12-31T23:45:00`.
- Scaling: `log1p` counts followed by countywise mean/std scaling.
- Flow model: MLP width 256, depth 4, time embedding 64.
- Flow training: 100 epochs, batch size 1024, AdamW lr `1e-3`, weight decay `1e-5`.
- Sampling: reverse Euler, 100 steps, 4096 nominal samples.
- Metrics: RBF MMD² in `log1p` space; marginal histograms and county-correlation heatmaps.

Portfolio:

- Data: six ETFs, `SPY IWM EFA EEM AGG GLD`; daily adjusted close prices.
- Window: `2023-01-01` to `2025-12-31`; train through `2024-12-31`.
- Scaling: global-scaled returns, $(R-\mu)/s$, with $s$ the mean absolute training return.
- Flow model: MLP width 256, depth 4, time embedding 64.
- Flow training: 20000 epochs, batch size 1024, AdamW lr `1e-3`, weight decay `1e-5`.
- Sampling: reverse Euler, 100 steps, one nominal return sample per training day.
- DRO: softplus shortfall with `q=0.02`, `beta=4`; gammas `0.01 0.1 1 10`.
- DRO training: equal-weight robust initialization; 100000 nominal steps and 100000 robust GDA steps.
- Metrics: RBF MMD² in global-scaled return space; test and stress-test portfolio losses.

## Outputs

Outputs are flat: the result root contains files plus one `figures/` directory.

```text
# outage
results/default/
  config.json
  flow_loss.csv
  metrics.csv
  nominal.csv
  figures/

# portfolio
results/default/
  config.json
  flow_loss.csv
  dro_loss.csv
  metrics.csv
  weights.csv
  nominal.csv
  worst_case.csv
  figures/
```

Local full-run checks:

```text
outage mmd2_log1p ~= 1.7e-4
portfolio flow joint_mmd2 ~= 4.5e-4
portfolio best robust gamma by test/stress loss ~= 0.1
```
