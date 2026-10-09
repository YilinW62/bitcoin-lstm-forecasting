# Bitcoin Price Forecasting with LSTM

A research project by Yilin Wang exploring short-horizon Bitcoin price forecasting with PyTorch.

## Research question

How well can an LSTM use a window of historical daily Bitcoin prices to forecast the next several days, and how should those forecasts be evaluated?

## Implementation

- Loads an Excel dataset, converts the price column, cleans invalid prices, and orders observations.
- Builds rolling input windows and multi-day targets.
- Implements both encoder-decoder and direct multi-output LSTM architectures.
- The training entry point uses the **direct multi-output LSTM**.
- Command-line training defaults: 60-day lookback, 3-day forecast horizon, 50 epochs, 50 hidden units, and learning rate 0.0001.
- Reports RMSE, MAE, and R² and generates training and prediction plots.
- Includes a grid-search script and exploratory visualizations.

## Repository guide

| Path | Purpose |
| --- | --- |
| `train.py` | Training, evaluation, and multi-step forecasts |
| `model.py` | LSTM architectures |
| `data_processor.py` | Loading, cleaning, scaling, and sequence construction |
| `config.py` | Configuration defaults |
| `grid_search.py` | Original hyperparameter search |
| `visualize_data.py` | Exploratory plots and explicitly simulated illustrations |
| `requirements.txt` | Direct Python dependencies |
| `data/README.md` | Data acquisition and placement instructions |
| `docs/evaluation-notes.md` | Known limitations and improvement plan |
| `reports/poster.pdf` | Original project poster |
| `reports/README.md` | Poster information |

## Setup and run

Use a Python virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead.

Obtain the input data as described in [data/README.md](data/README.md), then run from the repository root:

```bash
python train.py --data_file "data/bitcoin_daily.xlsx" --device cpu --time_step 60 --pred_days 3
```

For available search options:

```bash
python grid_search.py --help
```

The standalone visualization script currently reads `Bitcoin_Historical_Data_10years.xlsx` from the working directory. Update that path before using a differently named file.

## Research materials

[View the project poster](reports/poster.pdf): "Application of LSTMs to Bitcoin Price Prediction" by Yilin Wang.

## Attribution

This repository contains Yilin Wang's submitted project materials. Third-party data and diagrams retain their respective rights.
