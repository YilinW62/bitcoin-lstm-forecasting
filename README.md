# Bitcoin Price Forecasting with LSTM

A research project by Yilin Wang exploring short-horizon Bitcoin price forecasting with PyTorch.

## Project status

This repository preserves the original implementation as a starting point for a reproducibility and evaluation review. **The reported historical results have not been independently reproduced, and the current evaluation has known data-leakage and model-selection limitations.** See [evaluation notes](docs/evaluation-notes.md) before interpreting the metrics.

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

The defaults in `config.py` differ from some defaults in `train.py`; command-line defaults take precedence in the training entry point.

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
| `reports/README.md` | Poster status |

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

These instructions reflect the script interface; a full training run has not yet been verified in this repository. Dependency versions are not yet locked.

The standalone visualization script currently reads `Bitcoin_Historical_Data_10years.xlsx` from the working directory. Update that path before using a differently named file.

## Results and interpretation

The original poster reports R² = 0.9656 and RMSE approximately USD 3,751. These are **historical, unverified results from the original evaluation**, not validated performance claims. The poster plots and summary also need to be reconciled with the exact run configuration.

No claim is made that this model outperforms a last-price baseline, forecasts returns reliably, or supports a profitable trading strategy.

## Next steps

1. Fit the scaler on training observations only.
2. Select hyperparameters using validation data, leaving the test set untouched.
3. Prevent multi-day target overlap across training and validation boundaries.
4. Compare against a last-observed-price baseline on identical forecast dates.
5. Report separate metrics for each forecast horizon.
6. Restore the best validation checkpoint consistently and record reproducible run settings.
7. Add meaningful tests for chronological splits, scaling, and sequence alignment.
8. Rerun experiments and update figures and the poster.

## Research materials

The [original project poster](reports/poster.pdf) covers Bitcoin data described as 2015–2025 and is preserved as supplied. Its historical results have not been independently reproduced. Planned revisions will reconcile the poster text, figures, and reported results with reproducible experiments. See [reports/README.md](reports/README.md) and [evaluation notes](docs/evaluation-notes.md).

## Attribution

This repository contains Yilin Wang's submitted project materials. Third-party data and diagrams retain their respective rights. No open-source license has been selected yet.

## English-only source update

All Python comments, docstrings, console messages, chart labels, and help text are in English; decorative emoji have been removed. The default Excel filename is now `Bitcoin_Historical_Data_10years.xlsx`. Rename a local input file accordingly, or pass its actual path with `--data_file` to the training and grid-search scripts. The visualization script uses the default filename directly.

The exported `best_config.json` metadata keys are now `best_parameters`, `performance_metrics`, `search_duration`, and `search_date`. Numerical model and training logic are unchanged by this language update.
