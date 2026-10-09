# Bitcoin Price Forecasting with LSTM

**Yilin Wang** · Time-series forecasting · Machine learning · Quantitative finance

This project explores short-horizon Bitcoin price forecasting using long short-term memory (LSTM) networks. It combines historical price analysis, multi-day prediction, and hyperparameter search in a Python workflow built with PyTorch.

[**View the research poster**](poster/poster.pdf) · [Source code](src/) · [Dataset](data/bitcoin_daily.xlsx) · [Project improvement plan](improvement-plan/evaluation-notes.md)

## Research objective

Investigate how historical daily Bitcoin prices can be used to forecast prices over the next several days, and examine how the lookback window, forecast horizon, learning rate, and network size affect predictive performance.

## Data

The project uses daily Bitcoin price data from **Investing.com**, covering **1 January 2015 to 23 September 2025**. The Excel dataset is included in [data/bitcoin_daily.xlsx](data/bitcoin_daily.xlsx).

Closing price is the model's input feature. Preprocessing includes date conversion, numeric price conversion, removal of missing and nonpositive prices, and chronological ordering. Prices are scaled with MinMaxScaler and organized into rolling input windows and multi-day targets.

## Methodology

1. **Explore the data:** visualize historical prices, moving averages, price distributions, and daily returns.
2. **Prepare sequences:** construct lookback windows and corresponding future price targets, using an 85% training-period and 15% test-period split, with a validation subset drawn from the training samples.
3. **Train the model:** use a direct multi-output LSTM to predict several future prices in one forward pass.
4. **Search hyperparameters:** vary lookback length, forecast horizon, hidden units, and learning rate.
5. **Evaluate and visualize:** calculate RMSE, MAE, and R², plot training history and test predictions, and generate a forecast from the final observed window.

The model module includes both an encoder-decoder LSTM and a direct multi-output LSTM. The training and grid-search scripts use the **direct multi-output model**.

## Research poster

The poster, *Application of LSTMs to Bitcoin Price Prediction*, presents the research motivation, LSTM methodology, exploratory analysis, hyperparameter search, and forecasting results.

The poster reports **R² = 0.9656** and **RMSE ≈ USD 3,751** for the selected model.

[**Read the full poster →**](poster/poster.pdf)

## Repository structure

| Location | Contents |
| --- | --- |
| `src/` | Python source code |
| `data/` | Bitcoin Excel dataset |
| `poster/` | Research poster |
| `improvement-plan/` | Project issues and planned improvements |
| `requirements.txt` | Python dependencies |

### Source files

| File | Purpose |
| --- | --- |
| `src/config.py` | Model and training settings |
| `src/data_processor.py` | Data loading, cleaning, scaling, and sequence preparation |
| `src/model.py` | LSTM architectures |
| `src/train.py` | Training, evaluation, forecasting, and result plots |
| `src/grid_search.py` | Hyperparameter search and comparison plots |
| `src/visualize_data.py` | Exploratory analysis and conceptual forecasting illustrations |

## Run the project

Run the following commands from the repository root.

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

### Train and forecast

```bash
python src/train.py --data_file data/bitcoin_daily.xlsx --device cpu --time_step 60 --pred_days 3
```

This command uses a 60-day lookback and a 3-day forecast horizon. Its remaining defaults are 50 training epochs, 50 LSTM hidden units, a batch size of 32, and a learning rate of 0.0001.

The script saves training and prediction plots, model weights, and the fitted scaler in the working directory.

### Run a quick hyperparameter search

```bash
python src/grid_search.py --data_file data/bitcoin_daily.xlsx --device cpu --quick
```

The quick search evaluates 16 configurations across lookback windows of 30 and 60 days, forecast horizons of 3 and 7 days, 50 and 100 hidden units, and learning rates of 0.0001 and 0.001. Each configuration uses up to 30 training epochs.

### Generate exploratory figures

The standalone visualization script reads `Bitcoin_Historical_Data_10years.xlsx` from the working directory. Copy the included Excel file to that name, then run:

```bash
python src/visualize_data.py
```

Use `python src/train.py --help` or `python src/grid_search.py --help` to view command-line options.

## Tools

Python · PyTorch · pandas · NumPy · scikit-learn · Matplotlib · openpyxl
