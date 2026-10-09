# Project issues and planned improvements

This document records the current implementation and planned improvements for the project.

## Data scaling

**Current implementation:** MinMaxScaler is fitted on all prices before the chronological split.

**Planned change:** Fit the scaler on training observations and use the fitted scaler to transform validation and test observations.

## Model selection

**Current implementation:** `src/grid_search.py` ranks configurations using test metrics and selects a configuration using test R².

**Planned change:** Select parameters using validation metrics, then evaluate the selected configuration on the test set.

## Forecast target alignment

**Current implementation:** `src/train.py` and `src/grid_search.py` create rolling windows before splitting training and validation samples. Adjacent samples can share target dates across the split.

**Planned change:** Split samples by target dates or remove boundary-crossing samples.

## Checkpoint handling

**Current implementation:** The grid-search trainer records the best validation loss without restoring the associated weights. The main trainer restores the best checkpoint in the early-stopping branch.

**Planned change:** Restore the best validation checkpoint after both early stopping and normal training completion.

## Baseline and forecast horizons

**Current implementation:** Evaluation pools predictions across windows and forecast horizons.

**Planned change:** Report separate day-1, day-2, and day-3 metrics. Add a persistence baseline that repeats the last observed price and evaluate it on the same forecast origins as the LSTM.

## Visualization

**Current implementation:** Some figures in `src/visualize_data.py` use trend extrapolation and random noise. The script also calculates a rolling standard deviation of price levels.

**Planned change:** Identify illustrative forecasts in figure labels and generate prediction figures from trained-model outputs. Use labels that match the calculated quantity, or calculate return volatility for volatility figures.

## Reproducibility

**Planned change:** Set random seeds before model initialization and record experiment settings. Check date ordering, duplicate dates, missing days, and input-target alignment.

## Poster and experiment results

**Current record:** The poster describes the dataset as covering 2015–2025 and reports R² = 0.9656 and RMSE of approximately USD 3,751.

**Planned change:** Rerun the experiments using the updated evaluation procedure. Record the dataset, model architecture, training parameters, and metrics for each run, and update the poster's figures and conclusion to match those records.

## Input validation and dates

**Current implementation:** The sorting check compares the first and last dates. Date parsing exceptions are caught and execution continues. Validation samples are selected using a size calculated from the training sample count.

**Planned change:** Sort observations by parsed date; define handling for duplicate dates and missing calendar days; validate required columns, dates, numeric prices, and the cleaned dataset. Check partition sizes against the lookback and forecast horizon, and explicitly handle a zero-sized validation split.

## Dataset documentation

**Planned change:** Record the source URL, retrieval date, actual date range, currency, price definition, missing observations, and cleaning steps.
