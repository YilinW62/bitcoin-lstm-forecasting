# Known issues and improvement plan

This document records issues found by inspecting the original source. It does not claim that they have already been corrected.

## Scaling leakage

`DataProcessor.normalize_data()` fits MinMaxScaler on all prices before temporal splitting. Fit on the training period only, and transform validation/test data using that fitted scaler.

## Model selection

`src/grid_search.py` ranks configurations using test metrics and uses test R² to recommend a configuration. Use validation metrics for selection, then evaluate the chosen configuration on held-out test data.

## Overlapping forecast targets

Both `src/train.py` and `src/grid_search.py` construct windows before splitting training and validation samples. Adjacent windows share target dates. Split by target dates or purge boundary-crossing targets so that a training label cannot also be a validation label.

## Checkpoint handling

The grid-search trainer tracks the best validation loss but does not restore the corresponding weights. The main trainer restores its checkpoint in the early-stopping branch; normal completion also needs to restore the best checkpoint.

## Baseline and horizons

Add a persistence baseline that repeats the last observed price for every horizon. Evaluate both models on identical forecast origins, with separate day-1, day-2, and day-3 scores. Current pooled metrics flatten overlapping windows and horizons.

## Visualization and claims

Some forecasts in `src/visualize_data.py` are simulated with trend extrapolation and random noise. They are illustrations, not trained-model outputs.

Direct multi-output prediction avoids recursive feedback of predicted prices; it does not eliminate forecast errors or establish superior performance.

The rolling standard deviation of price levels is not the same measure as return volatility. Label it accurately or calculate volatility from returns.

## Reproducibility

Seed randomness before model initialization, save run configurations and dependency versions, and validate date sorting, duplicates, missing days, and sequence alignment.

## Poster and historical results

The supplied poster describes Bitcoin data as 2015–2025 and reports R² = 0.9656 and RMSE approximately USD 3,751. These historical results have not been independently reproduced. Reconcile the reported dataset, architecture, hyperparameters, plots, and metrics with a recorded run before treating them as validated performance.

Check that the poster conclusion and figures agree with the final experiment results.

A high price-level R² alone does not establish performance against a last-price baseline, reliable return prediction, or a profitable trading strategy. Any such claim requires its own evaluation.

## Input validation and date integrity

`DataProcessor.load_data()` decides whether to sort by comparing only the first and last dates. This can miss disorder within the file. Always sort by parsed date, validate monotonicity, and define a policy for duplicate dates and missing calendar days.

Date parsing errors are caught and execution continues; subsequent date operations can fail. Validate required columns, date parsing, numeric conversion, finite positive prices, and a nonempty cleaned dataset, with actionable error messages.

Check that each partition has enough observations for the lookback and forecast horizon. Guard against zero validation samples: with `val_size = 0`, slicing with `[-0:]` and `[:-0]` produces an unintended full validation set and empty training set.

## Dataset documentation

Record the dataset source URL, retrieval date, actual first and last dates, currency, price definition, missing observations, and cleaning choices so the experiment input can be identified and reproduced.
