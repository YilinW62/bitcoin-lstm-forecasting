# Project reflection and improvement plan

This document summarizes the limitations I have identified in my Bitcoin LSTM forecasting project and the improvements I plan to make. These are areas for further work, rather than completed changes.

## Scaling leakage

My current implementation fits MinMaxScaler on all prices before splitting the data chronologically. This allows information from later periods to influence the scaling of training observations. I plan to fit the scaler on the training period only and use the same fitted scaler to transform validation and test data.

## Model selection

In `src/grid_search.py`, I currently rank configurations using test metrics and recommend a configuration based on test R². I plan to use validation metrics for parameter selection and reserve the test set for the final evaluation.

## Overlapping forecast targets

In `src/train.py` and `src/grid_search.py`, I create rolling windows before splitting training and validation samples. Adjacent windows can share target dates across the split. I plan to split by target dates or remove boundary-crossing samples so that training and validation targets remain separate.

## Checkpoint handling

My grid-search trainer records the best validation loss without restoring the corresponding model weights. In the main trainer, I restore the best checkpoint when early stopping occurs, but not when training finishes normally. I plan to restore the best validation checkpoint consistently before evaluation.

## Baseline and forecast horizons

My current evaluation pools predictions across overlapping windows and forecast horizons. I plan to report separate metrics for day 1, day 2, and day 3, and compare my LSTM against a persistence baseline that repeats the last observed price. I will evaluate both models using the same forecast origins.

## Visualization and interpretation

Some forecasts in `src/visualize_data.py` use trend extrapolation and random noise. I need to distinguish these illustrations from predictions produced by my trained model.

My direct multi-output approach avoids feeding predicted prices back into the model, but this alone does not establish better forecasting performance. I plan to assess its performance through comparative evaluation.

I also need to distinguish the rolling standard deviation of price levels from return volatility. I plan to use the appropriate label or calculate volatility from returns.

## Reproducibility

I plan to set random seeds before model initialization and record the settings used for each experiment. I also need to check date ordering, duplicate dates, missing days, and the alignment of input sequences with their targets.

## Poster and reported results

My poster describes the dataset as covering 2015–2025 and reports R² = 0.9656 and RMSE of approximately USD 3,751. I still need to reproduce these results under the revised evaluation procedure and link each reported metric and figure to the corresponding dataset, architecture, and training parameters.

I plan to update the poster's figures and conclusion to reflect the final experiment results. I also recognize that a high price-level R² alone does not demonstrate improvement over a last-price baseline, reliable return prediction, or trading profitability.

## Input validation and date integrity

My current date-sorting check compares only the first and last dates, which can miss disorder within the dataset. I plan to sort all observations by parsed date and define how to handle duplicate dates and missing calendar days.

The data-loading code currently catches date parsing errors and continues. I need to validate required columns, dates, numeric prices, and the cleaned dataset before training, and provide clear error messages for invalid inputs.

I also need to check that each partition contains enough observations for the lookback window and forecast horizon. In particular, I plan to handle a zero-sized validation split explicitly, since slicing with `[-0:]` and `[:-0]` can produce a full validation set and an empty training set.

## Dataset documentation

I plan to document the dataset source URL, retrieval date, actual date range, currency, price definition, missing observations, and cleaning choices so that the input to my experiments can be identified and reproduced.
