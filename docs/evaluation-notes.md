# Known issues and improvement plan

This document records issues found by inspecting the original source. It does not claim that they have already been corrected.

## Scaling leakage

`DataProcessor.normalize_data()` fits MinMaxScaler on all prices before temporal splitting. Fit on the training period only, and transform validation/test data using that fitted scaler.

## Model selection

`grid_search.py` ranks configurations using test metrics and uses test R² to recommend a configuration. Use validation metrics for selection, then evaluate the chosen configuration on held-out test data.

## Overlapping forecast targets

Both `train.py` and `grid_search.py` construct windows before splitting training and validation samples. Adjacent windows share target dates. Split by target dates or purge boundary-crossing targets so that a training label cannot also be a validation label.

## Checkpoint handling

The grid-search trainer tracks the best validation loss but does not restore the corresponding weights. The main trainer restores its checkpoint in the early-stopping branch; normal completion also needs to restore the best checkpoint.

## Baseline and horizons

Add a persistence baseline that repeats the last observed price for every horizon. Evaluate both models on identical forecast origins, with separate day-1, day-2, and day-3 scores. Current pooled metrics flatten overlapping windows and horizons.

## Visualization and claims

Some forecasts in `visualize_data.py` are simulated with trend extrapolation and random noise. They are illustrations, not trained-model outputs.

Direct multi-output prediction avoids recursive feedback of predicted prices; it does not eliminate forecast errors or establish superior performance.

The rolling standard deviation of price levels is not the same measure as return volatility. Label it accurately or calculate volatility from returns.

## Reproducibility

Seed randomness before model initialization, save run configurations and dependency versions, and validate date sorting, duplicates, missing days, and sequence alignment.

## Validation performed for this import

All six Python files passed AST syntax parsing. Training, dependency installation, and numerical performance were not tested.

## Poster and historical results

The supplied poster describes Bitcoin data as 2015–2025 and reports R² = 0.9656 and RMSE approximately USD 3,751. These historical results have not been independently reproduced. Reconcile the reported dataset, architecture, hyperparameters, plots, and metrics with a recorded run before treating them as validated performance.

Review the poster for remaining draft or placeholder text and complete its conclusion where needed. Do not infer these edits have been completed from the PDF upload.

A high price-level R² alone does not establish performance against a last-price baseline, reliable return prediction, or a profitable trading strategy. Any such claim requires its own evaluation.

## Input validation and date integrity

`DataProcessor.load_data()` decides whether to sort by comparing only the first and last dates. This can miss disorder within the file. Always sort by parsed date, validate monotonicity, and define a policy for duplicate dates and missing calendar days.

Date parsing errors are caught and execution continues; subsequent date operations can fail. Validate required columns, date parsing, numeric conversion, finite positive prices, and a nonempty cleaned dataset, with actionable error messages.

Check that each partition has enough observations for the lookback and forecast horizon. Guard against zero validation samples: with `val_size = 0`, slicing with `[-0:]` and `[:-0]` produces an unintended full validation set and empty training set.

## Configuration and command consistency

Defaults in `config.py` and `train.py` differ; document and unify the intended experiment configuration. Command-line values currently take precedence in the training entry point.

The original dataset filename contains spaces and Chinese text; the scripts use `Bitcoin_Historical_Data_10years.xlsx`. README examples use an explicit `data/bitcoin_daily.xlsx` path. Standardize the input path across all entry points, including the visualization script.

The grid-search recommendation command omits the dataset path, device, epochs, batch size, and dropout. `train.py` does not expose a dropout argument. Export a complete configuration and make it usable by the training entry point.

## Dataset provenance and redistribution

The dataset is not included in this repository. Redistribution terms have not been confirmed. Confirm permission before uploading it, and record the source URL, retrieval date, actual first and last dates, currency, price definition, missing observations, and cleaning choices.

## Experiment artifacts and environment

The repository contains source and the poster, but no recorded experiment outputs, trained checkpoints, or reproducibility manifest. After correcting evaluation, save authentic prediction figures, forecast dates, per-horizon metrics, baseline results, split boundaries, configuration, seed, and model/scaler artifacts where appropriate.

`requirements.txt` lists dependencies without pinned versions. Record the Python version and lock a tested environment. A full dependency installation and end-to-end training run remain unverified.

Add meaningful automated tests for training-only scaler fitting, chronological split boundaries, disjoint target dates, sequence shapes and date alignment, checkpoint restoration, and invalid or insufficient input data.

## Attribution and release readiness

No open-source license has been selected. Choose one for the original code if intended, and verify third-party data and diagram attribution separately.

The repository is currently private. For an RA application, arrange reviewer access or make the repository public after reviewing what will be shared. Visibility changes are not part of this documentation update.

## Suggested implementation order

1. Fix scaling, validation selection, and overlapping targets.
2. Restore best checkpoints and validate dates and input sizes.
3. Unify configuration, seed initialization, and dependency versions.
4. Add the persistence baseline and per-horizon metrics, then rerun experiments.
5. Save authentic results and update the poster to match the recorded run.
6. Complete dataset provenance, attribution, license selection, and reviewer access.

## Documentation convention

Keep issue descriptions, unverified-result qualifications, and improvement tasks in this document. The root README, data guide, and poster guide describe the project, usage, and available materials.
