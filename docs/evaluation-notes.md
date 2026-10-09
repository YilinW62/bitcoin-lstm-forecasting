# Evaluation review

This document records issues found by inspecting the original source. It does not claim that they have already been corrected.

## Scaling leakage

`DataProcessor.normalize_data()` fits MinMaxScaler on all prices before temporal splitting. Fit on the training period only, and transform validation/test data using that fitted scaler.

## Model selection

`grid_search.py` ranks configurations using test metrics and uses test R² to recommend a configuration. Use validation metrics for selection, then evaluate the chosen configuration on held-out test data.

## Overlapping forecast targets

The multi-step training entry point constructs windows before splitting training and validation samples. Adjacent windows share target dates. Split by target dates or purge boundary-crossing targets so that a training label cannot also be a validation label.

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
