# Data

The original project uses a daily Bitcoin Excel dataset and identifies Investing.com as its source in the poster. The stated coverage is 2015–2025; verify exact first and last dates from the dataset before reporting an experiment.

The dataset is not distributed here because its redistribution terms have not been confirmed.

## Expected input

- `Date`: a date column.
- `Price` or `Close`: daily price, numeric or comma-separated numeric text.
- Other columns are not model inputs in the current implementation.

Obtain the historical daily data from the source, save an Excel copy as `data/bitcoin_daily.xlsx`, and pass its path using `--data_file`.

Record the source URL, download date, actual date range, missing observations, and price definition when preparing the final release. These details have not yet been verified.

The originally uploaded filename contains spaces, while the scripts' default filename contains underscores. Passing the explicit path avoids this mismatch.
