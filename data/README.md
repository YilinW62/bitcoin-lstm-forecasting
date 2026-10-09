# Data

The project uses daily Bitcoin prices in Excel format. The poster identifies Investing.com as the data source.

## Expected input

- `Date`: a date column.
- `Price` or `Close`: daily price, numeric or comma-separated numeric text.
- Other columns are not model inputs in the current implementation.

## Placement and use

Save the Excel dataset as `data/bitcoin_daily.xlsx` and run from the repository root:

```bash
python train.py --data_file "data/bitcoin_daily.xlsx" --device cpu
python grid_search.py --data_file "data/bitcoin_daily.xlsx" --device cpu --quick
```

The scripts accept an explicit input path through `--data_file`. For `visualize_data.py`, place the dataset at `Bitcoin_Historical_Data_10years.xlsx` in the working directory or edit the input path in the script.
