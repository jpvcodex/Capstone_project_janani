# Part 1 — Data Cleaning & EDA

## What this does

`part_1.py` pulls the public diamonds dataset (from the seaborn-data repo on
GitHub — if there's no internet it falls back to a synthetic dataset with the
same columns, so the script still runs). I deliberately messed up the data
first to make it look like a "raw client file" someone actually handed me,
then cleaned it up and did EDA on it.

Rough order of what happens:

- Load the raw data and take a first look (head, dtypes, shape).
- Check nulls per column. Anything over 20% missing gets dropped, anything
  under gets median-filled.
- Find and drop exact duplicate rows.
- Fix a column I intentionally broke — it had `"unknown"` / `"n/a"` text
  hidden in it so it read as `object` dtype instead of numeric. Coerce it
  back to numeric, impute the nulls that surfaces, and drop whichever column
  ends up over the 20% missing threshold. Also convert the repetitive string
  columns (`cut`, `color`, `clarity`) to `category` dtype to save memory.
- `describe()` + skewness per column, sorted by how skewed they are.
- IQR bounds on `price` and `carat` to flag outliers (I report these, don't
  drop them).
- Save 5 required plots to `plots/`:
  - `01_line_price_sorted.png`
  - `02_bar_mean_price_by_cut.png`
  - `03_histogram_most_skewed.png`
  - `04_scatter_carat_price.png`
  - `05_boxplot_price_by_cut.png`
- Correlation heatmap (`06_correlation_heatmap.png`) and calling out the
  strongest correlated pair.
- Compare mean vs. median imputation on the two most skewed columns and pick
  whichever makes more sense given the skew.
- Spearman vs. Pearson correlation, and the top 3 pairs where they disagree
  most.
- Group by `cut` and get mean/std/count of `price`.

Cleaned dataframe gets saved as `cleaned_data.csv` in this folder.

## Running it

```bash
pip install pandas numpy matplotlib seaborn
python part_1.py
```

No API keys needed for this part. It tries to download the dataset first;
if that fails it just generates a synthetic version with the same schema so
everything downstream still works.

Outputs: `cleaned_data.csv` (needed for Parts 2 and 3), `raw_diamonds.csv`
(the messy version, kept for reference), and the plots in `plots/`.

## Findings

- `depth` was missing ~23% of its values, so that one got dropped (over the
  20% cutoff). `table` was only missing ~7%, so it got median-filled instead.
- `carat` looked totally fine in `isnull().sum()` at first, but it was
  actually stored as text with `"unknown"`/`"n/a"` hidden in it. Once I
  coerced it to numeric, ~2,157 more nulls showed up that weren't visible
  before.
- 131 exact duplicate rows, removed.
- Switching `cut`/`color`/`clarity` to `category` dtype cut memory usage by
  about 80%, which was a bigger difference than I expected.
- `y` was the most skewed column (skew ≈ 2.44), `price` right behind it
  (≈1.62). Used median for both since mean gets pulled around too much when
  there's this much skew.
- `x` and `y` had the strongest correlation (r ≈ 0.975) — makes sense, they're
  both just diamond dimensions.
- IQR flagged 6.56% of rows as price outliers and 3.86% as carat outliers.
  Left them in on purpose — for diamonds, extreme values are usually real,
  not data entry errors.

