import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(HERE, "plots")
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
SEED = 42
NULL_THRESHOLD = 20.0  # percent


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Step 0: obtain a base (clean) copy of the public dataset, online or offline
# ---------------------------------------------------------------------------
def load_base_dataset() -> pd.DataFrame:
    try:
        df = pd.read_csv(DATA_URL)
        print(f"Loaded live public dataset from {DATA_URL} ({len(df)} rows).")
    except Exception as e:
        print(f"Could not reach {DATA_URL} ({e}). Using local synthetic fallback "
              f"with the identical schema so the pipeline still runs end-to-end.")
        df = _synthetic_diamonds()
    # keep only the canonical columns / order
    cols = ["carat", "cut", "color", "clarity", "depth", "table", "price", "x", "y", "z"]
    return df[cols].reset_index(drop=True)


def _synthetic_diamonds(n=8000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    cut = rng.choice(["Fair", "Good", "Very Good", "Premium", "Ideal"], size=n,
                      p=[0.03, 0.09, 0.22, 0.26, 0.40])
    color = rng.choice(list("DEFGHIJ"), size=n)
    clarity = rng.choice(["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], size=n)
    carat = np.round(rng.gamma(shape=2.0, scale=0.35, size=n), 2)
    depth = np.round(rng.normal(61.7, 1.4, size=n), 1)
    table = np.round(rng.normal(57.5, 2.2, size=n), 1)
    cut_bonus = pd.Series(cut).map({"Fair": 0, "Good": 200, "Very Good": 400,
                                     "Premium": 500, "Ideal": 600}).to_numpy()
    price = (carat ** 1.8) * 4500 + cut_bonus + rng.normal(0, 300, size=n)
    price = np.clip(price, 300, None).round(0)
    x = np.round(carat ** (1 / 3) * 5.5 + rng.normal(0, 0.05, size=n), 2)
    y = np.round(x + rng.normal(0, 0.03, size=n), 2)
    z = np.round(x * 0.61 + rng.normal(0, 0.03, size=n), 2)
    # a small number of physically-impossible extreme outliers, mirroring the
    # well-documented erroneous rows (y/z ~ 30-60mm) present in the real
    # public diamonds dataset
    bad_idx = rng.choice(n, size=3, replace=False)
    y[bad_idx] = rng.uniform(30, 59, size=3)
    z[bad_idx] = rng.uniform(25, 32, size=3)
    return pd.DataFrame({"carat": carat, "cut": cut, "color": color, "clarity": clarity,
                          "depth": depth, "table": table, "price": price,
                          "x": x, "y": y, "z": z})


# ---------------------------------------------------------------------------
# Step 0b: simulate the "raw file from the client" by injecting documented
# data-quality issues into the clean public dataset
# ---------------------------------------------------------------------------
def make_raw_file(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    df = df.copy()
    n = len(df)

    # (a) depth: ~23% missing -> will exceed the 20% threshold
    depth_na_idx = rng.choice(n, size=int(0.23 * n), replace=False)
    df.loc[depth_na_idx, "depth"] = np.nan

    # (b) table: ~7% missing -> below the 20% threshold, median-fillable
    table_na_idx = rng.choice(n, size=int(0.07 * n), replace=False)
    df.loc[table_na_idx, "table"] = np.nan

    # (c) carat: mis-typed on purpose. Store as string, and for ~4% of rows
    #     insert non-numeric placeholder tokens a data-entry system might use
    #     for unknown weight ("unknown", "n/a"). This is realistic "hidden"
    #     missingness: it will NOT show up in isnull().sum() until the dtype
    #     is corrected in Task 4.
    carat_str = df["carat"].astype(str)
    bad_carat_idx = rng.choice(n, size=int(0.04 * n), replace=False)
    token = rng.choice(["unknown", "n/a"], size=len(bad_carat_idx))
    carat_str.iloc[bad_carat_idx] = token
    df["carat"] = carat_str  # now dtype = object, a genuinely wrong dtype

    # (d) duplicate rows: append 45 exact duplicates of random existing rows
    dupes = df.sample(45, random_state=SEED)
    df = pd.concat([df, dupes], ignore_index=True)

    return df


# ===========================================================================
# TASK 1: load, first 5 rows, dtypes, shape
# ===========================================================================
def task1(df):
    hr("TASK 1: Load raw data")
    print("First 5 rows:")
    print(df.head())
    print("\nColumn dtypes:")
    print(df.dtypes)
    print(f"\nDataFrame shape: {df.shape}")


# ===========================================================================
# TASK 2: null value analysis + median fill for columns below threshold
# ===========================================================================
def task2(df):
    hr("TASK 2: Null value analysis")
    null_counts = df.isnull().sum()
    null_pct = (df.isnull().sum() / df.shape[0]) * 100
    null_table = pd.DataFrame({"null_count": null_counts, "null_pct": null_pct.round(2)})
    print(null_table)

    high_null_cols = null_table[null_table["null_pct"] > NULL_THRESHOLD].index.tolist()
    print(f"\nColumns exceeding {NULL_THRESHOLD}% missing: {high_null_cols}")

    fill_cols = []
    for col in df.select_dtypes(include=np.number).columns:
        pct = null_pct[col]
        if 0 < pct <= NULL_THRESHOLD:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            fill_cols.append(col)
            print(f"Filled '{col}' ({pct:.2f}% missing) with median = {median_val:.3f}")

    return df, high_null_cols, null_table


# ===========================================================================
# TASK 3: duplicate detection and removal
# ===========================================================================
def task3(df):
    hr("TASK 3: Duplicate detection")
    n_dupes = df.duplicated().sum()
    print(f"Duplicate rows found: {n_dupes}")

    null_pct_before = (df.isnull().sum() / df.shape[0]) * 100
    df = df.drop_duplicates().reset_index(drop=True)
    null_pct_after = (df.isnull().sum() / df.shape[0]) * 100

    print(f"Rows removed: {n_dupes}. New shape: {df.shape}")
    change = (null_pct_after - null_pct_before).round(3)
    print("\nChange in null % per column after de-duplication:")
    print(change)
    return df


# ===========================================================================
# TASK 4: dtype correction (carat -> numeric) + category conversion (cut)
# ===========================================================================
def task4(df, high_null_cols):
    hr("TASK 4: Data type correction")
    mem_before = df.memory_usage(deep=True).sum()
    print(f"Memory usage before correction: {mem_before / 1024:.2f} KB")

    # carat was intentionally stored as object with 'unknown'/'n/a' tokens
    print(f"\n'carat' dtype before correction: {df['carat'].dtype}")
    df["carat"] = pd.to_numeric(df["carat"], errors="coerce")
    new_nulls = df["carat"].isnull().sum()
    print(f"'carat' dtype after correction:  {df['carat'].dtype}")
    print(f"New nulls surfaced by coercion (previously hidden as text tokens): {new_nulls}")
    carat_median = df["carat"].median()
    df["carat"] = df["carat"].fillna(carat_median)
    print(f"Filled these with the median carat = {carat_median:.3f}")

    # drop the column that exceeded the 20% null threshold (documented decision)
    if high_null_cols:
        print(f"\nDropping column(s) exceeding {NULL_THRESHOLD}% missing: {high_null_cols}")
        df = df.drop(columns=high_null_cols)

    # repetitive string column -> category dtype
    df["cut"] = df["cut"].astype("category")
    df["color"] = df["color"].astype("category")
    df["clarity"] = df["clarity"].astype("category")

    mem_after = df.memory_usage(deep=True).sum()
    print(f"\nMemory usage after correction (incl. category conversion): {mem_after / 1024:.2f} KB")
    print(f"Memory saved: {(mem_before - mem_after) / 1024:.2f} KB "
          f"({100 * (mem_before - mem_after) / mem_before:.1f}%)")
    return df


# ===========================================================================
# TASK 5: descriptive stats + skewness
# ===========================================================================
def task5(df):
    hr("TASK 5: Descriptive statistics & skewness")
    print(df.describe().T)

    num_cols = df.select_dtypes(include=np.number).columns
    skew = df[num_cols].skew().sort_values(key=np.abs, ascending=False)
    print("\nSkewness (sorted by absolute value, descending):")
    print(skew)

    top2 = skew.index[:2].tolist()
    print(f"\nMost skewed column: '{skew.index[0]}' (skew = {skew.iloc[0]:.3f})")
    print(f"Top-2 skewed columns for Task 8a: {top2}")
    return skew, top2


# ===========================================================================
# TASK 6: IQR outlier detection (no dropping)
# ===========================================================================
def task6(df, cols=("price", "carat")):
    hr("TASK 6: IQR outlier detection")
    results = {}
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        pct_out = 100 * n_out / len(df)
        results[col] = dict(Q1=q1, Q3=q3, IQR=iqr, lower=lower, upper=upper,
                             n_outliers=n_out, pct_outliers=pct_out)
        print(f"\n{col}: Q1={q1:.3f}, Q3={q3:.3f}, IQR={iqr:.3f}, "
              f"bounds=({lower:.3f}, {upper:.3f})")
        print(f"  Outliers outside bounds: {n_out} ({pct_out:.2f}% of rows) - retained, not dropped.")
    return results


# ===========================================================================
# TASK 7: five required visualizations
# ===========================================================================
def task7(df):
    hr("TASK 7: Visualizations")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Line plot - one numeric variable sorted by row index
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(df)), df["price"].sort_values().values, color="steelblue")
    plt.title("Line Plot: Diamond Price, Sorted Ascending by Row Index")
    plt.xlabel("Row index (after sorting)")
    plt.ylabel("Price (USD)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "01_line_price_sorted.png"), dpi=150)
    plt.close()

    # 2. Bar chart - mean of numeric column across categories
    plt.figure(figsize=(8, 5))
    means = df.groupby("cut", observed=True)["price"].mean().sort_values()
    plt.bar(means.index.astype(str), means.values, color="darkorange")
    plt.title("Bar Chart: Mean Price by Cut Quality")
    plt.xlabel("Cut")
    plt.ylabel("Mean Price (USD)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "02_bar_mean_price_by_cut.png"), dpi=150)
    plt.close()

    # 3. Histogram of most skewed numeric column
    num_cols = df.select_dtypes(include=np.number).columns
    most_skewed_col = df[num_cols].skew().abs().idxmax()
    plt.figure(figsize=(8, 5))
    sns.histplot(df[most_skewed_col], bins=20, color="seagreen", kde=True)
    plt.title(f"Histogram: Distribution of '{most_skewed_col}' (most skewed column)")
    plt.xlabel(most_skewed_col)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "03_histogram_most_skewed.png"), dpi=150)
    plt.close()

    # 4. Scatter plot between two numeric columns expected to correlate
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df.sample(min(3000, len(df)), random_state=SEED),
                     x="carat", y="price", alpha=0.4, s=15, color="indigo")
    plt.title("Scatter Plot: Carat vs Price")
    plt.xlabel("Carat")
    plt.ylabel("Price (USD)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "04_scatter_carat_price.png"), dpi=150)
    plt.close()

    # 5. Box plot of numeric column split by categorical column
    plt.figure(figsize=(8, 5))
    order = [c for c in ["Fair", "Good", "Very Good", "Premium", "Ideal"] if c in df["cut"].unique()]
    sns.boxplot(data=df, x="cut", y="price", order=order if order else None)
    plt.title("Box Plot: Price Distribution by Cut")
    plt.xlabel("Cut")
    plt.ylabel("Price (USD)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "05_boxplot_price_by_cut.png"), dpi=150)
    plt.close()

    print(f"Saved 5 required plots to {PLOTS_DIR}")
    return most_skewed_col


# ===========================================================================
# TASK 8: correlation heatmap (Pearson)
# ===========================================================================
def task8(df):
    hr("TASK 8: Correlation heat map (Pearson)")
    num_cols = df.select_dtypes(include=np.number).columns
    corr = df[num_cols].corr(method="pearson")
    print(corr.round(3))

    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heat Map (Pearson, Numeric Features)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "06_correlation_heatmap.png"), dpi=150)
    plt.close()

    # find highest |correlation| pair, excluding the diagonal
    arr = corr.abs().to_numpy(copy=True)
    np.fill_diagonal(arr, 0)
    corr_abs = pd.DataFrame(arr, index=corr.index, columns=corr.columns)
    max_pair = corr_abs.stack().idxmax()
    max_val = corr.loc[max_pair]
    print(f"\nHighest absolute correlation pair: {max_pair} -> r = {max_val:.3f}")
    return corr, max_pair, max_val


# ===========================================================================
# TASK 8a: mean vs median for the two highest-skew columns, then impute
# ===========================================================================
def task8a(df, raw_df, top2_cols):
    hr("TASK 8a: Imputation strategy comparison (mean vs median)")
    for col in top2_cols:
        # reconstruct the column's pre-imputation values from the raw file
        if col == "carat":
            pre = pd.to_numeric(raw_df["carat"], errors="coerce")
        else:
            pre = raw_df[col]
        mean_val, median_val = pre.mean(), pre.median()
        print(f"\nColumn '{col}' BEFORE imputation -> mean = {mean_val:.3f}, median = {median_val:.3f}")
        skew_val = pre.skew()
        chosen = "median" if abs(skew_val) > 0.5 else "mean"
        print(f"  Skewness = {skew_val:.3f} -> chosen statistic: {chosen}")
        fill_val = median_val if chosen == "median" else mean_val
        remaining_na = df[col].isnull().sum()
        if remaining_na > 0:
            df[col] = df[col].fillna(fill_val)
        print(f"  Nulls remaining in '{col}' after imputation: {df[col].isnull().sum()}")
    return df


# ===========================================================================
# TASK 8b: Spearman vs Pearson comparison
# ===========================================================================
def task8b(df, pearson_corr):
    hr("TASK 8b: Spearman rank correlation vs Pearson")
    num_cols = df.select_dtypes(include=np.number).columns
    spearman_corr = df[num_cols].corr(method="spearman")
    print("Spearman correlation matrix:")
    print(spearman_corr.round(3))

    diff_arr = (spearman_corr - pearson_corr).abs().to_numpy(copy=True)
    np.fill_diagonal(diff_arr, 0)
    diff = pd.DataFrame(diff_arr, index=spearman_corr.index, columns=spearman_corr.columns)
    stacked = diff.stack()
    # drop symmetric duplicate pairs (a,b) vs (b,a)
    stacked = stacked[~stacked.index.map(frozenset).duplicated()]
    top3 = stacked.sort_values(ascending=False).head(3)
    print("\nTop 3 pairs by |Spearman - Pearson|:")
    print(top3.round(3))
    return spearman_corr, top3


# ===========================================================================
# TASK 8c: grouped aggregation
# ===========================================================================
def task8c(df, cat_col="cut", num_col="price"):
    hr("TASK 8c: Grouped aggregation")
    agg = df.groupby(cat_col, observed=True)[num_col].agg(["mean", "std", "count"])
    print(agg)

    highest_mean_group = agg["mean"].idxmax()
    highest_std_group = agg["std"].idxmax()
    ratio = agg["mean"].max() / agg["mean"].min()
    print(f"\nHighest mean group: {highest_mean_group}")
    print(f"Highest std group:  {highest_std_group}")
    print(f"Ratio of highest to lowest group mean: {ratio:.3f}")
    return agg, highest_mean_group, highest_std_group, ratio


def main():
    base = load_base_dataset()
    raw = make_raw_file(base)
    raw.to_csv(os.path.join(HERE, "raw_diamonds.csv"), index=False)

    df = raw.copy()
    task1(df)
    df, high_null_cols, _ = task2(df)
    df = task3(df)
    df = task4(df, high_null_cols)
    skew, top2 = task5(df)
    task6(df, cols=("price", "carat"))
    task7(df)
    pearson_corr, max_pair, max_val = task8(df)
    df = task8a(df, raw, top2)
    task8b(df, pearson_corr)
    task8c(df, cat_col="cut", num_col="price")

    hr("FINAL: save cleaned dataset")
    print(f"Final null counts:\n{df.isnull().sum()}")
    out_path = os.path.join(HERE, "cleaned_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved cleaned dataset to {out_path} - shape {df.shape}")


if __name__ == "__main__":
    main()