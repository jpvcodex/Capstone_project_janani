# Part 1 - Data Cleaning & EDA

For this part I used the diamonds dataset (the one from `mwaskom/seaborn-data` on
GitHub, ~54k rows - carat, cut, color, clarity, depth, table, price, x, y, z). The
real file from that source is actually pretty clean already, so to make this
assignment realistic I wrote a small step in my script that takes the clean version
and messes it up on purpose, the way an actual client export usually looks:

- `depth` - about 23% of the rows get set to null
- `table` - about 7% null
- `carat` - saved as text instead of numbers, and some rows get "unknown" or "n/a"
  typed in instead of a value
- threw in 45 duplicate rows at the end

That "messy" version is `raw_diamonds.csv` and that's what all the cleaning steps
below are actually working on. If for some reason the script can't reach the GitHub
url it just builds a fake dataset with the same columns so everything still runs,
but normally it pulls the real one.

### Task 1 - loading it in

Shape of the raw file is (53985, 10). Everything's in `run_log.txt` if you want to
see the actual head()/dtypes/shape output, but basically carat/cut/color/clarity all
come in as text and the rest are numeric.

### Task 2 - checking for nulls

| column | # null | % null |
|---|---|---|
| depth | 12416 | 23.00% |
| table | 3777 | 7.00% |
| everything else | 0 | 0% |

depth is over the 20% cutoff so I didn't fill it, I ended up just dropping it later
in task 4. table is under the cutoff so that one got filled with its median (57.0).

I used the median instead of the mean here because a couple of the numeric columns
(carat, price, y, z) are pretty skewed - if there's a handful of really big values
the mean gets dragged up by them, but the median just sits in the middle of the data
no matter what, so it's a more honest "typical" value to fill in with.

### Task 3 - duplicates

Found 131 duplicate rows and dropped them, so the dataset went from 53985 down to
53854 rows. Checked whether that changed the null percentages and it barely did
(depth moved by like 0.026%), which makes sense since the dupes were random and
not specifically the null rows.

### Task 4 - fixing data types

carat was the obvious one - it was stored as text because of those "unknown"/"n/a"
entries I put in. Once I ran `pd.to_numeric(errors='coerce')` on it, 2157 new nulls
showed up that isnull() couldn't see before (since they were technically strings,
not NaNs). Filled those with the carat median (0.700).

Also dropped depth here since it was over the 20% null threshold from task 2.

Then converted cut, color, and clarity to category dtype since they're just
repeated string labels.

Memory usage before all this: 13579.64 KB. After: 2683.33 KB. That's about an 80%
drop, mostly from the category conversion.

### Task 5 - stats & skew

Skew of each numeric column, biggest first:

| column | skew |
|---|---|
| y | 2.443 |
| price | 1.618 |
| z | 1.527 |
| carat | 1.161 |
| table | 0.868 |
| x | 0.380 |

y has the highest skew by far (2.443, positive). Positive skew means there's a long
tail out to the right - most values sit in a normal range but a few really big ones
(this dataset has a handful of clearly wrong/extreme y values) drag the tail out.
The reason this matters for imputation is if I filled missing y values with the
mean, the mean itself is already inflated by those extreme values, so I'd be
pushing the "typical" replacement value higher than it should be. Median doesn't
have that problem, which is why I used median for it in task 8a.

### Task 6 - outliers (IQR method)

| column | Q1 | Q3 | IQR | lower bound | upper bound | # outliers | % of rows |
|---|---|---|---|---|---|---|---|
| price | 951.00 | 5325.75 | 4374.75 | -5611.13 | 11887.88 | 3533 | 6.56% |
| carat | 0.400 | 1.030 | 0.630 | -0.545 | 1.975 | 2078 | 3.86% |

I didn't drop any of these. My reasoning is that expensive/large diamonds are real,
not data entry mistakes - they're just rare, so removing them would throw away
legit signal that a price model would actually need. For part 2 I'm planning to
keep them as-is, and if I end up using a linear model I'll probably log-transform
price/carat instead of capping anything, since that handles the skew without
losing information.

### Task 7 - the 5 plots

All in the `plots/` folder:

1. **line plot** (`01_line_price_sorted.png`) - price sorted ascending. It's a
   pretty smooth curve that bends up sharply near the end, which is basically the
   skew showing up visually - most diamonds are cheaper, a smaller group are a lot
   more expensive and that's what pulls the tail up.
2. **bar chart** (`02_bar_mean_price_by_cut.png`) - mean price per cut category.
   Kind of counterintuitive - Premium has the highest average price and Ideal has
   the lowest, so cut quality by itself doesn't really track with price going up.
   I think what's happening is the lower-cut categories in this data just happen to
   have bigger/pricier stones in them, so it's confounded with size.
3. **histogram** (`03_histogram_most_skewed.png`) - distribution of y (most skewed
   column, from task 5). Most of it is bunched up between roughly 3-9mm, then
   there's a thin tail stretching out to 30-60mm from a few bad/extreme rows.
   Classic right-skew shape.
4. **scatter plot** (`04_scatter_carat_price.png`) - carat vs price. Clearly
   related, but not in a straight line - price barely moves at low carat then
   curves up steeper as carat increases. Matches what I found later too: Pearson
   is 0.904 but Spearman is even higher (0.945), which fits a relationship that's
   consistent but curved rather than linear.
5. **box plot** (`05_boxplot_price_by_cut.png`) - price by cut. Medians are all
   fairly close and on the lower side across every cut, but Premium and Fair have
   noticeably wider boxes and longer whiskers than Ideal, which is the tightest and
   lowest-median group. Again points to cut not being a clean predictor on its own.

### Task 8 - correlation heatmap

(`06_correlation_heatmap.png`) Highest correlation pair is x and y at r = 0.975.
I don't think this is really telling us anything causal though - x and y are both
just physical measurements (length and width) of the same stone, so the real
reason they move together is the diamond's overall size/carat weight - a bigger
stone is bigger in every dimension at once. So the "third variable" here is
basically just carat/size itself.

### Task 8a - mean vs median for the two most skewed columns

Before doing any imputation:

| column | mean | median | skew | which one I used |
|---|---|---|---|---|
| y | 5.735 | 5.710 | 2.432 | median |
| price | 3932.989 | 2401.000 | 1.619 | median |

Both are positively skewed so in both cases the mean is higher than the median -
for price it's not even close, the mean is around 64% higher than the median
because of those high-price stones pulling it up. Went with median for both since
it's a better stand-in for what a "typical" value actually looks like. After
filling, ran isnull().sum() again and confirmed 0 nulls left in either column.

### Task 8b - Spearman vs Pearson

Top 3 pairs where Spearman and Pearson disagreed the most:

| pair | |Spearman - Pearson| |
|---|---|
| price, y | 0.097 |
| price, z | 0.096 |
| price, x | 0.079 |

In all three, Spearman comes out higher than Pearson (e.g. price/y is 0.963 vs
0.865), which tells me these relationships are monotonic but not linear - price
does consistently go up with these dimensions, just not at a constant rate. Given
that, I'm going to lean on Spearman rather than Pearson when picking features in
part 2, because Pearson is basically underselling how related these variables
actually are.

### Task 8c - grouped aggregation (price by cut)

| cut | mean | std | count |
|---|---|---|---|
| Fair | 4346.11 | 3542.08 | 1604 |
| Good | 3921.50 | 3672.85 | 4895 |
| Ideal | 3461.25 | 3810.50 | 21510 |
| Premium | 4584.37 | 4349.32 | 13769 |
| Very Good | 3981.54 | 3935.67 | 12076 |

Premium has both the highest mean ($4584.37) and highest std ($4349.32). Ratio of
the highest mean to the lowest mean (Premium vs Ideal) is 4584.37 / 3461.25 = 1.324.

Honestly the standard deviations here are bigger than the means in every single
group, which isn't a great sign - it means cut on its own isn't a reliable way to
predict price for any individual diamond, there's just too much spread within each
category. And the 1.32x ratio between highest and lowest group means isn't huge
either, so I don't think cut carries much predictive power by itself. It'll
probably need to be paired with carat or the size columns in part 2 to actually be
useful.

### Output

`cleaned_data.csv` ends up as 53854 rows, 9 columns (depth got dropped, everything
else stayed). No nulls anywhere in it - double checked with isnull().sum().

### What's in this repo

- `eda_pipeline.py` - all the code, runs start to finish on its own
- `raw_diamonds.csv` - the messy input file
- `cleaned_data.csv` - the cleaned output (this is what parts 2 and 3 will use)
- `plots/` - all 6 charts (the 5 required ones + the heatmap)
- `run_log.txt` - console output from actually running the script
- `README.md` - this file
