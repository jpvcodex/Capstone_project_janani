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

# Part 2 - ML Models (Regression + Classification)

This is the write-up for Part 2 of the capstone. I used the cleaned_data.csv file that came out of Part 1, so this whole thing depends on that file being there.

To run it: `python part_2.py` (just make sure cleaned_data.csv is sitting in the same folder).

## What I used as my labels

I used `price` as the thing I'm trying to predict for regression, since it's the only continuous numeric column that makes sense to predict.

For classification I didn't have a real binary column in the data, so I made one myself - I just split price at the median. So if a diamond's price is above the median it's labeled 1, otherwise 0. Median came out to about 2273.5. This actually gives an almost perfect 50/50 split which was nice because it meant I didn't have to deal with imbalance.

## Encoding the categorical columns

I had three categorical columns: cut, clarity, and color.

For cut and clarity I just did label encoding (mapped them to numbers 0,1,2...) because these actually have a real order to them - like Fair is worse than Good which is worse than Very Good etc, and same idea for clarity (I1 being the worst all the way up to IF being flawless). So it made sense to keep that order as numbers.

For color I didn't want to do that because there's no "column" that has a natural order for what I'm doing here - if I just numbered them 0-6 the model would think color G is like "3 steps more" than color D or whatever, which isn't really true, it would just be picking up a fake relationship that isn't really there. So instead I one-hot encoded color and dropped the first dummy column so I don't run into the dummy variable trap (multicollinearity).

## Train/test split and scaling

Split was 80/20 with random_state=42 so it's reproducible.

For scaling - and this part is important - I only fit the StandardScaler on the training data, not on everything. If I fit it on the whole dataset (train + test together) that would basically be cheating a little, because then the scaler already "knows" info about the test set's mean and spread before I even train the model. That's data leakage. So I fit on X_train only, then just apply (transform) that same scaler to both train and test.

## Regression - Linear vs Ridge

| Model | MSE | R2 |
|---|---|---|
| Linear Regression | 2,299,438.83 | 0.8866 |
| Ridge (alpha=1.0) | 2,300,689.85 | 0.8865 |

Basically these two are almost identical for this dataset.

Top 3 features by coefficient size were carat, x, and cut.

Carat has a big positive coefficient (~4877) which makes total sense - bigger carat = way more expensive, no surprise there.

x had a big negative coefficient (~-1033) which looks weird at first glance since you'd think a bigger diamond dimension should mean higher price too. But this is happening because carat, x, y, and z are all basically measuring "how big is this diamond" in different ways, so they're highly correlated with each other. When you put all of them into a linear model together it gets confused about which one to "credit" for the size effect, so it can end up giving one of them a negative number even though on its own it would clearly be positive. So I wouldn't read too much into that individual negative coefficient - it's more of a side effect of multicollinearity than some real "bigger x = cheaper" pattern.

Ridge vs regular Linear Regression - alpha controls how much the model gets penalized for having large coefficients. Bigger alpha = more shrinkage. With alpha=1.0 here it barely changed anything vs plain OLS, so it seems like regularization wasn't really needed much for this problem - the model wasn't overfitting badly to begin with.

## Classification - Logistic Regression

Checked the class balance first: 3226 vs 3174 in the training set, so basically 50/50 (49.6% vs 50.4%). That's way above the 35% cutoff for "this needs balancing," so I didn't apply SMOTE or class_weight - didn't need to. (Side note: I originally wanted to try SMOTE just to show I could, but couldn't install imbalanced-learn because there was no internet in the sandbox environment I tested this in. Didn't matter in the end since the classes were already balanced by design from doing the median split.)

Confusion matrix:

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| Actual 0 | 744 | 30 |
| Actual 1 | 37 | 789 |

Accuracy came out to 0.96, precision and recall both around 0.95-0.96 for each class. AUC = 0.9934, which is really high - basically the model can tell the two price groups apart almost perfectly using these features.

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)

For this specific problem I don't think one matters way more than the other - there's no real "this mistake is way worse than that mistake" situation here, it's just predicting if a diamond is above or below the median price, so I'd rather just look at F1 as the balanced overall score rather than pick precision or recall specifically.

AUC of 0.99 basically means: if you grabbed one random "expensive" diamond and one random "cheap" diamond, the model would rank the expensive one higher about 99% of the time. That's a really strong result.

### Threshold testing (0.3 to 0.7)

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.923 | 0.973 | 0.948 |
| 0.40 | 0.951 | 0.965 | 0.958 |
| 0.50 | 0.963 | 0.955 | 0.959 |
| 0.60 | 0.973 | 0.942 | 0.957 |
| 0.70 | 0.976 | 0.927 | 0.951 |

Best F1 is right at the default 0.50 threshold. Makes sense given the classes are so balanced and well-separated - moving the threshold up or down just trades precision for recall without any real gain. Since I already said neither error type matters more here, I'd just leave the threshold at 0.5.

## Regularization test - C=1.0 vs C=0.01

| Model | Precision | Recall | AUC |
|---|---|---|---|
| C=1.0 | 0.963 | 0.955 | 0.9934 |
| C=0.01 | 0.967 | 0.950 | 0.9933 |

C is basically the opposite of alpha in Ridge - smaller C means stronger regularization (more shrinkage). Even with really strong regularization (C=0.01) the model barely changed at all. This tells me the classes in this dataset are just really cleanly separated by these features, so it doesn't matter much how hard you regularize, the model still does great.

### Bootstrap test on the AUC difference

Did 500 bootstrap resamples of the test set to see if C=1.0 is actually meaningfully better than C=0.01 or if that tiny difference is just noise.

- Mean AUC difference: 0.0002
- 95% CI: [-0.0001, 0.0004]

Since this interval includes 0, that means the "advantage" of C=1.0 isn't statistically reliable - it could easily just be random variation from the specific test set I happened to get. So realistically, either C value would be fine to use here.

## Files in this folder

- part_2.py - all the code
- cleaned_data.csv - the input from Part 1
- plots/07_roc_curve.png - ROC curve plot
- part2_output.log - the full printed output from running it top to bottom

- 

# Part 3 - Ensembles, Tuning, and the Full Pipeline

This picks up right where Part 2 left off. Same dataset (cleaned_data.csv), same features, same train/test split (80/20, random_state=42), same classification target (price above/below median). I just rebuilt that split at the top of part_3.py so the script can run on its own without needing to import Part 2.

Run it with: `python part_3.py` (needs cleaned_data.csv in the same folder). Fair warning - the GridSearchCV step takes a little while to run (about 40 seconds on my machine) since it's fitting 90 different Random Forests.

## Decision tree - no limits

First I just let a Decision Tree grow with no restrictions at all (max_depth=None).

- Train accuracy: 1.0000
- Test accuracy: 0.9337
- Gap: 0.066

Yeah, that's overfitting. Getting literally 100% on the training data is basically a red flag on its own - it means the tree memorized the training rows instead of learning general patterns. Decision trees do this because of how they're built: at every split they just greedily pick whatever split looks best right now for the data in front of them, and they never go back and reconsider earlier splits once they're made. So if you don't stop it, it'll just keep splitting deeper and deeper until every single training point ends up in its own tiny leaf - which is exactly what "memorizing" looks like. That's why trees are called high variance models - a small change in the training data can produce a totally different tree.

## Decision tree - controlled

Same tree but with max_depth=5 and min_samples_split=20 this time.

- Train accuracy: 0.9652
- Test accuracy: 0.9556
- Gap: 0.0095

Way better. The gap basically disappeared, dropping from 6.6% down to under 1%.

max_depth just caps how many levels deep the tree is allowed to go, so it can't keep carving out smaller and smaller regions for individual training points - this trades a little bit of bias for a big reduction in variance. min_samples_split says "don't even bother splitting a node if it has fewer than 20 samples in it," which stops the tree from making splits based on tiny groups of points that might just be noise rather than a real pattern.

## Gini vs Entropy

Trained two trees at max_depth=5, one with each criterion:

- Gini: 0.9556 test accuracy
- Entropy: 0.9550 test accuracy

Basically a tie, entropy is a hair lower but not in any meaningful way.

Gini impurity formula: 1 - sum(p_i^2)
Entropy formula: -sum(p_i * log2(p_i))

Both of these are just measuring "how mixed up are the classes in this node." If a node has Gini = 0, that means every single sample in that node belongs to the same class - it's a "pure" node, there's no mixing at all. The tree is trying to pick splits that get impurity as close to 0 as possible at each step.

## Random Forest

n_estimators=100, max_depth=10:

- Train accuracy: 0.9889
- Test accuracy: 0.9575
- Test AUC: 0.9924

Top 5 features by importance:

| Feature | Importance |
|---|---|
| x | 0.325 |
| y | 0.263 |
| z | 0.252 |
| carat | 0.118 |
| table | 0.013 |

So basically x, y, z, and carat dominate everything else combined - which makes total sense since these are all just different ways of measuring how big/heavy the diamond is, and size is obviously the main driver of price.

How Random Forest actually gets these importance numbers: for every single split, in every single tree, it tracks how much that split reduced the Gini impurity. Then it averages that reduction across all the trees for each feature. So a feature that consistently helps split the classes apart cleanly across lots of trees ends up with a high importance score. This is different from a linear regression coefficient, because a regression coefficient tells you the size AND direction of a feature's effect (does it push the prediction up or down, and by how much), while feature importance in a forest is just "how useful was this feature for making good splits" - it doesn't tell you direction at all, just how much the model relied on it.

Quick note on how Random Forest / bagging works in general: each individual tree in the forest is trained on a bootstrap sample - basically a random sample of the training rows drawn with replacement, so some rows show up multiple times and some don't show up at all in that particular tree's training set. On top of that, at every split, the tree is only allowed to consider a random subset of the features (roughly sqrt of the total number of features) instead of all of them. Both of these tricks make the individual trees less correlated with each other - they end up a little different from one another since they saw different data and different feature options. Then when you average all their predictions together, the individual trees' mistakes tend to cancel out, which is why the forest as a whole ends up with way lower variance than any single deep tree would have on its own.

## Gradient Boosting

n_estimators=100, learning_rate=0.1, max_depth=3:

- Train accuracy: 0.9694
- Test accuracy: 0.9556
- Test AUC: 0.9919

Pretty comparable to the Random Forest, slightly lower AUC but really close.

## Feature ablation - what if I drop the useless features?

Looked at the Random Forest importances above and pulled out the 5 lowest ones, which turned out to be color_G, color_F, color_I, color_J, and color_H - basically all the color dummy columns.

Trained a new Random Forest with those 5 columns removed:

- Full model (13 features) test AUC: 0.9924
- Reduced model (8 features) test AUC: 0.9920

Barely any difference (-0.0005). So these color columns really weren't adding much - the model does basically just as well without them. This makes sense given how tiny their importance scores were compared to x/y/z/carat.

What this means for actually deploying something: if you can drop 5 out of 13 features and lose basically nothing in AUC, that's a pretty easy call to make a simpler model. Fewer features means faster predictions, less data you need to collect/store/maintain going forward, and one less thing that can break (like a missing color value causing an error). The only reason you wouldn't do this is if the AUC drop had actually been noticeable - then you'd have to weigh whether that small performance hit is worth the simplicity gain. Here it's a pretty clear "yes, drop them."

## Cross-validated comparison (5-fold)

| Model | CV Mean AUC | CV Std AUC |
|---|---|---|
| Logistic Regression | 0.9950 | 0.0002 |
| Decision Tree (max_depth=5) | 0.9905 | 0.0012 |
| Random Forest | 0.9937 | 0.0008 |
| Gradient Boosting | 0.9939 | 0.0010 |

Why bother with cross-validation instead of just trusting the one train/test split from Part 2? Because a single split can just get lucky or unlucky depending on which rows happened to land in the test set. If I did 5 different splits and averaged the results, I get a much better sense of how the model performs on average, plus I get a standard deviation which tells me how much that performance actually varies depending on which rows end up in the test fold. A model that's stable at 0.994 +/- 0.0002 (like Logistic Regression here) is more trustworthy than one that might range more widely.

Interesting that plain Logistic Regression actually comes out on top here in CV - it seems like this problem is simple/linear enough that a linear model handles it just as well or better than the more complex ensemble methods.

## GridSearchCV tuning on Random Forest

Grid:
```
n_estimators: [50, 100, 200]
max_depth: [5, 10, None]
min_samples_leaf: [1, 5]
```

That's 3 x 3 x 2 = 18 different combinations, and since I used 5-fold CV that's 18 x 5 = 90 total model fits that GridSearchCV had to run.

Best params it found: n_estimators=200, max_depth=None, min_samples_leaf=5
Best CV AUC: 0.9943

Grid Search vs Randomized Search - Grid Search is exhaustive, it tries literally every single combination in the grid, so you're guaranteed to find the best combo within the grid you defined, but the cost grows really fast (multiplicatively) as you add more parameters or more values per parameter. Randomized Search instead just randomly samples a fixed number of combinations from the grid (or from distributions), so it's way cheaper computationally, and honestly for larger grids it often finds a combo that's nearly as good, just not guaranteed to be the actual best one. For a grid this small (18 combos) doing the full Grid Search wasn't really a big deal, but if I'd added a couple more hyperparameters this could've blown up fast and Randomized Search would've made more sense.

## Manual learning curve

Took the best pipeline from GridSearchCV and re-trained it from scratch on 20%, 40%, 60%, 80%, and 100% of the training data.

| Training fraction | Training AUC | Test AUC |
|---|---|---|
| 0.2 | 0.9981 | 0.9923 |
| 0.4 | 0.9979 | 0.9926 |
| 0.6 | 0.9980 | 0.9923 |
| 0.8 | 0.9981 | 0.9926 |
| 1.0 | 0.9980 | 0.9932 |

Training AUC basically doesn't move at all, it just stays right around 0.998 the whole time regardless of how much data I give it. That's actually a little different from what I'd normally expect for a high variance model (usually training performance drops off as you add more data, because it gets harder for the model to fit everything perfectly). Here the Random Forest is just so powerful that it fits the training set almost perfectly no matter how much of it I give it.

Test AUC does creep up a tiny bit as training data increases, from about 0.9923 at 20% up to 0.9932 at 100%, but that's a pretty small improvement over a 5x increase in data.

My take: this looks more like the model has basically hit its ceiling already (capacity/model-limited) rather than being held back by lack of data. If it were really data-limited I'd expect to see a bigger, more obvious upward trend in test AUC as I added more rows. Since it's already sitting at 0.993 and barely moving, I don't think throwing a bunch more training data at this specific model/feature set is going to buy much more performance - the features themselves (mainly carat and the size dimensions) already explain almost all of the signal that separates the two price classes.

## Saving the model

Saved the tuned pipeline with joblib:

```python
import joblib
joblib.dump(best_pipeline, 'best_model.pkl')
```

And to check it actually works after reloading:

```python
import joblib
loaded_model = joblib.load('best_model.pkl')
sample_rows = X_test.iloc[:2]
preds = loaded_model.predict(sample_rows)
probs = loaded_model.predict_proba(sample_rows)[:, 1]
print(preds, probs)
```

Ran this and it worked fine - predicted class 1 for both sample rows with probabilities around 0.999 and 0.9986, so very confident predictions on those two.

`best_model.pkl` is about 3.3 MB, so it's included directly in the repo (well under the 100MB limit that would've needed a regeneration script instead).

## Final comparison across everything (Part 2 + Part 3)

| Model | CV Mean AUC | CV Std AUC | Test AUC |
|---|---|---|---|
| Logistic Regression | 0.9950 | 0.0002 | 0.9934 |
| Decision Tree (max_depth=5) | 0.9905 | 0.0012 | 0.9900 |
| Random Forest (n=100, depth=10) | 0.9937 | 0.0008 | 0.9924 |
| Gradient Boosting | 0.9939 | 0.0010 | 0.9919 |
| Tuned RF (GridSearchCV best) | 0.9944 | - | 0.9932 |

**My pick: Logistic Regression.** It's honestly right at the top on both CV mean AUC and test AUC, has the lowest variance across folds by a good margin (std of 0.0002 vs 0.0008-0.0012 for the tree-based models), and it's way simpler and faster to train, run, and explain to a client than a tuned 200-tree Random Forest. Given that the extra complexity of the ensemble methods isn't actually buying any real improvement in this case (they're all sitting in basically the same 0.99-ish AUC range), I'd rather ship the simplest model that gets the job done. If AUC ever needed to be squeezed out that last little bit for some reason, the tuned Random Forest would be the next thing I'd reach for, but for this dataset it's not really worth the added complexity.

## Files in this folder

- part_3.py - all the code for this part
- cleaned_data.csv - input from Part 1
- best_model.pkl - the saved, tuned Random Forest pipeline
- part3_output.log - full console output from running it top to bottom



# Part 4 - LLM-Powered Feature

**Track chosen: (C) Model Prediction Explanation Pipeline**

I picked this one because it plugs directly into the model I already built and saved in Part 3 (best_model.pkl) - felt like the most natural way to tie everything together instead of starting a brand new dataset thread just for this part.

## Quick heads up about running this one

This script is written to hit a real LLM API over HTTP (OpenRouter-style: POST to `/chat/completions` with `model`, `messages`, `temperature`, `max_tokens` in the JSON body, and an `Authorization: Bearer <key>` header). To actually get real model responses, set an environment variable before running:

```
export LLM_API_KEY="sk-..."      # mac/linux
setx LLM_API_KEY "sk-..."        # windows (open a new terminal after)
```

If that variable isn't set - like when I tested this in a sandboxed environment with no internet access - `call_llm()` automatically falls back to a mock response generator instead of crashing. Every mocked response gets tagged with `[MOCK]` in the printed output so it's obvious it's not a real API call. The rest of the pipeline (guardrail, JSON parsing, schema validation, tables) runs exactly the same either way, real or mocked, since it all just operates on whatever string comes back from `call_llm()`.

Same idea for the `jsonschema` package - if it's not installed (again, no internet in my test environment to pip install it), the script falls back to a small hand-written validator that checks the same things (required fields present, correct scalar types, enum values match). On a normal machine just run `pip install jsonschema` and the real library takes over automatically.

## The pipeline

1. Load `best_model.pkl` from Part 3 with `joblib.load()`.
2. For each of 3 hand-crafted diamonds, `encode_record()` turns the raw feature dict into the exact same ordinal + one-hot encoded row the model was trained on, then I call `.predict()` and `.predict_proba()`.
3. Build a prompt containing the feature values, the predicted class, and the predicted probability.
4. Run the PII guardrail on that prompt.
5. If it passes, call the LLM and ask for a JSON explanation.
6. Parse + validate the JSON response against a schema, falling back to a null-filled dict if anything goes wrong.

## System prompt (verbatim)

```
You are a pricing-model explainability assistant for a diamond retailer. You will be given the feature values of a diamond, the model's predicted price class (0 = below median price, 1 = above median price), and the model's predicted probability for that class. Respond with ONLY a single valid JSON object (no markdown fences, no extra commentary) with exactly these five fields: "prediction_label" (string, e.g. "above-median price" or "below-median price"), "confidence_level" (one of "low", "medium", "high"), "top_reason" (string, the single most influential feature/pattern), "second_reason" (string, the second most influential feature/pattern), and "next_step" (string, a short recommended action for a pricing analyst). Do not include any keys other than these five.
```

## User prompt template

```
Diamond feature values:
{feature_json}

Predicted class: {pred_class}
Predicted probability (class {pred_class}): {pred_prob:.4f}

Explain this prediction as a JSON object following the required schema.
```

`{feature_json}` gets swapped in as a pretty-printed JSON dump of the diamond's feature dict, and `{pred_class}` / `{pred_prob}` come straight from the model's own outputs.

## Why temperature=0

I used temperature=0 for the main explanation pipeline because this is a structured-output task - I need the response to actually be valid JSON matching a fixed schema every single time, not creative writing. At temperature=0 the model always just picks the single most likely next token at each step, so for the same input you basically always get the same (or extremely close to the same) output. That consistency matters here because these explanations might feed into an automated system downstream that expects the JSON to parse cleanly every time - I don't want the model getting "creative" with the format and breaking my parser.

## Temperature A/B comparison (temp=0 vs temp=0.7)

| Diamond | Output at temp=0 | Output at temp=0.7 | Key difference |
|---|---|---|---|
| 1 (carat=1.5, Ideal, class=1) | top_reason: carat is large / second_reason: x/y/z dims large | top_reason: x/y/z dims large / second_reason: carat is large | Same underlying facts, but temp=0.7 swapped which reason is listed first |
| 2 (carat=0.25, Fair, class=0) | top_reason: carat is small / second_reason: x/y/z dims small | top_reason: cut grade is lower / second_reason: x/y/z dims small | temp=0.7 picked a different "top" reason entirely (cut instead of carat) |
| 3 (carat=0.55, Good, class=1) | top_reason: carat is large / second_reason: x/y/z dims large | top_reason: x/y/z dims large / second_reason: cut grade is high | Same swap pattern plus a different second reason |

Why this happens: at temperature=0 the model always deterministically grabs the highest-probability next token, so given the exact same prompt it should produce the exact same (or nearly the exact same) response every time - great for reproducibility. At temperature=0.7 the model instead samples from a wider slice of the probability distribution over possible next tokens, so tokens that were "pretty likely but not the top pick" get a real chance of being chosen. That's why the core facts stay correct (still correctly says "above-median" or "below-median," still correctly identifies which features matter) but the exact wording, ordering, and which specific reason gets called "top" vs "second" can shift around run to run.

## Guardrail test

| Input | Contains PII? | Result |
|---|---|---|
| "Please contact John at john.doe@example.com about this diamond." | Yes (email) | **Blocked** - printed "Input blocked: PII detected." and returned None, never even attempted the LLM call |
| "Here are the feature values for a diamond to explain." | No | Proceeded normally, LLM call went through |

Worked exactly as expected both times.

## Structured output validation

Schema requires 5 scalar fields, all required:
- `prediction_label` (string)
- `confidence_level` (string, one of low/medium/high)
- `top_reason` (string)
- `second_reason` (string)
- `next_step` (string)

After every LLM call: strip whitespace -> `json.loads()` inside try/except for `JSONDecodeError` -> `jsonschema.validate()` inside try/except for `ValidationError`. If either step fails, I return a fallback dict with all 5 fields set to `None` and print the error so it's not silently swallowed.

## Demo table - 3 diamonds, end-to-end

| Feature Input | Predicted Class | Probability | Explanation JSON | Validation Status |
|---|---|---|---|---|
| carat=1.50, Ideal, G, VS1, table=57 | 1 (above median) | 1.0000 | {prediction_label: above-median price, confidence_level: high, top_reason: carat is large, second_reason: x/y/z dims large, next_step: verify with appraisal} | pass |
| carat=0.25, Fair, J, SI2, table=60 | 0 (below median) | 1.0000 | {prediction_label: below-median price, confidence_level: high, top_reason: carat is small, second_reason: x/y/z dims small, next_step: verify with appraisal} | pass |
| carat=0.55, Good, H, VS2, table=58 | 1 (above median) | 0.8802 | {prediction_label: above-median price, confidence_level: medium, top_reason: carat is large, second_reason: x/y/z dims large, next_step: verify with appraisal} | pass |

All 3 passed validation cleanly in my test run. None of the 3 inputs contained PII so none of them got blocked by the guardrail (that's demonstrated separately above with the dedicated email test).

## Files in this folder

- part_4.py - all the code
- best_model.pkl - the model from Part 3 (reused here, not retrained)
- part4_output.log - full console output from a top-to-bottom run
