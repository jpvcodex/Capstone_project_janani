# Part 2 — Regression & Classification

## What this does

`part_2.py` picks up `cleaned_data.csv` from Part 1 and builds the first
round of models.

- Load the cleaned data, drop `price` out of the feature matrix `X`, and
  set up two targets: `y_reg` (price itself, for regression) and `y_clf`
  (1 if price is above the median, 0 otherwise, for classification).
- Encode the categorical columns:
  - `cut` and `clarity` get ordinal encoding since they have a real
    quality order to them.
  - `color` gets one-hot encoded with `drop_first=True` since there's no
    natural ordering there.
- Split 80/20, then fit `StandardScaler` only on the training set (fitting
  it on everything would leak test-set info into training).
- Train Linear Regression and Ridge (alpha=1.0), compare MSE/R² and
  coefficients.
- Logistic Regression for the classification target — check class balance
  first (would've used `class_weight="balanced"` if the minority class was
  under 35%, but it wasn't needed here), then confusion matrix,
  classification report, ROC-AUC, and save the ROC curve.
- Sweep decision thresholds from 0.30 to 0.70 and check precision/recall/F1
  at each to find whichever threshold maximizes F1.
- Compare Logistic Regression at C=1.0 vs C=0.01.
- Bootstrap (500 iterations) to get a 95% CI on the AUC difference between
  those two C values, and check if that interval crosses zero.

## Running it

```bash
pip install pandas numpy matplotlib scikit-learn
python part_2.py
```

Needs `cleaned_data.csv` from Part 1 first — script looks for it at
`../part1/cleaned_data.csv` (change `cleaned_path` in `main()` if your
folders are laid out differently).

No API keys needed here.

Outputs: `plots/07_roc_curve.png` (numbered to line up with Part 1's plots)
plus everything printed to console.

## Findings

- The above/below-median split came out almost exactly 50/50 on its own, so
  no imbalance handling was actually needed.
- Linear Regression got R² ≈ 0.87, MSE ≈ 1.88M. Ridge came out basically
  identical, which tells me multicollinearity isn't really an issue here at
  that regularization strength.
- Biggest regression coefficients by magnitude: `carat`, `x`, `clarity` —
  which lines up with what you'd expect, since those are the main things
  that actually drive diamond price.
- Logistic Regression hit AUC ≈ 0.998 and ~97% accuracy on the median split
  — the classes are very cleanly separable with these features.
- Best F1 threshold turned out to be 0.40, not the default 0.50, so slightly
  favoring recall.
- C=1.0 vs C=0.01 barely moved the needle — bootstrap 95% CI on the AUC
  difference was [-0.0002, 0.0001], which includes zero, so that difference
  isn't really distinguishable from noise at this sample size.
