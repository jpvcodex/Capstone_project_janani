# Part 3 — Tree-Based Models, Tuning & Serialization

## What this does

`part_3.py` rebuilds the same feature matrix and split as Part 2 (it's
self-contained so it can run on its own) and then moves on to tree-based
models.

- Setup: reload `cleaned_data.csv`, redo the same encoding + 80/20 split +
  `StandardScaler` as Part 2.
- Unconstrained Decision Tree (`max_depth=None`) as a baseline — shows the
  overfitting gap between train and test accuracy.
- Controlled Decision Tree (`max_depth=5, min_samples_split=20`) — shows how
  much constraining depth closes that gap.
- Compared Gini vs. Entropy as splitting criteria, both at `max_depth=5`.
- Random Forest (`n_estimators=100, max_depth=10`) — train/test accuracy,
  test ROC-AUC, full feature importances.
- Gradient Boosting (`n_estimators=100, learning_rate=0.1, max_depth=3`) —
  same metrics, for comparison.
- Feature ablation: dropped the 5 lowest-importance features (per the Random
  Forest) and re-checked AUC to see how much they were actually contributing.
- 5-fold CV comparing Logistic Regression, Decision Tree, Random Forest, and
  Gradient Boosting on mean/std ROC-AUC.
- GridSearchCV over a `SimpleImputer → StandardScaler → RandomForestClassifier`
  pipeline, tuning `n_estimators`, `max_depth`, `min_samples_leaf` (18
  configs × 5 folds = 90 fits total — took a while to run).
- Manual learning curve: trained the best pipeline on 20/40/60/80/100% of
  the training data to see how train/test AUC change with more data.
- Saved the best GridSearchCV pipeline to `best_model.pkl` with joblib, then
  reloaded it and predicted on 2 held-out rows just to confirm it actually
  works after reloading.
- Final comparison table pulling together CV mean/std AUC and test AUC for
  every model across Parts 2 and 3.

## Running it

```bash
pip install pandas numpy matplotlib scikit-learn joblib
python part_3.py
```

Needs `cleaned_data.csv` from Part 1 — script looks for it at
`../part1/cleaned_data.csv` (change `cleaned_path` in `main()` if your
folder layout is different).

No API keys needed.

Outputs:
- `best_model.pkl` — the serialized best pipeline. **Part 4 needs this** —
  either copy/symlink it into `part4/`, or point Part 4's `model_path` at
  `../part3/best_model.pkl`.
- Everything else printed to console (summarized below).

## Findings

- Unconstrained Decision Tree hit 100% training accuracy but only ~97.05%
  on test — about a 3-point overfitting gap. Capping it at `max_depth=5`
  closed that almost completely (train ≈ test ≈ 96.4–96.5%), at the cost of
  a bit of raw accuracy.
- Gini beat Entropy slightly at the same depth (96.47% vs 96.16% test
  accuracy).
- Random Forest and Gradient Boosting basically tied (test AUC ≈ 0.998 for
  both), and both clearly beat the single tree and logistic regression.
- Feature importance was dominated by the diamond's physical dimensions —
  `y`, `x`, `z` together made up over 80% of Random Forest importance, with
  `carat` next. Dropping the 5 least important features (the color one-hot
  columns plus `cut`) only cost ~0.0009 AUC, so those aren't pulling much
  weight.
- GridSearchCV (all 90 fits) landed on `n_estimators=200, max_depth=None,
  min_samples_leaf=1` as the best config, CV AUC ≈ 0.9982.
- The learning curve barely moved from 20% to 100% of the training data
  (0.9972 → 0.9978 test AUC) — feels like the model's close to its ceiling
  on this feature set and more data wouldn't help much at this point.
