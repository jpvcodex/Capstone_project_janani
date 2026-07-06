# Diamonds Capstone Project

## Part 1 — Data Cleaning & EDA

**Data:** `diamonds` dataset (mwaskom/seaborn-data, ~54k rows). Script intentionally "messes up" the clean source to simulate a real client export: nulls in `depth` (23%) and `table` (7%), `carat` stored as text with "unknown"/"n/a" entries, and 45 duplicate rows added. Falls back to a synthetic dataset if the GitHub URL is unreachable.

**Steps:**
- **Load:** raw shape (53985, 10).
- **Nulls:** `depth` 23% null → dropped (over 20% cutoff). `table` 7% null → filled with median (57.0). Used median (not mean) since carat/price/y/z are skewed.
- **Duplicates:** 131 found and dropped → 53854 rows. Null % barely changed.
- **Dtypes:** `carat` converted via `pd.to_numeric(errors='coerce')`, exposing 2157 hidden nulls → filled with median (0.700). `cut`/`color`/`clarity` converted to `category`. Memory dropped from 13,579 KB → 2,683 KB (~80%).
- **Skew:** `y` most skewed (2.443), driven by extreme outlier values.
- **Outliers (IQR):** price 6.56%, carat 3.86% flagged — kept, since they're real high-end diamonds, not errors. Plan to log-transform for linear models instead of capping.
- **Plots (in `plots/`):** sorted price line, mean price by cut bar chart, `y` histogram, carat-vs-price scatter (Pearson 0.904, Spearman 0.945 — curved not linear relationship), price-by-cut boxplot. Interesting finding: Premium has highest mean price, Ideal lowest — cut alone doesn't predict price, likely confounded with size.
- **Correlation heatmap:** x/y both dimensions of the same stone — the underlying driver of their correlation is really carat/size (r = 0.975).
- **Mean vs median (y, price):** both used median for imputation due to skew.
- **Spearman vs Pearson:** price vs y/z/x relationships are monotonic but non-linear — Spearman consistently higher, so Spearman preferred for feature selection in Part 2.
- **Grouped price by cut:** Premium highest mean & std; std exceeds mean in every group → cut alone is a weak predictor.

**Output:** `cleaned_data.csv` — 53854 rows, 9 columns, no nulls.

**Files:** `eda_pipeline.py`, `raw_diamonds.csv`, `cleaned_data.csv`, `plots/`, `run_log.txt`

---

## Part 2 — ML Models (Regression + Classification)

Run: `python part_2.py` (needs `cleaned_data.csv`).

- **Labels:** Regression target = `price`. Classification target = binary split at median price (~2273.5), giving a near-perfect 50/50 class balance.
- **Encoding:** `cut`/`clarity` label-encoded (ordinal). `color` one-hot encoded, first dummy dropped (no natural order → avoids fake ordinal relationship / dummy trap).
- **Split/scaling:** 80/20 split (`random_state=42`). Scaler fit only on training data to avoid leakage.

**Regression:**

| Model | MSE | R² |
|---|---|---|
| Linear Regression | 2,299,438.83 | 0.8866 |
| Ridge (α=1.0) | 2,300,689.85 | 0.8865 |

Top features: carat, x, cut. `x`'s negative coefficient is a multicollinearity artifact (carat/x/y/z all measure size), not a real effect. Ridge barely changed results — no meaningful overfitting to begin with.

**Classification (Logistic Regression):**
- Balanced classes (49.6/50.4%) → no SMOTE needed.
- Accuracy 0.96, AUC 0.9934.
- Threshold sweep (0.3–0.7): best F1 at default 0.50.
- C=1.0 vs C=0.01: negligible difference; 500-sample bootstrap CI on AUC diff includes 0 → not statistically meaningful.

**Files:** `part_2.py`, `cleaned_data.csv`, `plots/07_roc_curve.png`, `part2_output.log`

---

## Part 3 — Ensembles, Tuning, Full Pipeline

Run: `python part_3.py` (rebuilds the same split; GridSearch step takes ~40s).

- **Unrestricted decision tree:** train 1.00 / test 0.9337 (gap 0.066) — clear overfitting.
- **Controlled tree** (max_depth=5, min_samples_split=20): train 0.9652 / test 0.9556 (gap 0.0095) — much better generalization.
- **Gini vs Entropy:** virtually identical (0.9556 vs 0.9550).
- **Random Forest** (n=100, depth=10): test acc 0.9575, AUC 0.9924. Top features: x, y, z, carat (all size-related).
- **Gradient Boosting:** comparable, slightly lower AUC (0.9919).
- **Feature ablation:** dropping 5 low-importance color columns cost almost nothing (AUC 0.9924 → 0.9920) — simpler model is the right call.
- **5-fold CV comparison:**

| Model | CV Mean AUC | CV Std |
|---|---|---|
| Logistic Regression | 0.9950 | 0.0002 |
| Decision Tree (d=5) | 0.9905 | 0.0012 |
| Random Forest | 0.9937 | 0.0008 |
| Gradient Boosting | 0.9939 | 0.0010 |

- **GridSearchCV** (18 combos × 5 folds = 90 fits): best = n_estimators=200, max_depth=None, min_samples_leaf=5, CV AUC 0.9943.
- **Learning curve:** training AUC flat (~0.998) regardless of data size; test AUC creeps from 0.9923→0.9932 — model appears capacity-limited, not data-limited.
- **Saved model:** `best_model.pkl` (~3.3 MB) via joblib, verified reload/predict works.

**Final pick: Logistic Regression** — top or near-top AUC, lowest variance, far simpler than the tuned 200-tree Random Forest.

**Files:** `part_3.py`, `cleaned_data.csv`, `best_model.pkl`, `part3_output.log`

---

## Part 4 — LLM-Powered Feature (Model Prediction Explanation Pipeline)

Chosen since it builds directly on `best_model.pkl` from Part 3.

**Setup:** Calls an OpenRouter-style chat completions API using `LLM_API_KEY` env var. Falls back to a mock generator (tagged `[MOCK]`) if no key/internet; falls back to a hand-written JSON validator if `jsonschema` isn't installed. Rest of pipeline (guardrail, parsing, validation) runs identically either way.

**Pipeline:** load model → encode 3 sample diamonds → predict class/probability → build prompt → run PII guardrail → call LLM → parse & validate JSON response (5 required fields: `prediction_label`, `confidence_level`, `top_reason`, `second_reason`, `next_step`) → fallback to null-filled dict on failure.

**Why temperature=0:** structured-output task needs consistent, parseable JSON every run — deterministic token selection avoids format breakage.

**Temp 0 vs 0.7 comparison:** core facts (class direction, correct features) stayed consistent across both, but ordering/wording of reasons shifted at 0.7 due to sampling from a wider probability distribution.

**Guardrail test:** email-containing input correctly blocked before any LLM call; clean input passed through normally.

**Demo:** 3 sample diamonds run end-to-end, all passed schema validation.

**Files:** `part_4.py`, `best_model.pkl`, `part4_output.log`
