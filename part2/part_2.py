import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score,
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score,
)

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(HERE, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
SEED = 42


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ===========================================================================
# TASK 1: load cleaned data, define X, y_reg, y_clf
# ===========================================================================
def task1_load_and_define(path):
    hr("TASK 1: Load cleaned_data.csv and define X / y_reg / y_clf")
    df = pd.read_csv(path)
    # dtypes lost through CSV round-trip: cut/color/clarity were saved as plain
    # strings, so we simply treat them as categorical text columns here.
    print(f"Loaded cleaned_data.csv - shape {df.shape}")
    print(df.head())

    target = "price"
    X = df.drop(columns=[target]).copy()
    y_reg = df[target].copy()
    y_clf = (y_reg > y_reg.median()).astype(int)

    print(f"\nRegression label y_reg = '{target}' (continuous, USD)")
    print(f"Classification label y_clf = 1 if {target} > median({y_reg.median():.2f}) else 0")
    print(f"y_clf class balance:\n{y_clf.value_counts(normalize=True).round(3)}")
    return df, X, y_reg, y_clf


# ===========================================================================
# TASK 2: encode categorical columns
#   - cut, clarity: natural order -> ordinal label encoding
#   - color: no meaningful order for this exercise -> one-hot encoding
# ===========================================================================
CUT_ORDER = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
CLARITY_ORDER = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4,
                  "VVS2": 5, "VVS1": 6, "IF": 7}


def task2_encode(X):
    hr("TASK 2: Encode categorical columns")
    X = X.copy()

    print("Ordinal label encoding: 'cut' ->", CUT_ORDER)
    X["cut"] = X["cut"].map(CUT_ORDER)
    print("Ordinal label encoding: 'clarity' ->", CLARITY_ORDER)
    X["clarity"] = X["clarity"].map(CLARITY_ORDER)

    print("\nOne-hot encoding 'color' (drop_first=True to avoid multicollinearity)")
    X = pd.get_dummies(X, columns=["color"], drop_first=True)
    # get_dummies returns bool columns in modern pandas -> cast to int for
    # a clean numeric feature matrix
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

    print(f"\nEncoded feature matrix shape: {X.shape}")
    print("Columns:", list(X.columns))
    return X


# ===========================================================================
# TASK 3: leak-free split + scaling
# ===========================================================================
def task3_split_and_scale(X, y_reg, y_clf):
    hr("TASK 3: Leak-free train/test split and scaling")
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=SEED
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    scaler = StandardScaler()
    scaler.fit(X_train)  # fit ONLY on training data - no leakage
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    print("Scaler fitted on X_train ONLY, then applied to both X_train and X_test.")
    print("(Fitting on the full dataset would leak test-set mean/variance into training.)")
    return X_train_scaled, X_test_scaled, y_reg_train, y_reg_test, y_clf_train, y_clf_test, scaler


# ===========================================================================
# TASK 4: Linear Regression + Ridge Regression
# ===========================================================================
def task4_regression(X_train, X_test, y_reg_train, y_reg_test, feature_names):
    hr("TASK 4: Linear Regression")
    lin = LinearRegression()
    lin.fit(X_train, y_reg_train)
    y_pred_reg = lin.predict(X_test)

    mse_lin = mean_squared_error(y_reg_test, y_pred_reg)
    r2_lin = r2_score(y_reg_test, y_pred_reg)
    print(f"Linear Regression -> MSE = {mse_lin:.3f}, R2 = {r2_lin:.4f}")

    coef_table = pd.Series(lin.coef_, index=feature_names).sort_values(key=np.abs, ascending=False)
    print("\nCoefficients (sorted by |value|):")
    print(coef_table)
    top3 = coef_table.head(3)
    print(f"\nTop-3 features by |coefficient|: {list(top3.index)}")

    hr("TASK 4b: Ridge Regression (alpha=1.0)")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_reg_train)
    y_pred_ridge = ridge.predict(X_test)
    mse_ridge = mean_squared_error(y_reg_test, y_pred_ridge)
    r2_ridge = r2_score(y_reg_test, y_pred_ridge)
    print(f"Ridge Regression -> MSE = {mse_ridge:.3f}, R2 = {r2_ridge:.4f}")

    comparison = pd.DataFrame({
        "Model": ["Linear Regression (OLS)", "Ridge (alpha=1.0)"],
        "MSE": [mse_lin, mse_ridge],
        "R2": [r2_lin, r2_ridge],
    })
    print("\nComparison table:")
    print(comparison.to_string(index=False))

    return {
        "lin_model": lin, "ridge_model": ridge,
        "mse_lin": mse_lin, "r2_lin": r2_lin,
        "mse_ridge": mse_ridge, "r2_ridge": r2_ridge,
        "coef_table": coef_table, "top3": top3,
        "comparison": comparison,
    }


# ===========================================================================
# TASK 5: Logistic Regression classification
# ===========================================================================
def task5_classification(X_train, X_test, y_clf_train, y_clf_test):
    hr("TASK 5: Logistic Regression classification")
    counts = y_clf_train.value_counts()
    props = y_clf_train.value_counts(normalize=True)
    print("y_clf_train class counts (before any imbalance handling):")
    print(counts)
    print(props.round(3))

    minority_share = props.min()
    if minority_share < 0.35:
        strategy = "class_weight='balanced'"
        print(f"\nMinority class share = {minority_share:.2%} (< 35%) -> imbalance handling required.")
        print(f"Chosen strategy: {strategy} "
              "(SMOTE from imblearn was considered but the runtime has no network "
              "access to install imbalanced-learn; class_weight='balanced' achieves the "
              "same goal by re-weighting the loss function instead of resampling rows).")
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=SEED)
    else:
        print(f"\nMinority class share = {minority_share:.2%} (>= 35%) -> classes are reasonably "
              "balanced; median-split guarantees ~50/50, so no resampling is applied.")
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)

    clf.fit(X_train, y_clf_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_clf_test, y_pred)
    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification report:")
    report = classification_report(y_clf_test, y_pred)
    print(report)

    fpr, tpr, thresholds = roc_curve(y_clf_test, y_proba)
    auc = roc_auc_score(y_clf_test, y_proba)
    print(f"AUC = {auc:.4f}")

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="crimson", label=f"Logistic Regression (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.title("ROC Curve - Logistic Regression (price above/below median)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.annotate(f"AUC = {auc:.3f}", xy=(0.55, 0.15), fontsize=12,
                 bbox=dict(boxstyle="round", fc="white", ec="crimson"))
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "07_roc_curve.png"), dpi=150)
    plt.close()

    return {
        "model": clf, "y_pred": y_pred, "y_proba": y_proba,
        "cm": cm, "report": report, "fpr": fpr, "tpr": tpr, "auc": auc,
        "class_counts_before": counts, "strategy": strategy if minority_share < 0.35 else "none needed",
    }


# ===========================================================================
# TASK 5b: decision-threshold sensitivity
# ===========================================================================
def task5b_threshold_sensitivity(y_clf_test, y_proba):
    hr("TASK 5b: Decision-threshold sensitivity (0.30 - 0.70)")
    rows = []
    for t in [0.30, 0.40, 0.50, 0.60, 0.70]:
        preds = (y_proba >= t).astype(int)
        p = precision_score(y_clf_test, preds, zero_division=0)
        r = recall_score(y_clf_test, preds, zero_division=0)
        f1 = f1_score(y_clf_test, preds, zero_division=0)
        rows.append({"Threshold": t, "Precision": p, "Recall": r, "F1": f1})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    best_row = table.loc[table["F1"].idxmax()]
    print(f"\nThreshold that maximises F1: {best_row['Threshold']:.2f} (F1 = {best_row['F1']:.4f})")
    return table, best_row


# ===========================================================================
# TASK 6: regularization experiment (C=1.0 vs C=0.01)
# ===========================================================================
def task6_regularization(X_train, X_test, y_clf_train, y_clf_test, class_weight):
    hr("TASK 6: Regularization experiment (C=1.0 vs C=0.01)")
    clf_c1 = LogisticRegression(max_iter=1000, C=1.0, class_weight=class_weight, random_state=SEED)
    clf_c1.fit(X_train, y_clf_train)
    proba_c1 = clf_c1.predict_proba(X_test)[:, 1]
    pred_c1 = clf_c1.predict(X_test)

    clf_c001 = LogisticRegression(max_iter=1000, C=0.01, class_weight=class_weight, random_state=SEED)
    clf_c001.fit(X_train, y_clf_train)
    proba_c001 = clf_c001.predict_proba(X_test)[:, 1]
    pred_c001 = clf_c001.predict(X_test)

    metrics = pd.DataFrame({
        "Model": ["Logistic C=1.0", "Logistic C=0.01"],
        "Precision": [precision_score(y_clf_test, pred_c1), precision_score(y_clf_test, pred_c001)],
        "Recall": [recall_score(y_clf_test, pred_c1), recall_score(y_clf_test, pred_c001)],
        "AUC": [roc_auc_score(y_clf_test, proba_c1), roc_auc_score(y_clf_test, proba_c001)],
    })
    print(metrics.to_string(index=False))

    return {
        "clf_c1": clf_c1, "clf_c001": clf_c001,
        "proba_c1": proba_c1, "proba_c001": proba_c001,
        "metrics": metrics,
    }


# ===========================================================================
# TASK 6b: bootstrap confidence interval for AUC difference
# ===========================================================================
def task6b_bootstrap_auc_ci(y_clf_test, proba_c1, proba_c001, n_boot=500):
    hr("TASK 6b: Bootstrap confidence interval for AUC difference (C=1.0 minus C=0.01)")
    y_arr = np.asarray(y_clf_test)
    rng = np.random.default_rng(SEED)
    diffs = np.empty(n_boot)

    n = len(y_arr)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        y_sample = y_arr[idx]
        # a bootstrap sample can occasionally contain only one class; skip
        # AUC computation for that (rare) sample by treating the diff as 0
        if len(np.unique(y_sample)) < 2:
            diffs[i] = 0.0
            continue
        auc1 = roc_auc_score(y_sample, proba_c1[idx])
        auc2 = roc_auc_score(y_sample, proba_c001[idx])
        diffs[i] = auc1 - auc2

    mean_diff = diffs.mean()
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    print(f"Mean AUC difference (C=1.0 - C=0.01): {mean_diff:.4f}")
    print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    excludes_zero = (ci_low > 0) or (ci_high < 0)
    print(f"CI excludes zero: {excludes_zero}")
    return mean_diff, ci_low, ci_high, excludes_zero


def main():
    cleaned_path = os.path.join(HERE, "..", "part1", "cleaned_data.csv")
    
    df, X, y_reg, y_clf = task1_load_and_define(cleaned_path)
    X_enc = task2_encode(X)
    (X_train, X_test, y_reg_train, y_reg_test,
     y_clf_train, y_clf_test, scaler) = task3_split_and_scale(X_enc, y_reg, y_clf)

    reg_results = task4_regression(X_train, X_test, y_reg_train, y_reg_test, X_enc.columns)
    clf_results = task5_classification(X_train, X_test, y_clf_train, y_clf_test)
    thresh_table, best_thresh_row = task5b_threshold_sensitivity(y_clf_test, clf_results["y_proba"])

    class_weight = "balanced" if clf_results["strategy"] != "none needed" else None
    reg_exp = task6_regularization(X_train, X_test, y_clf_train, y_clf_test, class_weight)
    mean_diff, ci_low, ci_high, excludes_zero = task6b_bootstrap_auc_ci(
        y_clf_test, reg_exp["proba_c1"], reg_exp["proba_c001"]
    )

    hr("SUMMARY")
    print(f"Linear Regression: MSE={reg_results['mse_lin']:.2f}, R2={reg_results['r2_lin']:.4f}")
    print(f"Ridge Regression:  MSE={reg_results['mse_ridge']:.2f}, R2={reg_results['r2_ridge']:.4f}")
    print(f"Logistic Regression AUC: {clf_results['auc']:.4f}")
    print(f"Best F1 threshold: {best_thresh_row['Threshold']:.2f}")
    print(f"Bootstrap AUC diff (C=1.0 vs C=0.01): mean={mean_diff:.4f}, "
          f"95% CI=({ci_low:.4f}, {ci_high:.4f}), excludes zero: {excludes_zero}")
    print(f"\nPlots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()