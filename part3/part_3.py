import os
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ===========================================================================
# Rebuild the same X / y_clf / train-test split / scaling used in Part 2
# (kept self-contained here so this script can run on its own)
# ===========================================================================
CUT_ORDER = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
CLARITY_ORDER = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4,
                  "VVS2": 5, "VVS1": 6, "IF": 7}


def load_and_prepare(path):
    hr("SETUP: rebuild Part 2's X / y_clf / train-test split / scaling")
    df = pd.read_csv(path)
    target = "price"
    X = df.drop(columns=[target]).copy()
    y_reg = df[target].copy()
    y_clf = (y_reg > y_reg.median()).astype(int)

    X["cut"] = X["cut"].map(CUT_ORDER)
    X["clarity"] = X["clarity"].map(CLARITY_ORDER)
    X = pd.get_dummies(X, columns=["color"], drop_first=True)
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=SEED
    )

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"Feature columns: {list(X.columns)}")
    return X, X_train, X_test, X_train_scaled, X_test_scaled, y_clf_train, y_clf_test


# ===========================================================================
# TASK 1: unconstrained Decision Tree baseline
# ===========================================================================
def task1_unconstrained_tree(X_train_scaled, X_test_scaled, y_clf_train, y_clf_test):
    hr("TASK 1: Decision Tree baseline (unconstrained, max_depth=None)")
    tree = DecisionTreeClassifier(random_state=SEED)
    tree.fit(X_train_scaled, y_clf_train)
    train_acc = accuracy_score(y_clf_train, tree.predict(X_train_scaled))
    test_acc = accuracy_score(y_clf_test, tree.predict(X_test_scaled))
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print(f"Train-test gap: {train_acc - test_acc:.4f}")
    return tree, train_acc, test_acc


# ===========================================================================
# TASK 2: controlled Decision Tree
# ===========================================================================
def task2_controlled_tree(X_train_scaled, X_test_scaled, y_clf_train, y_clf_test):
    hr("TASK 2: Controlled Decision Tree (max_depth=5, min_samples_split=20)")
    tree = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=SEED)
    tree.fit(X_train_scaled, y_clf_train)
    train_acc = accuracy_score(y_clf_train, tree.predict(X_train_scaled))
    test_acc = accuracy_score(y_clf_test, tree.predict(X_test_scaled))
    test_auc = roc_auc_score(y_clf_test, tree.predict_proba(X_test_scaled)[:, 1])
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print(f"Test ROC-AUC:   {test_auc:.4f}")
    print(f"Train-test gap: {train_acc - test_acc:.4f}")
    return tree, train_acc, test_acc, test_auc


# ===========================================================================
# TASK 3: Gini vs Entropy at max_depth=5
# ===========================================================================
def task3_gini_vs_entropy(X_train_scaled, X_test_scaled, y_clf_train, y_clf_test):
    hr("TASK 3: Gini vs Entropy (both max_depth=5)")
    results = {}
    for crit in ["gini", "entropy"]:
        tree = DecisionTreeClassifier(max_depth=5, criterion=crit, random_state=SEED)
        tree.fit(X_train_scaled, y_clf_train)
        test_acc = accuracy_score(y_clf_test, tree.predict(X_test_scaled))
        results[crit] = test_acc
        print(f"criterion='{crit}' -> test accuracy = {test_acc:.4f}")
    return results


# ===========================================================================
# TASK 4: Random Forest + feature importances
# ===========================================================================
def task4_random_forest(X_train_scaled, X_test_scaled, y_clf_train, y_clf_test, feature_names):
    hr("TASK 4: Random Forest (n_estimators=100, max_depth=10)")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED)
    rf.fit(X_train_scaled, y_clf_train)
    train_acc = accuracy_score(y_clf_train, rf.predict(X_train_scaled))
    test_acc = accuracy_score(y_clf_test, rf.predict(X_test_scaled))
    test_auc = roc_auc_score(y_clf_test, rf.predict_proba(X_test_scaled)[:, 1])
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print(f"Test ROC-AUC:   {test_auc:.4f}")

    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\nTop 5 features by importance:")
    print(importances.head(5))
    print("\nAll feature importances (sorted):")
    print(importances)

    return rf, train_acc, test_acc, test_auc, importances


# ===========================================================================
# TASK 4a: Gradient Boosting
# ===========================================================================
def task4a_gradient_boosting(X_train_scaled, X_test_scaled, y_clf_train, y_clf_test):
    hr("TASK 4a: Gradient Boosting (n_estimators=100, learning_rate=0.1, max_depth=3)")
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=SEED)
    gb.fit(X_train_scaled, y_clf_train)
    train_acc = accuracy_score(y_clf_train, gb.predict(X_train_scaled))
    test_acc = accuracy_score(y_clf_test, gb.predict(X_test_scaled))
    test_auc = roc_auc_score(y_clf_test, gb.predict_proba(X_test_scaled)[:, 1])
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print(f"Test ROC-AUC:   {test_auc:.4f}")
    return gb, train_acc, test_acc, test_auc


# ===========================================================================
# TASK 4b: feature ablation using RF importances
# ===========================================================================
def task4b_feature_ablation(X_train_scaled, X_test_scaled, y_clf_train, y_clf_test,
                             importances, full_auc):
    hr("TASK 4b: Feature ablation study (drop 5 lowest-importance features)")
    lowest5 = importances.sort_values(ascending=True).head(5).index.tolist()
    print(f"5 lowest-importance features (to be dropped): {lowest5}")

    X_train_reduced = X_train_scaled.drop(columns=lowest5)
    X_test_reduced = X_test_scaled.drop(columns=lowest5)

    rf_reduced = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED)
    rf_reduced.fit(X_train_reduced, y_clf_train)
    reduced_auc = roc_auc_score(y_clf_test, rf_reduced.predict_proba(X_test_reduced)[:, 1])

    print(f"\nFull model test AUC (all {X_train_scaled.shape[1]} features):    {full_auc:.4f}")
    print(f"Reduced model test AUC ({X_train_reduced.shape[1]} features):        {reduced_auc:.4f}")
    print(f"AUC change from dropping lowest-5 features: {reduced_auc - full_auc:+.4f}")
    return lowest5, reduced_auc


# ===========================================================================
# TASK 5: cross-validated comparison
# ===========================================================================
def task5_cv_comparison(X_train_scaled, y_clf_train):
    hr("TASK 5: 5-fold cross-validated ROC-AUC comparison")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Decision Tree (max_depth=5)": DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=SEED),
    }

    rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X_train_scaled, y_clf_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        rows.append({"Model": name, "CV Mean AUC": scores.mean(), "CV Std AUC": scores.std()})
        print(f"{name:32s} -> mean AUC = {scores.mean():.4f}, std = {scores.std():.4f}")

    table = pd.DataFrame(rows)
    return table


# ===========================================================================
# TASK 6: GridSearchCV on a Pipeline
# ===========================================================================
def task6_grid_search(X_train, y_clf_train):
    hr("TASK 6: GridSearchCV over Random Forest pipeline")
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        RandomForestClassifier(random_state=SEED),
    )

    param_grid = {
        "randomforestclassifier__n_estimators": [50, 100, 200],
        "randomforestclassifier__max_depth": [5, 10, None],
        "randomforestclassifier__min_samples_leaf": [1, 5],
    }
    n_configs = 1
    for v in param_grid.values():
        n_configs *= len(v)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    total_fits = n_configs * cv.get_n_splits()
    print(f"Parameter grid has {n_configs} configurations x {cv.get_n_splits()} folds "
          f"= {total_fits} total model fits.")

    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    t0 = time.time()
    grid.fit(X_train, y_clf_train)  # unscaled - pipeline handles imputation + scaling
    elapsed = time.time() - t0
    print(f"GridSearchCV finished in {elapsed:.1f}s")
    print(f"\nBest params: {grid.best_params_}")
    print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")

    return grid.best_estimator_, grid.best_params_, grid.best_score_, n_configs, total_fits


# ===========================================================================
# TASK 7: serialize best model + reload/predict demo
# ===========================================================================
def task7_serialize(best_pipeline, X_test, feature_names):
    hr("TASK 7: Serialize best model with joblib")
    model_path = os.path.join(HERE, "best_model.pkl")
    joblib.dump(best_pipeline, model_path)
    print(f"Saved best pipeline to {model_path}")

    # --- reload and predict demo ---
    loaded_model = joblib.load(model_path)
    sample_rows = X_test.iloc[:2].copy()
    preds = loaded_model.predict(sample_rows)
    probs = loaded_model.predict_proba(sample_rows)[:, 1]
    print("\nReload-and-predict demo on 2 hand-picked test rows:")
    print(sample_rows)
    print(f"Predicted classes: {preds}")
    print(f"Predicted probabilities (class=1): {probs}")
    return model_path


def main():
    cleaned_path = os.path.join(HERE, "cleaned_data.csv")
    (X, X_train, X_test, X_train_scaled, X_test_scaled,
     y_clf_train, y_clf_test) = load_and_prepare(cleaned_path)
    feature_names = X.columns

    tree_full, tr_acc_full, te_acc_full = task1_unconstrained_tree(
        X_train_scaled, X_test_scaled, y_clf_train, y_clf_test)
    tree_ctrl, tr_acc_ctrl, te_acc_ctrl, tree_ctrl_auc = task2_controlled_tree(
        X_train_scaled, X_test_scaled, y_clf_train, y_clf_test)
    gini_entropy = task3_gini_vs_entropy(
        X_train_scaled, X_test_scaled, y_clf_train, y_clf_test)

    rf, rf_train_acc, rf_test_acc, rf_test_auc, importances = task4_random_forest(
        X_train_scaled, X_test_scaled, y_clf_train, y_clf_test, feature_names)
    gb, gb_train_acc, gb_test_acc, gb_test_auc = task4a_gradient_boosting(
        X_train_scaled, X_test_scaled, y_clf_train, y_clf_test)
    lowest5, reduced_auc = task4b_feature_ablation(
        X_train_scaled, X_test_scaled, y_clf_train, y_clf_test, importances, rf_test_auc)

    cv_table = task5_cv_comparison(X_train_scaled, y_clf_train)

    best_pipeline, best_params, best_cv_score, n_configs, total_fits = task6_grid_search(
        X_train, y_clf_train)

    # ---- manual learning curve (rewritten inline to correctly use unscaled splits) ----
    hr("TASK 6b: Manual learning curve (20% -> 100% of training data)")
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    lc_rows = []
    n_train = len(X_train)
    best_rf_params = {
        k.replace("randomforestclassifier__", ""): v
        for k, v in best_pipeline.get_params().items()
        if k.startswith("randomforestclassifier__")
    }
    for f in fractions:
        n_rows = int(f * n_train)
        X_sub = X_train.iloc[:n_rows]
        y_sub = y_clf_train.iloc[:n_rows]
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            RandomForestClassifier(**best_rf_params),
        )
        pipe.fit(X_sub, y_sub)
        train_auc = roc_auc_score(y_sub, pipe.predict_proba(X_sub)[:, 1])
        test_auc = roc_auc_score(y_clf_test, pipe.predict_proba(X_test)[:, 1])
        lc_rows.append({"Training fraction": f, "Training AUC": train_auc, "Test AUC": test_auc})

    lc_table = pd.DataFrame(lc_rows)
    print(lc_table.to_string(index=False))

    model_path = task7_serialize(best_pipeline, X_test, feature_names)

    # test-set AUC of the tuned/best pipeline (fit on full unscaled X_train, predicts on unscaled X_test)
    tuned_test_auc = roc_auc_score(y_clf_test, best_pipeline.predict_proba(X_test)[:, 1])

    hr("FINAL MASTER COMPARISON TABLE (Part 2 + Part 3 models)")
    logreg = LogisticRegression(max_iter=1000, random_state=SEED)
    logreg.fit(X_train_scaled, y_clf_train)
    logreg_test_auc = roc_auc_score(y_clf_test, logreg.predict_proba(X_test_scaled)[:, 1])

    master_rows = [
        {"Model": "Logistic Regression", "CV Mean AUC": cv_table.loc[cv_table.Model == "Logistic Regression", "CV Mean AUC"].values[0],
         "CV Std AUC": cv_table.loc[cv_table.Model == "Logistic Regression", "CV Std AUC"].values[0],
         "Test AUC": logreg_test_auc},
        {"Model": "Decision Tree (max_depth=5)", "CV Mean AUC": cv_table.loc[cv_table.Model == "Decision Tree (max_depth=5)", "CV Mean AUC"].values[0],
         "CV Std AUC": cv_table.loc[cv_table.Model == "Decision Tree (max_depth=5)", "CV Std AUC"].values[0],
         "Test AUC": tree_ctrl_auc},
        {"Model": "Random Forest (n=100, depth=10)", "CV Mean AUC": cv_table.loc[cv_table.Model == "Random Forest", "CV Mean AUC"].values[0],
         "CV Std AUC": cv_table.loc[cv_table.Model == "Random Forest", "CV Std AUC"].values[0],
         "Test AUC": rf_test_auc},
        {"Model": "Gradient Boosting", "CV Mean AUC": cv_table.loc[cv_table.Model == "Gradient Boosting", "CV Mean AUC"].values[0],
         "CV Std AUC": cv_table.loc[cv_table.Model == "Gradient Boosting", "CV Std AUC"].values[0],
         "Test AUC": gb_test_auc},
        {"Model": "Tuned RF (GridSearchCV best)", "CV Mean AUC": best_cv_score,
         "CV Std AUC": np.nan, "Test AUC": tuned_test_auc},
    ]
    master_table = pd.DataFrame(master_rows)
    print(master_table.to_string(index=False))

    hr("SUMMARY")
    print(f"Unconstrained tree: train={tr_acc_full:.4f}, test={te_acc_full:.4f}, gap={tr_acc_full - te_acc_full:.4f}")
    print(f"Controlled tree:    train={tr_acc_ctrl:.4f}, test={te_acc_ctrl:.4f}, gap={tr_acc_ctrl - te_acc_ctrl:.4f}")
    print(f"Gini/Entropy test acc: {gini_entropy}")
    print(f"Random Forest: train_acc={rf_train_acc:.4f}, test_acc={rf_test_acc:.4f}, test_auc={rf_test_auc:.4f}")
    print(f"Gradient Boosting: train_acc={gb_train_acc:.4f}, test_acc={gb_test_acc:.4f}, test_auc={gb_test_auc:.4f}")
    print(f"Ablation: full_auc={rf_test_auc:.4f}, reduced_auc={reduced_auc:.4f}, dropped={lowest5}")
    print("\nCross-validated comparison:")
    print(cv_table.to_string(index=False))
    print(f"\nGridSearchCV: {n_configs} configs x 5 folds = {total_fits} fits")
    print(f"Best params: {best_params}")
    print(f"Best CV AUC: {best_cv_score:.4f}")
    print("\nLearning curve:")
    print(lc_table.to_string(index=False))
    print(f"\nModel saved at: {model_path}")


if __name__ == "__main__":
    main()

