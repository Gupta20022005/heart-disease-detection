"""Reproducible training pipeline for heart-disease detection.

Usage:  python pipeline.py --data Cardiovascular_Disease_Dataset.csv --out artifacts
Stages: load -> validate -> clean -> stratified split -> sklearn Pipelines (scale + model)
        -> hold-out eval -> 5-fold CV -> select best (by CV ROC-AUC) -> save artifacts.
"""
import argparse, json, os, random, time
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef)

SEED = 42
CONTINUOUS = ["age", "restingBP", "serumcholestrol", "maxheartrate", "oldpeak"]
EXPECTED = CONTINUOUS + ["gender", "chestpain", "fastingbloodsugar", "restingrelectro",
                         "exerciseangia", "slope", "noofmajorvessels", "target"]


# Accepted alternative spellings (data-dictionary vs. CSV) -> canonical column name
ALIASES = {"restingelectro": "restingrelectro", "exerciseangina": "exerciseangia",
           "serumcholesterol": "serumcholestrol", "restingbp": "restingBP"}


def normalise_columns(df):
    """Case-insensitive match to the canonical names, plus the known alias spellings."""
    canon = {c.lower(): c for c in EXPECTED + ["patientid"]}
    canon.update({k.lower(): v for k, v in ALIASES.items()})
    return df.rename(columns={c: canon.get(str(c).strip().lower(), c) for c in df.columns})


def load_and_validate(path_or_df):
    df = path_or_df.copy() if isinstance(path_or_df, pd.DataFrame) else pd.read_csv(path_or_df)
    df = normalise_columns(df).drop(columns=["patientid"], errors="ignore")
    missing = [c for c in EXPECTED if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing expected columns: {missing}")
    if df["target"].nunique() != 2:
        raise ValueError("`target` must be binary (0/1).")
    if df[EXPECTED].isnull().any().any():
        raise ValueError("Dataset contains missing values; impute or drop them first.")
    return df.drop_duplicates().reset_index(drop=True)


def build_pipeline(clf, features):
    """Scale continuous columns only (fit inside the pipeline => leak-free, incl. in CV)."""
    pre = ColumnTransformer([("scale", MinMaxScaler(), CONTINUOUS)], remainder="passthrough",
                            verbose_feature_names_out=False).set_output(transform="pandas")
    return Pipeline([("prep", pre), ("clf", clf)])


def get_classifiers():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver="lbfgs", C=100, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_features="sqrt", random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8,
                                 colsample_bytree=0.8, eval_metric="logloss", random_state=SEED, verbosity=0),
        "KNN": KNeighborsClassifier(n_neighbors=5, metric="minkowski", n_jobs=-1),
        "SVC": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=SEED),
    }


DEPLOY_MODEL = "XGBoost"   # model served by the Streamlit site


def run_pipeline(data, out_dir=None, test_size=0.30, verbose=True):
    random.seed(SEED); np.random.seed(SEED)
    df = load_and_validate(data)
    X, y = df.drop("target", axis=1), df["target"]
    features = X.columns.tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=SEED, stratify=y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    models, cv, rows = {}, {}, []
    for name, clf in get_classifiers().items():
        t0 = time.time()
        pipe = build_pipeline(clf, features)
        cv[name] = cross_val_score(pipe, X_tr, y_tr, cv=skf, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_tr, y_tr)
        pred, proba = pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1]
        rows.append({"Model": name, "Accuracy": accuracy_score(y_te, pred),
                     "Precision": precision_score(y_te, pred, zero_division=0),
                     "Recall": recall_score(y_te, pred, zero_division=0),
                     "F1 Score": f1_score(y_te, pred, zero_division=0),
                     "ROC-AUC": roc_auc_score(y_te, proba), "MCC": matthews_corrcoef(y_te, pred),
                     "CV ROC-AUC (mean)": cv[name].mean(), "CV ROC-AUC (std)": cv[name].std()})
        models[name] = pipe
        if verbose:
            print(f"  {name:<20} test AUC {rows[-1]['ROC-AUC']:.4f} | CV {cv[name].mean():.4f} ± {cv[name].std():.4f} ({time.time()-t0:.1f}s)")

    results = pd.DataFrame(rows).set_index("Model").sort_values("ROC-AUC", ascending=False)
    best = results["CV ROC-AUC (mean)"].idxmax()   # select on CV, report on untouched test set
    bundle = dict(models=models, X_te=X_te, y_te=y_te, results=results, cv=cv, features=features,
                  best_model=best, deploy_model=DEPLOY_MODEL, meta=dict(seed=SEED, n_rows=len(df), n_train=len(X_tr), n_test=len(X_te)))

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(bundle, os.path.join(out_dir, "model_bundle.joblib"))
        joblib.dump(models[best], os.path.join(out_dir, "best_model.pkl"))              # best by CV
        joblib.dump(models[DEPLOY_MODEL], os.path.join(out_dir, "xgboost_model.pkl"))   # served by the website
        results.round(4).to_csv(os.path.join(out_dir, "results.csv"))
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(dict(best_model=best, selected_by="5-fold CV ROC-AUC (train partition)",
                           test_metrics=results.loc[best].round(4).to_dict(), **bundle["meta"]), f, indent=2)
    return bundle


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()
    if a.data is None:  # default: first .csv in the current folder
        csvs = sorted(f for f in os.listdir(".") if f.lower().endswith(".csv"))
        if not csvs:
            raise SystemExit("No .csv found in this folder. Use --data path/to/file.csv")
        a.data = csvs[0]
    print(f"Training pipeline on {a.data} …")
    b = run_pipeline(a.data, a.out)
    print(f"\nBest model (CV ROC-AUC): {b['best_model']}\nArtifacts saved to {a.out}/")
    if b["best_model"] != b["deploy_model"]:
        print(f"Note: website serves {b['deploy_model']}, but {b['best_model']} scored higher in CV.")