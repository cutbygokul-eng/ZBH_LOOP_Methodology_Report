"""
Machine Failure Prediction - single script
Team ZBH LOOP - DATA DIVERGENCE, Track 1

USAGE
  Train (builds machine_failure_model.joblib from train.csv):
      python model.py train            # full run with cross-validation scores
      python model.py train --quick    # fast run, skips cross-validation

  Predict (runs the saved model on any CSV with the 6 required columns):
      python model.py predict your_file.csv

LIBRARIES: only scikit-learn, pandas, numpy (joblib is installed with scikit-learn).
All preprocessing (missing-value filling, scaling, feature engineering) is inside the saved model.
"""
import sys
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (precision_recall_curve, f1_score, average_precision_score,
                              precision_score, recall_score)

MODEL_FILE = "machine_failure_model.joblib"
RAW_FEATURES = ["Type", "Air temperature [K]", "Process temperature [K]",
                 "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
TARGET = "Machine failure"


AIR, PROC, RPM, TQ, WEAR = ("Air temperature [K]", "Process temperature [K]",
                            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]")
NUM_COLS = [AIR, PROC, RPM, TQ, WEAR]
CAT_COLS = ["Type"]


def build_pipeline():
    """Missing-value filling + scaling + 3 engineered features + blend of two models.
    Uses only standard scikit-learn parts, so the saved file loads with plain joblib."""
    median = lambda: SimpleImputer(strategy="median")
    preprocess = ColumnTransformer([
        ("num", Pipeline([("impute", median()), ("scale", StandardScaler())]), NUM_COLS),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT_COLS),
        # process temperature minus air temperature
        ("temp_gap", Pipeline([("impute", median()), ("gap", FunctionTransformer(np.diff))]), [AIR, PROC]),
        # torque x speed, torque x tool wear
        ("torque_x_speed", Pipeline([("impute", median()),
                                      ("mul", PolynomialFeatures(2, interaction_only=True, include_bias=False))]), [TQ, RPM]),
        ("torque_x_wear", Pipeline([("impute", median()),
                                     ("mul", PolynomialFeatures(2, interaction_only=True, include_bias=False))]), [TQ, WEAR]),
    ])
    gb = GradientBoostingClassifier(random_state=42)
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    blend = VotingClassifier(estimators=[("gb", gb), ("rf", rf)], voting="soft")
    return Pipeline([("prep", preprocess), ("clf", blend)])


def train(quick=False):
    print("scikit-learn version:", sklearn.__version__)

    df = pd.read_csv("train.csv")
    if df.shape[1] < 7 or TARGET not in df.columns:
        sys.exit(f"'{TARGET}' column not found. Columns seen: {list(df.columns)}. "
                  f"Make sure train.csv is the original training file.")

    X = df[RAW_FEATURES]
    y = df[TARGET].astype(int)
    pipe = build_pipeline()

    best_threshold = 0.6526  # fallback used in --quick mode (found by cross-validation)
    if not quick:
        print("Running 5-fold cross-validation (this takes a few minutes)...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        proba_cv = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]

        precisions, recalls, thresholds = precision_recall_curve(y, proba_cv)
        f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
        best_idx = np.nanargmax(f1s[:-1])
        best_threshold = float(thresholds[best_idx])

        pr_auc = average_precision_score(y, proba_cv)
        preds_cv = (proba_cv >= best_threshold).astype(int)
        print(f"PR-AUC:    {pr_auc:.4f}")
        print(f"F1:        {f1_score(y, preds_cv):.4f}")
        print(f"Precision: {precision_score(y, preds_cv):.4f}")
        print(f"Recall:    {recall_score(y, preds_cv):.4f}")
        print(f"Best threshold: {best_threshold:.4f}")
    else:
        print("Quick mode: skipping cross-validation, using saved threshold", best_threshold)

    pipe.fit(X, y)

    artifact = {
        "model": pipe,
        "threshold": best_threshold,
        "model_name": "Blended GradientBoosting + RandomForest",
        "features": RAW_FEATURES,
        "target": TARGET,
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(artifact, MODEL_FILE)
    print(f"Saved {MODEL_FILE}")

    reloaded = joblib.load(MODEL_FILE)
    sample = pd.DataFrame([{"Type": "L", "Air temperature [K]": 298.0, "Process temperature [K]": 308.0,
                             "Rotational speed [rpm]": 1500, "Torque [Nm]": 40.0, "Tool wear [min]": 100}])
    proba = reloaded["model"].predict_proba(sample[RAW_FEATURES])[:, 1]
    print("Self-check probability:", proba.round(4))


def predict(input_file):
    print("scikit-learn version:", sklearn.__version__)
    artifact = joblib.load(MODEL_FILE)
    model = artifact["model"]
    threshold = float(artifact["threshold"])
    print(f"Threshold used: {threshold:.4f}")

    df = pd.read_csv(input_file)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        sys.exit(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    work = df.copy()
    for col in RAW_FEATURES[1:]:
        bad = pd.to_numeric(work[col], errors="coerce").isna() & work[col].notna()
        if bad.any():
            print(f"Warning: {bad.sum()} invalid value(s) in '{col}', treated as missing.")
        work[col] = pd.to_numeric(work[col], errors="coerce")

    proba = model.predict_proba(work[RAW_FEATURES])[:, 1]
    out = df.copy()
    out["failure_probability"] = proba.round(4)
    out["predicted_failure"] = (proba >= threshold).astype(int)
    out["prediction_label"] = out["predicted_failure"].map({0: "Healthy", 1: "Potential failure"})

    out_name = f"predictions_{input_file.split('/')[-1]}"
    out.to_csv(out_name, index=False)
    print(out.to_string())
    print(f"\nDone: {len(out)} rows predicted. Saved to {out_name}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "predict"):
        sys.exit("Usage:\n  python model.py train [--quick]\n  python model.py predict <file.csv>")

    mode = sys.argv[1]
    if mode == "train":
        train(quick="--quick" in sys.argv)
    else:
        if len(sys.argv) < 3:
            sys.exit("Usage: python model.py predict <file.csv>")
        predict(sys.argv[2])
