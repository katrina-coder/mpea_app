"""
retrain_models.py
─────────────────
Run this ONCE to re-save all Random Forest models using YOUR installed
version of scikit-learn.  This fixes the sklearn pickle incompatibility
error that occurs when the original .joblib files (saved with sklearn 0.24)
are loaded by sklearn 1.x.

Usage:
    python retrain_models.py

Output:
    Overwrites all .joblib files inside the  models/  folder.
    The PyTorch generator (.pt) file is NOT touched — it is version-safe.
"""

import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from joblib import dump

print(f"scikit-learn version: {sklearn.__version__}")
print("Re-training all Random Forest models …\n")

DATASET_PATH = os.path.join("models", "MPEA_parsed_dataset.xlsx")
MODEL_DIR    = "models"

# ── Load dataset ──────────────────────────────────────────────────────────────
print(f"Loading dataset from: {DATASET_PATH}")
df       = pd.read_excel(DATASET_PATH)
arr      = df.to_numpy()
features = arr[:, 14:53].astype(float)   # 32 element ratios + 7 process one-hots

# ── Mechanical property regressors ────────────────────────────────────────────
# Column indices in the dataset (matches original training code)
MECHANICAL = {
    "hardness":   2,
    "yield":      3,
    "tensile":    4,
    "elongation": 5,
}

REGRESSOR_FILENAME = {
    "hardness":   "hardness_regressor.joblib",
    "yield":      "yield_regressor.joblib",
    "tensile":    "tensile_regressor.joblib",
    "elongation": "elongation_regressor.joblib",
}

for prop, col_idx in MECHANICAL.items():
    target      = arr[:, col_idx].astype(float)
    valid       = ~np.isnan(target)
    X, y        = features[valid], target[valid]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=49
    )

    rf = RandomForestRegressor(
        n_estimators=100, max_depth=50, random_state=0, oob_score=True
    )
    rf.fit(X_train, y_train)
    r2 = r2_score(y_test, rf.predict(X_test))

    out_path = os.path.join(MODEL_DIR, REGRESSOR_FILENAME[prop])
    dump(rf, out_path)
    print(f"  ✓  {prop:12s}  R² = {r2:.4f}   →  saved to {out_path}")

# ── Phase classifiers ─────────────────────────────────────────────────────────
# Column indices in the dataset (matches original training code)
PHASES = {
    "FCC": 10,
    "BCC": 11,
    "HCP": 12,
    "IM":  13,
}

for phase, col_idx in PHASES.items():
    target      = arr[:, col_idx].astype(float)
    valid       = ~np.isnan(target)
    X, y        = features[valid], target[valid]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=14
    )

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=50, random_state=4, oob_score=True
    )
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))

    out_path = os.path.join(MODEL_DIR, f"{phase}_classifier.joblib")
    dump(rf, out_path)
    print(f"  ✓  {phase:12s}  Accuracy = {acc:.4f}   →  saved to {out_path}")

print("\n✅  All models re-saved successfully with your scikit-learn version.")
print("    You can now run:  streamlit run app.py")
