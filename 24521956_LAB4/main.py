import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier as SklearnDT
from sklearn.ensemble import RandomForestClassifier as SklearnRF

from decision_tree import DecisionTreeClassifier
from random_forest import RandomForestClassifier
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

red = pd.read_csv(DATA_DIR / "winequality-red.csv", sep=";")
white = pd.read_csv(DATA_DIR / "winequality-white.csv", sep=";")

red["type"] = 0
white["type"] = 1
df = pd.concat([red, white], ignore_index=True)

X = df.drop(columns=["quality"]).values.astype(np.float64)
y = df["quality"].values

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=" * 55)
print("  Wine Quality – Decision Tree & Random Forest")
print("=" * 55)
print(f"\nDataset  : {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes  : {len(np.unique(y))}  (quality scores re-encoded 0-based)")
print(f"Train/Test split: {len(X_train)} / {len(X_test)}\n")


print("── Assignment 1: Custom Decision Tree ──")

dt_custom = DecisionTreeClassifier(max_depth=100, min_samples_split=5, min_samples_leaf=2)
dt_custom.fit(X_train, y_train)
y_pred_dt = dt_custom.predict(X_test)

score_dt = f1_score(y_test, y_pred_dt, average="macro")
print(f"  Macro F1-score: {score_dt:.4f}\n")

print("── Assignment 2: Custom Random Forest ──")

rf_custom = RandomForestClassifier(
    n_estimators=50,
    max_depth=100,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
)
rf_custom.fit(X_train, y_train)
y_pred_rf = rf_custom.predict(X_test)

score_rf = f1_score(y_test, y_pred_rf, average="macro")
print(f"  Macro F1-score: {score_rf:.4f}\n")

print("── Assignment 3: scikit-learn Decision Tree ──")

dt_sk = SklearnDT(max_depth=100, min_samples_split=5, min_samples_leaf=2, random_state=42)
dt_sk.fit(X_train, y_train)
y_pred_dt_sk = dt_sk.predict(X_test)

score_dt_sk = f1_score(y_test, y_pred_dt_sk, average="macro")
print(f"  Macro F1-score: {score_dt_sk:.4f}\n")

print("── Assignment 3: scikit-learn Random Forest ──")

rf_sk = SklearnRF(
    n_estimators=50,
    max_depth=100,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
)
rf_sk.fit(X_train, y_train)
y_pred_rf_sk = rf_sk.predict(X_test)

score_rf_sk = f1_score(y_test, y_pred_rf_sk, average="macro")
print(f"  Macro F1-score: {score_rf_sk:.4f}\n")
