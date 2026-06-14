#!/usr/bin/env python3
"""Train XGBoost multi-label bias classifier from JSONL + 3 CSV datasets."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.ml.bias_classifier import save_model
from backend.ml.dataset_builder import build_training_dataset
from backend.ml.features import BIAS_LABEL_KEYS, FEATURE_COLUMNS


def main() -> int:
    try:
        from sklearn.metrics import f1_score
        from sklearn.model_selection import train_test_split
        from sklearn.multioutput import MultiOutputClassifier
        from xgboost import XGBClassifier
    except ImportError:
        print("[ERROR] Install ML dependencies: pip install xgboost scikit-learn joblib")
        return 1

    print("Building training dataset (JSONL + 3 CSVs)...")
    x_rows, y_rows, counts = build_training_dataset()
    print("Source counts:", counts)
    print(f"Total examples: {len(x_rows)}")

    if len(x_rows) < 50:
        print("[ERROR] Not enough training data.")
        return 1

    x = np.asarray(x_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.int32)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

    base = XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model = MultiOutputClassifier(base)
    print("Training XGBoost multi-label classifier...")
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    per_label_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)

    thresholds: dict[str, float] = {}
    estimators = model.estimators_
    for idx, key in enumerate(BIAS_LABEL_KEYS):
        if hasattr(estimators[idx], "predict_proba"):
            proba = estimators[idx].predict_proba(x_test)[:, -1]
            thresholds[key] = float(np.clip(np.percentile(proba, 35), 0.35, 0.65))
        else:
            thresholds[key] = 0.45
        print(f"  {key:24} f1={per_label_f1[idx]:.3f} threshold={thresholds[key]:.2f}")

    metrics = {
        "macro_f1": macro_f1,
        "per_label_f1": {BIAS_LABEL_KEYS[i]: float(per_label_f1[i]) for i in range(len(BIAS_LABEL_KEYS))},
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "feature_count": len(FEATURE_COLUMNS),
        "source_counts": counts,
    }

    model_path = save_model(model, thresholds, metrics)
    print(f"\n[SUCCESS] Model saved to {model_path}")
    print(f"Macro F1: {macro_f1:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
