from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["study_hours", "attendance", "previous_marks"]
TARGET = "final_marks"
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "student_data.csv"
MODEL_PATH = ROOT / "models" / "student_model.pkl"
METRICS_PATH = ROOT / "outputs" / "model_metrics.json"
EVALUATION_PATH = ROOT / "outputs" / "evaluation.txt"


@dataclass
class ModelResult:
    name: str
    mae: float
    rmse: float
    r2: float
    cv_mae: float
    cv_r2: float


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    df = df[FEATURES + [TARGET]].copy()
    for c in FEATURES + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    if df.empty:
        raise ValueError("Dataset is empty")
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values")
    duplicates = int(df.duplicated().sum())
    if duplicates:
        df = df.drop_duplicates().copy()
    bounds = {c: (float(df[c].min()), float(df[c].max())) for c in FEATURES}
    target_bounds = (float(df[TARGET].min()), float(df[TARGET].max()))
    if any(low < 0 for low, _ in bounds.values()) or not 0 <= target_bounds[0] <= 100 or target_bounds[1] > 100:
        raise ValueError("Dataset contains values outside expected 0–100 academic ranges")
    df.attrs["duplicates_removed"] = duplicates
    return df


def candidate_models() -> dict[str, object]:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, max_depth=5),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=120, max_depth=2, learning_rate=0.04),
    }


def train_and_evaluate() -> dict:
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    test_size = max(1, int(round(len(df) * 0.20)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    cv = KFold(n_splits=min(5, len(df)), shuffle=True, random_state=42)
    results = []
    for name, estimator in candidate_models().items():
        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_test)
        mae = float(mean_absolute_error(y_test, pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        r2 = float(r2_score(y_test, pred))
        cv_mae = -float(cross_val_score(estimator, X, y, scoring="neg_mean_absolute_error", cv=cv).mean())
        cv_r2 = float(cross_val_score(estimator, X, y, scoring="r2", cv=cv).mean())
        results.append(ModelResult(name, mae, rmse, r2, cv_mae, cv_r2))

    best = sorted(results, key=lambda r: (r.cv_mae, -r.cv_r2))[0]
    final_model = candidate_models()[best.name]
    final_model.fit(X, y)

    metadata = {
        "model_name": best.name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features": FEATURES,
        "target": TARGET,
        "dataset_rows": int(len(df)),
        "dataset_rows_after_cleaning": int(len(df)),
        "duplicates_removed": int(df.attrs.get("duplicates_removed", 0)),
        "holdout_size": int(len(X_test)),
        "cv_folds": int(min(5, len(df))),
        "selection_method": "Lowest 5-fold cross-validation MAE; CV R² used as tie-breaker.",
        "metrics": [asdict(r) for r in results],
        "selected_metrics": asdict(best),
        "feature_summary": {
            "study_hours": {"min": float(df.study_hours.min()), "max": float(df.study_hours.max()), "mean": float(df.study_hours.mean())},
            "attendance": {"min": float(df.attendance.min()), "max": float(df.attendance.max()), "mean": float(df.attendance.mean())},
            "previous_marks": {"min": float(df.previous_marks.min()), "max": float(df.previous_marks.max()), "mean": float(df.previous_marks.mean())},
            "final_marks": {"min": float(df.final_marks.min()), "max": float(df.final_marks.max()), "mean": float(df.final_marks.mean())},
        },
        "limitations": [
            "The supplied dataset contains only 20 records.",
            "The evaluation results describe this dataset and validation procedure; they are not a guarantee of real-world prediction accuracy.",
            "Predictions outside the training feature ranges are extrapolations and should be treated cautiously.",
        ],
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": final_model, "metadata": metadata}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    evaluation_lines = [
        "Student Performance Prediction System — Model Evaluation",
        "=" * 60,
        "",
        f"Training records: {metadata['dataset_rows']}",
        f"Holdout test records: {metadata['holdout_size']}",
        f"Cross-validation folds: {metadata['cv_folds']}",
        f"Selected model: {metadata['model_name']}",
        f"Selection rule: {metadata['selection_method']}",
        "",
        "Selected model metrics",
        "-----------------------",
        f"MAE      : {best.mae:.4f}",
        f"RMSE     : {best.rmse:.4f}",
        f"R²       : {best.r2:.4f}",
        f"CV MAE   : {best.cv_mae:.4f}",
        f"CV R²    : {best.cv_r2:.4f}",
        "",
        "Model comparison",
        "----------------",
    ]
    evaluation_lines.extend(
        f"{r.name:<18} | MAE {r.mae:.4f} | RMSE {r.rmse:.4f} | R² {r.r2:.4f} | CV MAE {r.cv_mae:.4f} | CV R² {r.cv_r2:.4f}"
        for r in results
    )
    evaluation_lines.extend([
        "",
        "Limitations",
        "-----------",
        *[f"- {item}" for item in metadata["limitations"]],
    ])
    EVALUATION_PATH.write_text("\n".join(evaluation_lines) + "\n", encoding="utf-8")
    return metadata


def load_model_bundle() -> tuple[object, dict]:
    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle["model"], bundle.get("metadata", {})
    return bundle, {}


if __name__ == "__main__":
    print(json.dumps(train_and_evaluate(), indent=2))
