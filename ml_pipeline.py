from __future__ import annotations

import hashlib
import json
import re
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


def _data_hash() -> str:
    return hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


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
    feature_bounds = {c: (float(df[c].min()), float(df[c].max())) for c in FEATURES}
    target_bounds = (float(df[TARGET].min()), float(df[TARGET].max()))
    if any(low < 0 for low, _ in feature_bounds.values()) or not 0 <= target_bounds[0] <= 100 or target_bounds[1] > 100:
        raise ValueError("Dataset contains values outside expected 0–100 academic ranges")
    df.attrs["duplicates_removed"] = duplicates
    return df


def candidate_models() -> dict[str, object]:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, max_depth=5),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42, n_estimators=120, max_depth=2, learning_rate=0.04
        ),
    }


def _final_estimator(model: object) -> object:
    if hasattr(model, "named_steps"):
        return model.named_steps.get("model", model)
    return model


def _explainability(model: object, values: dict[str, float]) -> dict:
    estimator = _final_estimator(model)
    if hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float).reshape(-1)
        intercept = float(np.asarray(estimator.intercept_).reshape(-1)[0])
        drivers = []
        for feature, coefficient in zip(FEATURES, coefficients):
            contribution = float(coefficient * values[feature])
            drivers.append({
                "feature": feature,
                "coefficient": float(coefficient),
                "input_value": float(values[feature]),
                "contribution": contribution,
            })
        return {
            "type": "linear_coefficients",
            "intercept": intercept,
            "features": drivers,
            "note": "For the selected linear model, contribution is coefficient × input. Contributions explain the raw linear prediction; the displayed prediction is bounded to 0–100 by the application.",
        }
    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_, dtype=float).reshape(-1)
        return {
            "type": "tree_feature_importance",
            "intercept": None,
            "features": [
                {"feature": f, "importance": float(v), "input_value": float(values[f])}
                for f, v in zip(FEATURES, importances)
            ],
            "note": "Feature importance indicates relative influence used by the tree ensemble; it is not a causal effect.",
        }
    return {"type": "unavailable", "intercept": None, "features": [], "note": "Explainability metadata is not available for the selected estimator."}


def train_and_evaluate() -> dict:
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    test_size = max(1, int(round(len(df) * 0.20)))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    cv = KFold(n_splits=min(5, len(df)), shuffle=True, random_state=42)
    results: list[ModelResult] = []
    holdout_predictions: dict[str, np.ndarray] = {}

    for name, estimator in candidate_models().items():
        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_test)
        holdout_predictions[name] = np.asarray(pred, dtype=float)
        mae = float(mean_absolute_error(y_test, pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        r2 = float(r2_score(y_test, pred))
        cv_mae = -float(
            cross_val_score(estimator, X, y, scoring="neg_mean_absolute_error", cv=cv).mean()
        )
        cv_r2 = float(cross_val_score(estimator, X, y, scoring="r2", cv=cv).mean())
        results.append(ModelResult(name, mae, rmse, r2, cv_mae, cv_r2))

    best = sorted(results, key=lambda r: (r.cv_mae, -r.cv_r2))[0]
    final_model = candidate_models()[best.name]
    final_model.fit(X, y)

    hash_value = _data_hash()
    model_version = f"{_slug(best.name)}-{hash_value[:10]}"
    selected_holdout = holdout_predictions[best.name]
    holdout_rows = []
    for idx, actual, predicted in zip(X_test.index.tolist(), y_test.tolist(), selected_holdout.tolist()):
        holdout_rows.append({
            "source_row": int(idx) + 2,
            "study_hours": float(df.loc[idx, "study_hours"]),
            "attendance": float(df.loc[idx, "attendance"]),
            "previous_marks": float(df.loc[idx, "previous_marks"]),
            "actual_final_marks": float(actual),
            "predicted_final_marks": round(float(predicted), 4),
            "residual": round(float(actual - predicted), 4),
        })

    numeric = df[FEATURES + [TARGET]]
    correlations = numeric.corr(numeric_only=True)[TARGET].drop(TARGET).to_dict()
    feature_summary = {}
    for column in FEATURES + [TARGET]:
        feature_summary[column] = {
            "min": float(df[column].min()),
            "max": float(df[column].max()),
            "mean": float(df[column].mean()),
            "median": float(df[column].median()),
            "std": float(df[column].std(ddof=0)),
        }

    metadata = {
        "model_name": best.name,
        "model_family": type(_final_estimator(final_model)).__name__,
        "model_version": model_version,
        "training_data_hash": hash_value,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features": FEATURES,
        "target": TARGET,
        "dataset_rows": int(len(df)),
        "dataset_rows_after_cleaning": int(len(df)),
        "duplicates_removed": int(df.attrs.get("duplicates_removed", 0)),
        "holdout_size": int(len(X_test)),
        "cv_folds": int(min(5, len(df))),
        "selection_method": "Lowest cross-validation MAE; cross-validation R² used as tie-breaker.",
        "metrics": [asdict(r) for r in results],
        "selected_metrics": asdict(best),
        "holdout_rows": holdout_rows,
        "correlations_with_target": {k: float(v) for k, v in correlations.items()},
        "feature_summary": feature_summary,
        "target_summary": {
            "mean": float(df[TARGET].mean()),
            "median": float(df[TARGET].median()),
            "std": float(df[TARGET].std(ddof=0)),
            "q1": float(df[TARGET].quantile(0.25)),
            "q3": float(df[TARGET].quantile(0.75)),
        },
        "explainability": _explainability(final_model, {f: float(df[f].mean()) for f in FEATURES}),
        "limitations": [
            f"The supplied dataset contains only {len(df)} records.",
            "Evaluation metrics describe this dataset and validation procedure; they are not a guarantee of real-world prediction accuracy.",
            "Predictions outside the training feature ranges are extrapolations and should be treated cautiously.",
            "The current dataset contains no subject-wise, semester-wise, demographic, assignment, or internal-assessment fields, so those analytics are not fabricated by the application.",
        ],
        "data_availability": {
            "subject_level": False,
            "semester_level": False,
            "assignment_level": False,
            "internal_assessment": False,
        },
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": final_model, "metadata": metadata}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    evaluation_lines = [
        "Student Performance Prediction System — Model Evaluation",
        "=" * 64,
        "",
        f"Training records       : {metadata['dataset_rows']}",
        f"Holdout test records   : {metadata['holdout_size']}",
        f"Cross-validation folds : {metadata['cv_folds']}",
        f"Selected model         : {metadata['model_name']}",
        f"Model version          : {metadata['model_version']}",
        f"Training data SHA-256  : {metadata['training_data_hash']}",
        f"Selection rule         : {metadata['selection_method']}",
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
        "Observed holdout rows",
        "---------------------",
        *[
            f"CSV row {r['source_row']}: actual={r['actual_final_marks']:.2f}, predicted={r['predicted_final_marks']:.2f}, residual={r['residual']:.2f}"
            for r in holdout_rows
        ],
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
