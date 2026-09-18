from __future__ import annotations

import math
from typing import Any

import pandas as pd

from ml_pipeline import FEATURES


def classify_prediction(prediction: float) -> str:
    if prediction >= 75:
        return "High (75–100)"
    if prediction >= 60:
        return "Moderate (60–74.9)"
    if prediction >= 45:
        return "Needs support (45–59.9)"
    return "Low (<45)"


def validate_features(payload: dict) -> dict[str, float]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    clean: dict[str, float] = {}
    bounds = {"study_hours": (0, 24), "attendance": (0, 100), "previous_marks": (0, 100)}
    for key, (lo, hi) in bounds.items():
        if key not in payload:
            raise ValueError(f"Missing required field: {key}")
        try:
            value = float(payload[key])
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number")
        if not math.isfinite(value) or value < lo or value > hi:
            raise ValueError(f"{key} must be between {lo} and {hi}")
        clean[key] = value
    return clean


def guidance_for(features: dict[str, float], prediction: float, metadata: dict) -> list[str]:
    guidance = []
    summary = metadata.get("feature_summary", {})
    for key, label, unit in (
        ("study_hours", "Study hours", " hours/week"),
        ("attendance", "Attendance", "%"),
        ("previous_marks", "Previous marks", "%"),
    ):
        mean = summary.get(key, {}).get("mean")
        if mean is None:
            continue
        relation = "below" if features[key] < mean else "at or above"
        guidance.append(f"{label} is {relation} the training-data average of {mean:.2f}{unit}.")
    target_mean = summary.get("final_marks", {}).get("mean")
    if target_mean is not None:
        relation = "below" if prediction < target_mean else "at or above"
        guidance.append(f"Predicted final marks are {relation} the training-data average of {target_mean:.2f}%.")
    return guidance


def support_indicator(features: dict[str, float], prediction: float, metadata: dict) -> tuple[str, list[str], int]:
    """Create a transparent, data-derived support indicator.

    This is not a probability. Each feature is compared with the observed training-data
    mean and normalized by the observed distance from its training minimum.
    """
    summary = metadata.get("feature_summary", {})
    shortfalls: list[float] = []
    recs: list[str] = []
    feature_labels = {
        "study_hours": "study hours",
        "attendance": "attendance",
        "previous_marks": "previous marks",
    }
    for key in FEATURES:
        info = summary.get(key, {})
        mean, low = info.get("mean"), info.get("min")
        if mean is None or low is None:
            continue
        if features[key] < mean:
            denom = max(float(mean) - float(low), 1e-9)
            shortfall = min(1.0, (float(mean) - features[key]) / denom)
            shortfalls.append(shortfall)
            recs.append(f"{feature_labels[key].capitalize()} are below the training-data average of {float(mean):.2f}.")
        else:
            shortfalls.append(0.0)

    target = summary.get("final_marks", {})
    target_mean = target.get("mean")
    if target_mean is not None and prediction < target_mean:
        denom = max(float(target_mean) - float(target.get("min", 0)), 1e-9)
        shortfalls.append(min(1.0, (float(target_mean) - prediction) / denom))
        recs.append(f"Predicted marks are below the training-data average of {float(target_mean):.2f}%.")
    elif target_mean is not None:
        shortfalls.append(0.0)

    average_shortfall = sum(shortfalls) / len(shortfalls) if shortfalls else 0.0
    score = int(round(100 * (1 - average_shortfall)))
    if score >= 80:
        level = "Lower support need"
    elif score >= 60:
        level = "Review recommended"
    else:
        level = "Higher support need"

    if not recs:
        recs.append("The supplied inputs are at or above the training-data averages used for this support indicator.")
    return level, recs, max(0, min(100, score))


def prediction_scope(features: dict[str, float], metadata: dict) -> list[str]:
    notes = []
    summary = metadata.get("feature_summary", {})
    for key, label in (
        ("study_hours", "Study hours"),
        ("attendance", "Attendance"),
        ("previous_marks", "Previous marks"),
    ):
        low = summary.get(key, {}).get("min")
        high = summary.get(key, {}).get("max")
        value = features[key]
        if low is not None and high is not None and not (low <= value <= high):
            notes.append(f"{label} ({value:g}) is outside the training range {low:g}–{high:g}.")
    return notes


def explain_features(model: object, features: dict[str, float], metadata: dict) -> dict:
    explanation = metadata.get("explainability", {})
    items = []
    feature_rows = explanation.get("features", [])
    if explanation.get("type") == "linear_coefficients":
        for row in feature_rows:
            feature = row["feature"]
            coefficient = float(row.get("coefficient", 0))
            value = float(features[feature])
            contribution = coefficient * value
            items.append({
                "feature": feature,
                "input_value": value,
                "coefficient": coefficient,
                "contribution": contribution,
                "direction": "positive" if coefficient >= 0 else "negative",
            })
        items.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    elif explanation.get("type") == "tree_feature_importance":
        for row in feature_rows:
            feature = row["feature"]
            items.append({
                "feature": feature,
                "input_value": float(features[feature]),
                "importance": float(row.get("importance", 0)),
            })
        items.sort(key=lambda item: item.get("importance", 0), reverse=True)
    return {
        "type": explanation.get("type", "unavailable"),
        "intercept": explanation.get("intercept"),
        "features": items,
        "note": explanation.get("note", "No explanation metadata is available."),
    }


def build_prediction(features: dict[str, float], model: object, metadata: dict) -> dict[str, Any]:
    frame = pd.DataFrame([features], columns=FEATURES)
    raw_prediction = float(model.predict(frame)[0])
    prediction = max(0.0, min(100.0, raw_prediction))
    band = classify_prediction(prediction)
    support_level, recommendations, support_score = support_indicator(features, prediction, metadata)
    rmse = float(metadata.get("selected_metrics", {}).get("rmse", 0) or 0)
    low = round(max(0.0, prediction - rmse), 2) if rmse else None
    high = round(min(100.0, prediction + rmse), 2) if rmse else None
    return {
        **features,
        "raw_predicted_marks": round(raw_prediction, 4),
        "predicted_marks": round(prediction, 2),
        "performance_band": band,
        "risk_level": support_level,
        "risk_score": support_score,
        "recommendations": recommendations,
        "prediction_low": low,
        "prediction_high": high,
        "guidance": guidance_for(features, prediction, metadata),
        "training_range_notes": prediction_scope(features, metadata),
        "explainability": explain_features(model, features, metadata),
        "model_name": metadata.get("model_name", "Unknown"),
        "model_version": metadata.get("model_version"),
        "model_mae": round(float(metadata.get("selected_metrics", {}).get("mae", 0)), 3),
        "model_rmse": round(rmse, 3) if rmse else None,
    }
