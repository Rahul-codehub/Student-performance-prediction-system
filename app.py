from __future__ import annotations

import math
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

from database import add_prediction, add_student, attach_student_to_prediction, csv_text, dashboard_stats, delete_student, get_prediction, init_db, list_predictions, list_students
from ml_pipeline import FEATURES, load_data, load_model_bundle, train_and_evaluate

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
init_db()


def load_bundle_safe():
    try:
        return load_model_bundle()
    except FileNotFoundError:
        train_and_evaluate()
        return load_model_bundle()


def classify_prediction(prediction: float) -> str:
    if prediction >= 75:
        return "High (75–100)"
    if prediction >= 60:
        return "Moderate (60–74.9)"
    if prediction >= 45:
        return "Needs support (45–59.9)"
    return "Low (<45)"


def guidance_for(features: dict, prediction: float, metadata: dict) -> list[str]:
    guidance = []
    summary = metadata.get("feature_summary", {})
    for key, label, unit in (
        ("study_hours", "Study hours", "hours/week"),
        ("attendance", "Attendance", "%"),
        ("previous_marks", "Previous marks", "%"),
    ):
        mean = summary.get(key, {}).get("mean")
        if mean is None:
            continue
        if features[key] < mean:
            guidance.append(f"{label} is below the training-data average of {mean:.2f}{unit}.")
        else:
            guidance.append(f"{label} is at or above the training-data average of {mean:.2f}{unit}.")
    if prediction < 60:
        guidance.append("The predicted mark is below 60, so this case may warrant an academic review.")
    else:
        guidance.append("The predicted mark is 60 or above based on the current model input.")
    return guidance


def validate_features(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    clean = {}
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


def risk_and_recommendations(study_hours: float, attendance: float, previous_marks: float, prediction: float) -> tuple[str, list[str], int]:
    score = 100.0
    score -= max(0, 75 - attendance) * 0.55
    score -= max(0, 6 - study_hours) * 3.2
    score -= max(0, 60 - previous_marks) * 0.35
    score -= max(0, 60 - prediction) * 0.6
    score = max(0, min(100, score))
    if prediction >= 75 and attendance >= 75:
        level = "Excellent"
    elif prediction >= 60:
        level = "On Track"
    elif prediction >= 45:
        level = "Needs Support"
    else:
        level = "At Risk"
    recs = []
    if attendance < 75:
        recs.append("Raise attendance above 75% with a consistent weekly attendance plan.")
    if study_hours < 6:
        recs.append("Target at least 6 focused study hours per week and track completion.")
    if previous_marks < 60:
        recs.append("Revisit weak topics from the previous assessment and use practice tests.")
    if prediction < 60:
        recs.append("Schedule faculty support and a short weekly progress review.")
    if not recs:
        recs.append("Maintain current habits and add periodic mock tests to protect performance.")
    return level, recs, round(score)


def prediction_scope(features: dict, metadata: dict) -> list[str]:
    notes = []
    summary = metadata.get("feature_summary", {})
    for key, label in (("study_hours", "Study hours"), ("attendance", "Attendance"), ("previous_marks", "Previous marks")):
        low = summary.get(key, {}).get("min")
        high = summary.get(key, {}).get("max")
        value = features[key]
        if low is not None and high is not None and not (low <= value <= high):
            notes.append(f"{label} ({value:g}) is outside the training range {low:g}–{high:g}.")
    return notes


def predict_one(features: dict) -> dict:
    model, metadata = load_bundle_safe()
    import pandas as pd
    frame = pd.DataFrame([features], columns=FEATURES)
    prediction = float(model.predict(frame)[0])
    prediction = max(0.0, min(100.0, prediction))
    band = classify_prediction(prediction)
    risk_level, recommendations, risk_score = risk_and_recommendations(**features, prediction=prediction)
    rmse = float(metadata.get("selected_metrics", {}).get("rmse", 0) or 0)
    low = round(max(0.0, prediction - rmse), 2) if rmse else None
    high = round(min(100.0, prediction + rmse), 2) if rmse else None
    return {
        **features,
        "predicted_marks": round(prediction, 2),
        "performance_band": band,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "recommendations": recommendations,
        "prediction_low": low,
        "prediction_high": high,
        "guidance": guidance_for(features, prediction, metadata),
        "training_range_notes": prediction_scope(features, metadata),
        "model_name": metadata.get("model_name", "Unknown"),
        "model_mae": round(float(metadata.get("selected_metrics", {}).get("mae", 0)), 3),
        "model_rmse": round(rmse, 3) if rmse else None,
    }


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    model, meta = load_bundle_safe()
    return jsonify({"status": "healthy", "model": meta.get("model_name", type(model).__name__), "dataset_rows": meta.get("dataset_rows", 0)})


@app.get("/api/dashboard")
def dashboard():
    _, meta = load_bundle_safe()
    ds = load_data()
    stats = dashboard_stats()
    stats.update(
        {
            "dataset": {
                "rows": len(ds),
                "columns": list(ds.columns),
                "mean_final_marks": round(float(ds.final_marks.mean()), 2),
                "median_final_marks": round(float(ds.final_marks.median()), 2),
                "missing_values": int(ds.isna().sum().sum()),
                "duplicates": int(ds.duplicated().sum()),
                "feature_ranges": meta.get("feature_summary", {}),
            },
            "model": {
                "name": meta.get("model_name"),
                "trained_at": meta.get("trained_at"),
                "selected_metrics": meta.get("selected_metrics", {}),
                "all_models": meta.get("metrics", []),
                "dataset_rows": meta.get("dataset_rows"),
                "holdout_size": meta.get("holdout_size"),
                "cv_folds": meta.get("cv_folds"),
                "selection_method": meta.get("selection_method"),
                "limitations": meta.get("limitations", []),
            },
        }
    )
    return jsonify(stats)


@app.get("/api/history")
def history():
    q = request.args.get("q", "").strip()
    return jsonify({"predictions": list_predictions(q)})


@app.get("/api/data")
def data_view():
    ds = load_data()
    _, meta = load_bundle_safe()
    return jsonify({
        "columns": list(ds.columns),
        "rows": ds.to_dict(orient="records"),
        "summary": meta.get("feature_summary", {}),
        "missing_values": int(ds.isna().sum().sum()),
        "duplicate_rows": int(ds.duplicated().sum()),
        "duplicates_removed": meta.get("duplicates_removed", 0),
    })


@app.get("/api/export/<table>.csv")
def export_csv(table: str):
    try:
        body = csv_text(table)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    return Response(body, mimetype="text/csv", headers={"Content-Disposition": f"attachment; filename={table}.csv"})


@app.post("/api/predict")
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        features = validate_features(payload)
        result = predict_one(features)
        prediction_id = add_prediction(result)
        result["prediction_id"] = prediction_id
        return jsonify({"success": True, "prediction": result})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        app.logger.exception("Prediction failure")
        return jsonify({"success": False, "error": "Prediction service failed. Check server logs."}), 500


@app.post("/api/students")
def create_student():
    try:
        payload = request.get_json(silent=True) or {}
        name = str(payload.get("name", "")).strip()
        if len(name) < 2 or len(name) > 80:
            raise ValueError("Student name must be 2–80 characters")
        prediction_id = payload.get("prediction_id")
        if prediction_id is not None:
            try:
                prediction_id = int(prediction_id)
            except (TypeError, ValueError):
                raise ValueError("prediction_id must be a valid integer")
            existing = get_prediction(prediction_id)
            if not existing:
                raise ValueError("Prediction record was not found")
            result = {
                "study_hours": existing["study_hours"],
                "attendance": existing["attendance"],
                "previous_marks": existing["previous_marks"],
                "predicted_marks": existing["predicted_marks"],
                "performance_band": existing.get("performance_band", "Not classified"),
                "risk_level": existing.get("risk_level", "Not classified"),
                "risk_score": existing.get("risk_score"),
                "recommendations": [x for x in str(existing.get("recommendations", "")).split(" | ") if x],
                "prediction_low": existing.get("prediction_low"),
                "prediction_high": existing.get("prediction_high"),
                "model_name": existing.get("model_name", "Unknown"),
                "guidance": guidance_for(
                    {"study_hours": existing["study_hours"], "attendance": existing["attendance"], "previous_marks": existing["previous_marks"]},
                    existing["predicted_marks"],
                    load_bundle_safe()[1],
                ),
                "prediction_id": prediction_id,
            }
        else:
            features = validate_features(payload)
            result = predict_one(features)
            prediction_id = add_prediction(result)
            result["prediction_id"] = prediction_id
        student_code_raw = str(payload.get("student_code", "")).strip()
        student_code = student_code_raw[:30] or None
        row_id = add_student(
            {
                "name": name,
                "student_code": student_code,
                "prediction_id": prediction_id,
                **result,
                "recommendations": " | ".join(result.get("recommendations", [])),
                "guidance": " | ".join(result.get("guidance", [])),
            }
        )
        if prediction_id:
            attach_student_to_prediction(prediction_id, student_code)
        return jsonify({"success": True, "id": row_id, "student": {"name": name, "student_code": student_code, **result}}), 201
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        app.logger.exception("Student create failure")
        return jsonify({"success": False, "error": "Could not save student. Student IDs must be unique."}), 409 if "UNIQUE" in str(e) else 500


@app.get("/api/students")
def students():
    q = request.args.get("q", "").strip()
    return jsonify({"students": list_students(q)})


@app.delete("/api/students/<int:student_id>")
def remove_student(student_id: int):
    ok = delete_student(student_id)
    return (jsonify({"success": True}), 200) if ok else (jsonify({"success": False, "error": "Student not found"}), 404)


@app.post("/api/retrain")
def retrain():
    try:
        meta = train_and_evaluate()
        return jsonify({"success": True, "model": meta})
    except Exception as e:
        app.logger.exception("Retraining failure")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
