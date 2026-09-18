from __future__ import annotations

from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

from core import build_prediction, validate_features
from database import (
    add_prediction,
    add_student,
    attach_student_to_prediction,
    csv_text,
    dashboard_stats,
    delete_student,
    get_prediction,
    init_db,
    list_predictions,
    list_students,
)
from ml_pipeline import FEATURES, load_data, load_model_bundle, train_and_evaluate

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
init_db()

ADVANCED_FIELDS = [
    "student_id",
    "semester",
    "subject",
    "assignment_score",
    "internal_marks",
    "quiz_score",
    "practical_score",
]


def load_bundle_safe():
    try:
        return load_model_bundle()
    except FileNotFoundError:
        train_and_evaluate()
        return load_model_bundle()


def _prediction_from_features(features: dict) -> dict:
    model, metadata = load_bundle_safe()
    return build_prediction(features, model, metadata)


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    model, meta = load_bundle_safe()
    return jsonify(
        {
            "status": "healthy",
            "model": meta.get("model_name", type(model).__name__),
            "model_version": meta.get("model_version"),
            "dataset_rows": meta.get("dataset_rows", 0),
            "training_data_hash": meta.get("training_data_hash"),
        }
    )


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
                "correlations_with_target": meta.get("correlations_with_target", {}),
            },
            "model": {
                "name": meta.get("model_name"),
                "family": meta.get("model_family"),
                "version": meta.get("model_version"),
                "trained_at": meta.get("trained_at"),
                "training_data_hash": meta.get("training_data_hash"),
                "selected_metrics": meta.get("selected_metrics", {}),
                "all_models": meta.get("metrics", []),
                "dataset_rows": meta.get("dataset_rows"),
                "holdout_size": meta.get("holdout_size"),
                "cv_folds": meta.get("cv_folds"),
                "selection_method": meta.get("selection_method"),
                "limitations": meta.get("limitations", []),
                "holdout_rows": meta.get("holdout_rows", []),
                "explainability": meta.get("explainability", {}),
                "model_family": meta.get("model_family"),
            },
            "data_availability": meta.get("data_availability", {}),
        }
    )
    return jsonify(stats)


@app.get("/api/analytics")
def analytics():
    ds = load_data()
    _, meta = load_bundle_safe()
    stats = dashboard_stats()
    numeric = ds[FEATURES + ["final_marks"]]
    return jsonify(
        {
            "dataset": {
                "rows": len(ds),
                "columns": list(ds.columns),
                "feature_summary": meta.get("feature_summary", {}),
                "target_summary": meta.get("target_summary", {}),
                "correlations_with_target": meta.get("correlations_with_target", {}),
                "distribution": {
                    "final_marks": numeric["final_marks"].tolist(),
                    "study_hours": numeric["study_hours"].tolist(),
                    "attendance": numeric["attendance"].tolist(),
                    "previous_marks": numeric["previous_marks"].tolist(),
                },
            },
            "evaluation": {
                "selected_model": meta.get("model_name"),
                "holdout_rows": meta.get("holdout_rows", []),
                "metrics": meta.get("metrics", []),
                "selected_metrics": meta.get("selected_metrics", {}),
            },
            "runtime": {
                "total_predictions": stats["total_predictions"],
                "saved_students": stats["saved_students"],
                "predictions_by_day": stats.get("predictions_by_day", []),
                "performance_bands": stats.get("performance_bands", {}),
                "support_levels": stats.get("support_levels", {}),
            },
            "data_availability": meta.get("data_availability", {}),
            "advanced_fields_not_present": [field for field in ADVANCED_FIELDS if field not in ds.columns],
        }
    )


@app.get("/api/history")
def history():
    q = request.args.get("q", "").strip()
    return jsonify({"predictions": list_predictions(q)})


@app.get("/api/data")
def data_view():
    ds = load_data()
    _, meta = load_bundle_safe()
    return jsonify(
        {
            "columns": list(ds.columns),
            "rows": ds.to_dict(orient="records"),
            "summary": meta.get("feature_summary", {}),
            "missing_values": int(ds.isna().sum().sum()),
            "duplicate_rows": int(ds.duplicated().sum()),
            "duplicates_removed": meta.get("duplicates_removed", 0),
            "training_data_hash": meta.get("training_data_hash"),
        }
    )


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
        result = _prediction_from_features(features)
        prediction_id = add_prediction(result)
        result["prediction_id"] = prediction_id
        return jsonify({"success": True, "prediction": result})
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception:
        app.logger.exception("Prediction failure")
        return jsonify({"success": False, "error": "Prediction service failed. Check server logs."}), 500


@app.post("/api/simulate")
def simulate():
    """Run a what-if scenario without writing to prediction history."""
    try:
        payload = request.get_json(silent=True) or {}
        features = validate_features(payload)
        result = _prediction_from_features(features)
        result.pop("prediction_id", None)
        return jsonify({"success": True, "simulation": result, "persisted": False})
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception:
        app.logger.exception("Simulation failure")
        return jsonify({"success": False, "error": "Scenario simulation failed."}), 500


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
                "model_version": existing.get("model_version"),
                "guidance": [],
                "prediction_id": prediction_id,
            }
        else:
            features = validate_features(payload)
            result = _prediction_from_features(features)
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
        return jsonify(
            {
                "success": True,
                "id": row_id,
                "student": {"name": name, "student_code": student_code, **result},
            }
        ), 201
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Student create failure")
        status = 409 if "UNIQUE" in str(exc).upper() else 500
        return jsonify({"success": False, "error": "Could not save student. Student IDs must be unique." if status == 409 else "Could not save student."}), status


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
    except Exception as exc:
        app.logger.exception("Retraining failure")
        return jsonify({"success": False, "error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
