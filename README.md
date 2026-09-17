# Student Performance Prediction System

A college-ready Flask application for predicting final marks from three academic inputs: **study hours per week, attendance percentage, and previous marks**. The interface separates training data, model evaluation, prediction history, and saved student records so numbers are not mixed together.

## What the application does

- Predicts final marks with the selected regression model.
- Records every successful prediction request in SQLite prediction history.
- Saves named student snapshots either from an existing prediction or through the Students → Add student form.
- Shows the actual number of predictions, saved students, and training records.
- Compares Linear Regression, Ridge Regression, Random Forest, and Gradient Boosting.
- Uses one fixed holdout split and deterministic 5-fold cross-validation for model comparison.
- Flags inputs outside the feature ranges present in the training dataset.
- Shows the training-data mean/minimum/maximum values used for context.
- Provides CSV export for student records and prediction history.
- Uses a migration-safe SQLite schema so older project databases can be opened without breaking prediction or student creation.
- Cleans up failed SQLite writes so duplicate student IDs do not leave the database locked.
- Provides a training-data view so the supplied records can be inspected directly.
- Provides a retraining action that rebuilds the model from the current CSV file.

## Important data interpretation

The supplied CSV contains **20 records**. The evaluation metrics therefore describe a very small dataset and should not be presented as general real-world accuracy. The application deliberately labels holdout metrics and cross-validation metrics as model evaluation rather than as observed student outcomes.

The performance band is a simple rule-based interpretation of the predicted mark. A separate risk level and heuristic support indicator are retained from the earlier version for continuity; they are derived rules, not model probabilities or observed outcomes:

- High: 75–100
- Moderate: 60–74.9
- Needs support: 45–59.9
- Low: below 45

These bands are application rules, not labels learned from the supplied dataset.

## Project structure

```text
student-performance-prediction/
├── app.py
├── database.py
├── ml_pipeline.py
├── retrain.py
├── requirements.txt
├── README.md
├── data/
│   └── student_data.csv
├── models/
│   └── student_model.pkl
├── outputs/
│   ├── model_metrics.json
│   ├── student_performance.db
│   └── evaluation.txt
├── templates/
│   └── index.html
├── static/
│   ├── app.js
│   └── styles.css
└── tests/
    ├── conftest.py
    ├── test_app.py
    └── test_ml_pipeline.py
```

## Run on Windows

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python retrain.py
python app.py
```

Open `http://127.0.0.1:5000`.

## Run on macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python retrain.py
python app.py
```

## Testing

```bash
pytest -q
```

The test suite checks dataset schema, deterministic model selection, finite/bounded predictions, stable inference, evaluation metadata, SQLite migration, student creation, duplicate-ID handling, prediction logging, and deletion. When Flask is installed, the route tests additionally exercise the real Flask test client. A standalone route harness is used during release validation when Flask is unavailable in the build sandbox.

## API

- `GET /health` — current service/model status.
- `GET /api/dashboard` — application activity, dataset summary, and model evaluation.
- `POST /api/predict` — create and log one prediction.
- `GET /api/history` — prediction history with optional search.
- `GET /api/students` — saved student records with optional search.
- `POST /api/students` — save a student snapshot, optionally linked to an existing prediction; without a prediction ID, it calculates and records a new prediction first.
- `DELETE /api/students/<id>` — delete one saved student record.
- `GET /api/data` — current training dataset and feature summary.
- `GET /api/export/students.csv` — export saved student records.
- `GET /api/export/prediction_history.csv` — export prediction history.
- `POST /api/retrain` — retrain candidate models from the current CSV.

## Current supplied model result

The current training file has 20 records. The selected model is Linear Regression under the deterministic selection procedure. The stored evaluation reports a holdout MAE of approximately 0.342 marks and a holdout R² of approximately 0.998 on 4 holdout records, with 5-fold CV MAE of approximately 0.427. Because the dataset is small, these metrics must be interpreted as an academic demonstration of the supplied data, not as a production accuracy guarantee.
