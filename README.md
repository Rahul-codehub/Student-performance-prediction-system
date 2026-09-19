# Student Performance Analytics & Prediction System

A college-ready Flask application that combines a supervised regression model with a web dashboard, SQLite persistence, model evaluation, explainability, data profiling, what-if simulation, student records, prediction history, and release tests.

## What is genuinely powered by data

The supplied training source is `data/student_data.csv`.

Current supplied dataset:

- 20 records
- Inputs: `study_hours`, `attendance`, `previous_marks`
- Target: `final_marks`
- 0 missing values
- 0 duplicate rows

The runtime database starts empty. The dashboard does not contain fabricated students, prediction counts, or sample activity. Prediction history and saved students appear only after actual user actions.

## Main modules

### 1. Prediction workspace

Accepts the three academic inputs, validates them, loads the selected trained model, produces a final-mark estimate, shows the performance band, displays model context and feature explanation, and records the prediction in SQLite.

### 2. Scenario lab

Runs two user-supplied scenarios through the same model without storing them in prediction history. This supports transparent what-if comparison without manufacturing student records.

### 3. Student records

Saves named prediction snapshots. Saving an existing prediction does not create another prediction. Direct student creation creates one prediction and one student record. Duplicate IDs are rejected safely.

### 4. Academic analytics

Uses only observed training data and actual runtime history. It includes:

- Feature/target correlations
- Final-mark distribution
- Target summary statistics
- Holdout actual-vs-predicted records
- Runtime prediction counts and bands
- Data-readiness disclosure for academic fields not present in the CSV

### 5. Model lab

Compares:

- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting

The selected model is the one with the lowest cross-validation MAE; cross-validation R² is the tie-breaker.

### 6. Model explainability

When the selected model exposes coefficients, the application shows coefficient values and the corresponding feature contribution for a prediction. For tree models, feature-importance metadata is used instead. These are explanations of model behavior, not causal claims.

### 7. Training data explorer

Shows the exact CSV records and observed minimum, maximum, mean, median, and standard deviation values.

### 8. Model versioning

Each training run records a model version derived from the selected model and SHA-256 hash of the training CSV. Prediction history stores the model version used for that request.

## Current model result

Using the supplied 20-row dataset and the deterministic training procedure:

- Selected model: **Linear Regression**
- Holdout size: **4 records**
- Holdout MAE: **0.3416**
- Holdout RMSE: **0.4291**
- Holdout R²: **0.9979**
- 5-fold CV MAE: **0.4266**
- 5-fold CV R²: **0.9981**

These are evaluation results on a very small supplied dataset. They must not be described as 99.8% prediction accuracy or as a guarantee of real-world or institution-wide performance.

## Support indicator

The application retains the previous “risk/support” concept but makes the interpretation explicit. The score is derived from shortfalls relative to observed training-data averages and is shown as a **data-derived support index**. It is not a calibrated probability, diagnosis, or classification model.

Performance bands are application rules applied to the predicted mark:

- High: 75–100
- Moderate: 60–74.9
- Needs support: 45–59.9
- Low: below 45

## Data that is intentionally not invented

The current CSV does not contain subject, semester, assignment, internal-assessment, or practical-score fields. The application therefore does not fabricate those dimensions. The Analytics page explicitly reports their absence so the system can be extended later when validated data is available.

## Project structure

```text
student-performance-prediction/
├── app.py
├── core.py
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
    ├── test_core.py
    ├── test_ml_pipeline.py
    ├── test_database.py
    ├── test_app.py
    └── test_release_contracts.py
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

In the build sandbox used for the release, **11 tests passed and 1 Flask test module was skipped** because Flask was not installed and packages could not be downloaded. The skipped module contains the real Flask test-client checks and should run after `pip install -r requirements.txt` in the user's environment. The non-Flask suite covers ML behavior, explainability, database migration, release contracts, deterministic training, and the zero-record release state.

## API

- `GET /health` — model/service status and model version.
- `GET /api/dashboard` — runtime activity, dataset summary, and model metadata.
- `GET /api/analytics` — observed dataset relationships, holdout details, and runtime analytics.
- `POST /api/predict` — create and log one prediction.
- `POST /api/simulate` — run a prediction without saving it to history.
- `GET /api/history` — prediction history with optional search.
- `GET /api/students` — saved student records with optional search.
- `POST /api/students` — save a student snapshot; existing predictions can be reused without double-counting.
- `DELETE /api/students/<id>` — delete a saved student record.
- `GET /api/data` — current training CSV and feature summary.
- `GET /api/export/students.csv` — export saved students.
- `GET /api/export/prediction_history.csv` — export prediction history.
- `POST /api/retrain` — retrain candidate models from the current CSV.

## Important academic limitation

The software architecture is intentionally richer than a single-prediction demo, but the underlying dataset remains small. A larger validated dataset is the main requirement before using the system for real academic decision-making.
