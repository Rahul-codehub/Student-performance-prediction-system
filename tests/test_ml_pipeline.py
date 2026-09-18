import json

import numpy as np
import pandas as pd

from ml_pipeline import FEATURES, TARGET, load_data, load_model_bundle, train_and_evaluate


def test_dataset_schema_and_cleanliness():
    df = load_data()
    assert all(c in df.columns for c in FEATURES + [TARGET])
    assert len(df) == 20
    assert df.isna().sum().sum() == 0
    assert df.duplicated().sum() == 0


def test_training_is_deterministic_except_timestamp():
    a = train_and_evaluate()
    b = train_and_evaluate()
    assert a['model_name'] == b['model_name']
    assert a['selected_metrics'] == b['selected_metrics']
    assert a['dataset_rows'] == b['dataset_rows'] == 20
    assert a['holdout_size'] == b['holdout_size'] == 4
    assert a['training_data_hash'] == b['training_data_hash']
    assert a['model_version'] == b['model_version']


def test_model_predictions_are_bounded_and_finite():
    model, _ = load_model_bundle()
    df = load_data()
    pred = model.predict(df[FEATURES])
    assert len(pred) == len(df)
    assert np.isfinite(pred).all()
    assert (pred >= 0).all() and (pred <= 100).all()


def test_known_input_prediction_is_stable():
    model, _ = load_model_bundle()
    x = pd.DataFrame([{'study_hours': 6, 'attendance': 82, 'previous_marks': 68}])
    a = float(model.predict(x)[0])
    b = float(model.predict(x)[0])
    assert a == b
    assert 0 <= a <= 100


def test_metadata_contains_real_evaluation_context():
    _, metadata = load_model_bundle()
    assert metadata['dataset_rows'] == 20
    assert metadata['holdout_size'] == 4
    assert metadata['cv_folds'] == 5
    assert metadata['limitations']
    assert metadata['holdout_rows']
    assert metadata['correlations_with_target']
    assert len(metadata['training_data_hash']) == 64


def test_saved_metadata_is_valid_json():
    with open('outputs/model_metrics.json', 'r', encoding='utf-8') as handle:
        metadata = json.load(handle)
    assert metadata['model_name']
    assert metadata['model_version']
