import pandas as pd
import pytest

from core import build_prediction, classify_prediction, support_indicator, validate_features
from ml_pipeline import FEATURES, load_model_bundle


def test_validation_accepts_valid_and_rejects_invalid_inputs():
    assert validate_features({'study_hours': 5, 'attendance': 80, 'previous_marks': 65})['attendance'] == 80.0
    with pytest.raises(ValueError):
        validate_features({'study_hours': 25, 'attendance': 80, 'previous_marks': 65})
    with pytest.raises(ValueError):
        validate_features({'study_hours': 'abc', 'attendance': 80, 'previous_marks': 65})


def test_prediction_band_rules_are_deterministic():
    assert classify_prediction(80) == 'High (75–100)'
    assert classify_prediction(60) == 'Moderate (60–74.9)'
    assert classify_prediction(50) == 'Needs support (45–59.9)'
    assert classify_prediction(20) == 'Low (<45)'


def test_prediction_explainability_is_real_model_metadata():
    model, metadata = load_model_bundle()
    features = {'study_hours': 6, 'attendance': 82, 'previous_marks': 68}
    result = build_prediction(features, model, metadata)
    assert 0 <= result['predicted_marks'] <= 100
    assert result['model_version'] == metadata['model_version']
    assert result['explainability']['features']
    assert all(item['feature'] in FEATURES for item in result['explainability']['features'])


def test_support_indicator_is_data_derived_not_probability():
    _, metadata = load_model_bundle()
    level, recs, score = support_indicator(
        {'study_hours': 1, 'attendance': 60, 'previous_marks': 45},
        50,
        metadata,
    )
    assert level in {'Lower support need', 'Review recommended', 'Higher support need'}
    assert 0 <= score <= 100
    assert recs
