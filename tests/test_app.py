import os

import pytest

os.environ['STUDENTIQ_TESTING'] = '1'
pytest.importorskip('flask')
import app as module
import database


@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch):
    monkeypatch.setattr(database, 'DB_PATH', tmp_path / 'student_performance.db')
    database.init_db()
    yield


def client():
    module.app.config['TESTING'] = True
    return module.app.test_client()


def test_health():
    response = client().get('/health')
    assert response.status_code == 200
    assert response.json['status'] == 'healthy'
    assert response.json['dataset_rows'] == 20


def test_predict_logs_real_prediction_and_returns_previous_model_context():
    response = client().post('/api/predict', json={'study_hours': 6, 'attendance': 82, 'previous_marks': 68})
    assert response.status_code == 200
    body = response.json['prediction']
    assert body['prediction_id'] >= 1
    assert 0 <= body['predicted_marks'] <= 100
    assert body['risk_level'] in {'Excellent', 'On Track', 'Needs Support', 'At Risk'}
    assert 0 <= body['risk_score'] <= 100
    assert isinstance(body['recommendations'], list)
    assert body['prediction_low'] <= body['predicted_marks'] <= body['prediction_high']
    assert client().get('/api/dashboard').json['total_predictions'] == 1


def test_predict_validation():
    response = client().post('/api/predict', json={'study_hours': 99, 'attendance': 82, 'previous_marks': 68})
    assert response.status_code == 400


def test_save_existing_prediction_does_not_create_extra_prediction():
    prediction = client().post('/api/predict', json={'study_hours': 5, 'attendance': 80, 'previous_marks': 65}).json['prediction']
    before = client().get('/api/dashboard').json['total_predictions']
    response = client().post('/api/students', json={
        'name': 'Test Student',
        'student_code': 'TEST-001',
        'prediction_id': prediction['prediction_id'],
    })
    assert response.status_code == 201
    after = client().get('/api/dashboard').json['total_predictions']
    assert after == before == 1
    student = client().get('/api/students?q=TEST-001').json['students'][0]
    history = client().get('/api/history?q=TEST-001').json['predictions'][0]
    assert student['name'] == 'Test Student'
    assert history['student_code'] == 'TEST-001'


def test_direct_add_student_creates_one_prediction_and_one_student():
    response = client().post('/api/students', json={
        'name': 'Direct Student',
        'student_code': 'DIRECT-001',
        'study_hours': 7,
        'attendance': 88,
        'previous_marks': 72,
    })
    assert response.status_code == 201
    body = response.json['student']
    assert body['name'] == 'Direct Student'
    assert body['prediction_id'] >= 1
    assert client().get('/api/dashboard').json['total_predictions'] == 1
    assert client().get('/api/students?q=DIRECT-001').json['students']


def test_duplicate_student_id_returns_conflict_and_does_not_lock_database():
    payload = {'name': 'Student A', 'student_code': 'DUP-001', 'study_hours': 6, 'attendance': 82, 'previous_marks': 68}
    assert client().post('/api/students', json=payload).status_code == 201
    duplicate = client().post('/api/students', json={**payload, 'name': 'Student B'})
    assert duplicate.status_code == 409
    follow_up = client().delete('/api/students/1')
    assert follow_up.status_code == 200


def test_dashboard_history_and_exports():
    body = client().get('/api/dashboard').json
    assert body['dataset']['rows'] == 20
    assert body['dataset']['missing_values'] == 0
    assert 'model' in body and 'holdout_size' in body['model']
    assert client().get('/api/history').status_code == 200
    csv_response = client().get('/api/export/prediction_history.csv')
    assert csv_response.status_code == 200
    assert 'text/csv' in csv_response.content_type
