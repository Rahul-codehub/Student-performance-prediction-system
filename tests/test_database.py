import sqlite3

import database


def test_old_schema_is_migrated_and_writes_work(tmp_path, monkeypatch):
    db_path = tmp_path / 'legacy.db'
    monkeypatch.setattr(database, 'DB_PATH', db_path)

    conn = sqlite3.connect(db_path)
    conn.executescript('''
        CREATE TABLE students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_code TEXT UNIQUE,
            name TEXT NOT NULL,
            study_hours REAL NOT NULL,
            attendance REAL NOT NULL,
            previous_marks REAL NOT NULL,
            predicted_marks REAL NOT NULL,
            risk_level TEXT NOT NULL,
            recommendations TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );
        CREATE TABLE prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_code TEXT,
            study_hours REAL NOT NULL,
            attendance REAL NOT NULL,
            previous_marks REAL NOT NULL,
            predicted_marks REAL NOT NULL,
            risk_level TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    ''')
    conn.commit(); conn.close()

    database.init_db()
    prediction_id = database.add_prediction({
        'study_hours': 6,
        'attendance': 82,
        'previous_marks': 68,
        'predicted_marks': 70,
        'risk_level': 'On Track',
        'risk_score': 80,
        'performance_band': 'Moderate (60–74.9)',
        'prediction_low': 69,
        'prediction_high': 71,
        'model_name': 'Linear Regression',
    })
    student_id = database.add_student({
        'name': 'Migration Test',
        'student_code': 'MIG-001',
        'study_hours': 6,
        'attendance': 82,
        'previous_marks': 68,
        'predicted_marks': 70,
        'risk_level': 'On Track',
        'risk_score': 80,
        'performance_band': 'Moderate (60–74.9)',
        'model_name': 'Linear Regression',
        'prediction_id': prediction_id,
    })
    assert prediction_id > 0
    assert student_id > 0
    assert database.list_students()[0]['student_code'] == 'MIG-001'
