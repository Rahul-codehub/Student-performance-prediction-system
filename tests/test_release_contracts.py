from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def test_required_api_contracts_and_ui_modules_present():
    app_text = (ROOT / 'app.py').read_text(encoding='utf-8')
    html = (ROOT / 'templates' / 'index.html').read_text(encoding='utf-8')
    js = (ROOT / 'static' / 'app.js').read_text(encoding='utf-8')
    required_routes = [
        '/health', '/api/dashboard', '/api/analytics', '/api/predict', '/api/simulate',
        '/api/students', '/api/history', '/api/data', '/api/retrain'
    ]
    for route in required_routes:
        assert route in app_text
    for view in ['overviewView', 'predictView', 'scenarioView', 'studentsView', 'analyticsView', 'historyView', 'dataView', 'modelView']:
        assert f'id="{view}"' in html
    for endpoint in ['/api/dashboard', '/api/analytics', '/api/predict', '/api/simulate', '/api/students', '/api/history']:
        assert endpoint in js


def test_no_demo_runtime_records_are_shipped():
    db = ROOT / 'outputs' / 'student_performance.db'
    assert db.exists()
    import sqlite3
    con = sqlite3.connect(db)
    try:
        assert con.execute('SELECT COUNT(*) FROM students').fetchone()[0] == 0
        assert con.execute('SELECT COUNT(*) FROM prediction_history').fetchone()[0] == 0
    finally:
        con.close()


def test_frontend_avoids_fake_dashboard_counts():
    js = (ROOT / 'static' / 'app.js').read_text(encoding='utf-8')
    assert 'No prediction requests have been recorded yet' in js
    assert 'No runtime prediction records exist yet' in js
    assert not re.search(r'total_predictions\s*[:=]\s*[1-9]', js)
