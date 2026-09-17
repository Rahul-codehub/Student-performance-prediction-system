from __future__ import annotations

import csv
import io
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "outputs" / "student_performance.db"


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_column(conn: sqlite3.Connection, table: str, name: str, definition: str) -> None:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if name not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def init_db() -> None:
    conn = connect()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_code TEXT UNIQUE,
            name TEXT NOT NULL,
            study_hours REAL NOT NULL,
            attendance REAL NOT NULL,
            previous_marks REAL NOT NULL,
            predicted_marks REAL NOT NULL,
            risk_level TEXT NOT NULL DEFAULT 'Not classified',
            risk_score INTEGER,
            recommendations TEXT NOT NULL DEFAULT '',
            performance_band TEXT NOT NULL DEFAULT 'Not classified',
            guidance TEXT NOT NULL DEFAULT '',
            model_name TEXT NOT NULL DEFAULT 'Unknown',
            prediction_id INTEGER,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_code TEXT,
            study_hours REAL NOT NULL,
            attendance REAL NOT NULL,
            previous_marks REAL NOT NULL,
            predicted_marks REAL NOT NULL,
            risk_level TEXT NOT NULL DEFAULT 'Not classified',
            risk_score INTEGER,
            performance_band TEXT NOT NULL DEFAULT 'Not classified',
            prediction_low REAL,
            prediction_high REAL,
            model_name TEXT NOT NULL DEFAULT 'Unknown',
            created_at TEXT NOT NULL
        );
        """
    )

    # Migrate databases created by the earlier project versions.
    for name, definition in {
        "risk_level": "TEXT NOT NULL DEFAULT 'Not classified'",
        "risk_score": "INTEGER",
        "recommendations": "TEXT NOT NULL DEFAULT ''",
        "performance_band": "TEXT NOT NULL DEFAULT 'Not classified'",
        "guidance": "TEXT NOT NULL DEFAULT ''",
        "model_name": "TEXT NOT NULL DEFAULT 'Unknown'",
        "prediction_id": "INTEGER",
    }.items():
        _ensure_column(conn, "students", name, definition)

    for name, definition in {
        "risk_level": "TEXT NOT NULL DEFAULT 'Not classified'",
        "risk_score": "INTEGER",
        "performance_band": "TEXT NOT NULL DEFAULT 'Not classified'",
        "prediction_low": "REAL",
        "prediction_high": "REAL",
        "model_name": "TEXT NOT NULL DEFAULT 'Unknown'",
    }.items():
        _ensure_column(conn, "prediction_history", name, definition)

    conn.commit()
    conn.close()


# Always initialize/migrate the database when this module is imported.
# This prevents an old packaged SQLite file from breaking the first request.
init_db()


def add_prediction(data: dict) -> int:
    conn = connect()
    try:
        cur = conn.execute(
            """INSERT INTO prediction_history
            (student_code, study_hours, attendance, previous_marks, predicted_marks,
             risk_level, risk_score, performance_band, prediction_low, prediction_high, model_name, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data.get("student_code"),
                data["study_hours"],
                data["attendance"],
                data["previous_marks"],
                data["predicted_marks"],
                data.get("risk_level", "Not classified"),
                data.get("risk_score"),
                data.get("performance_band", "Not classified"),
                data.get("prediction_low"),
                data.get("prediction_high"),
                data.get("model_name", "Unknown"),
                now_iso(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def add_student(data: dict) -> int:
    conn = connect()
    try:
        cur = conn.execute(
            """INSERT INTO students
            (student_code, name, study_hours, attendance, previous_marks, predicted_marks,
             risk_level, risk_score, recommendations, performance_band, guidance, model_name, prediction_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data.get("student_code"),
                data["name"],
                data["study_hours"],
                data["attendance"],
                data["previous_marks"],
                data["predicted_marks"],
                data.get("risk_level", "Not classified"),
                data.get("risk_score"),
                data.get("recommendations", ""),
                data.get("performance_band", "Not classified"),
                data.get("guidance", ""),
                data.get("model_name", "Unknown"),
                data.get("prediction_id"),
                now_iso(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def attach_student_to_prediction(prediction_id: int, student_code: str | None) -> None:
    conn = connect()
    conn.execute("UPDATE prediction_history SET student_code=? WHERE id=?", (student_code, int(prediction_id)))
    conn.commit()
    conn.close()


def list_students(query: str = "") -> list[dict]:
    conn = connect()
    if query:
        rows = conn.execute(
            "SELECT * FROM students WHERE name LIKE ? OR student_code LIKE ? ORDER BY id DESC",
            (f"%{query}%", f"%{query}%"),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM students ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_student(student_id: int) -> bool:
    conn = connect()
    cur = conn.execute("DELETE FROM students WHERE id=?", (student_id,))
    conn.commit()
    ok = cur.rowcount > 0
    conn.close()
    return ok


def get_prediction(prediction_id: int) -> dict | None:
    conn = connect()
    row = conn.execute("SELECT * FROM prediction_history WHERE id=?", (int(prediction_id),)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_predictions(query: str = "", limit: int = 100) -> list[dict]:
    conn = connect()
    if query:
        rows = conn.execute(
            """SELECT * FROM prediction_history
               WHERE CAST(id AS TEXT) LIKE ? OR student_code LIKE ? OR risk_level LIKE ? OR performance_band LIKE ? OR model_name LIKE ?
               ORDER BY id DESC LIMIT ?""",
            (f"%{query}%", f"%{query}%", f"%{query}%", f"%{query}%", f"%{query}%", int(limit)),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM prediction_history ORDER BY id DESC LIMIT ?", (int(limit),)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def dashboard_stats() -> dict:
    conn = connect()
    total_predictions = int(conn.execute("SELECT COUNT(*) FROM prediction_history").fetchone()[0])
    saved_students = int(conn.execute("SELECT COUNT(*) FROM students").fetchone()[0])
    prediction_summary = conn.execute(
        """SELECT AVG(predicted_marks) AS avg_prediction,
                  AVG(attendance) AS avg_attendance,
                  MIN(predicted_marks) AS min_prediction,
                  MAX(predicted_marks) AS max_prediction
           FROM prediction_history"""
    ).fetchone()
    latest = conn.execute(
        "SELECT created_at, predicted_marks, performance_band, risk_level, risk_score, model_name, student_code FROM prediction_history ORDER BY id DESC LIMIT 1"
    ).fetchone()
    recent = conn.execute("SELECT * FROM prediction_history ORDER BY id DESC LIMIT 8").fetchall()
    band_rows = conn.execute(
        "SELECT performance_band, COUNT(*) AS count FROM prediction_history GROUP BY performance_band"
    ).fetchall()
    conn.close()

    bands = {"High (75–100)": 0, "Moderate (60–74.9)": 0, "Needs support (45–59.9)": 0, "Low (<45)": 0}
    for row in band_rows:
        if row[0] in bands:
            bands[row[0]] = int(row[1])

    return {
        "total_predictions": total_predictions,
        "saved_students": saved_students,
        "avg_prediction": round(float(prediction_summary[0]), 2) if prediction_summary[0] is not None else None,
        "avg_attendance": round(float(prediction_summary[1]), 2) if prediction_summary[1] is not None else None,
        "min_prediction": round(float(prediction_summary[2]), 2) if prediction_summary[2] is not None else None,
        "max_prediction": round(float(prediction_summary[3]), 2) if prediction_summary[3] is not None else None,
        "performance_bands": bands,
        "last_prediction": dict(latest) if latest else None,
        "recent_predictions": [dict(r) for r in recent],
    }


def csv_text(table: str) -> str:
    if table not in {"students", "prediction_history"}:
        raise ValueError("Unsupported export")
    conn = connect()
    rows = conn.execute(f"SELECT * FROM {table} ORDER BY id DESC").fetchall()
    conn.close()
    output = io.StringIO()
    writer = csv.writer(output)
    if rows:
        writer.writerow(rows[0].keys())
        writer.writerows([tuple(row) for row in rows])
    return output.getvalue()
