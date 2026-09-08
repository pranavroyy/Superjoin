"""
SQLite storage for the fact knowledge layer.

Schema is intentionally generic (no hard-coded "revenue"/"director" columns)
so the system can absorb whatever kinds of facts the LLM extracts from
whatever PDFs it is given.
"""
import sqlite3
import json
import os
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    title TEXT,
    num_pages INTEGER,
    uploaded_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    page INTEGER,
    subject TEXT NOT NULL,
    attribute TEXT NOT NULL,
    value_raw TEXT NOT NULL,
    value_number REAL,
    unit TEXT,
    period TEXT,
    scope TEXT,
    quote TEXT NOT NULL,
    fact_type TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id_a INTEGER NOT NULL REFERENCES facts(id),
    fact_id_b INTEGER NOT NULL REFERENCES facts(id),
    relationship TEXT NOT NULL,  -- corroborates | contradicts | reconciled | unrelated
    explanation TEXT NOT NULL,
    confidence REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS extraction_issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    page INTEGER,
    issue TEXT NOT NULL,
    raw_excerpt TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


@contextmanager
def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def insert_document(filename, title, num_pages):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO documents (filename, title, num_pages) VALUES (?, ?, ?)",
            (filename, title, num_pages),
        )
        return cur.lastrowid


def insert_fact(document_id, page, subject, attribute, value_raw, value_number,
                 unit, period, scope, quote, fact_type):
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO facts
               (document_id, page, subject, attribute, value_raw, value_number,
                unit, period, scope, quote, fact_type)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (document_id, page, subject, attribute, value_raw, value_number,
             unit, period, scope, quote, fact_type),
        )
        return cur.lastrowid


def insert_relationship(fact_id_a, fact_id_b, relationship, explanation, confidence):
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO relationships (fact_id_a, fact_id_b, relationship, explanation, confidence)
               VALUES (?, ?, ?, ?, ?)""",
            (fact_id_a, fact_id_b, relationship, explanation, confidence),
        )
        return cur.lastrowid


def insert_issue(document_id, page, issue, raw_excerpt):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO extraction_issues (document_id, page, issue, raw_excerpt) VALUES (?, ?, ?, ?)",
            (document_id, page, issue, raw_excerpt),
        )


def all_facts(exclude_document_id=None):
    with get_conn() as conn:
        if exclude_document_id is not None:
            rows = conn.execute(
                "SELECT * FROM facts WHERE document_id != ?", (exclude_document_id,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM facts").fetchall()
        return [dict(r) for r in rows]


def facts_for_document(document_id):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM facts WHERE document_id = ?", (document_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_fact(fact_id):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        return dict(row) if row else None


def list_documents():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM documents ORDER BY id").fetchall()
        return [dict(r) for r in rows]


def list_relationships(relationship=None):
    with get_conn() as conn:
        if relationship:
            rows = conn.execute(
                "SELECT * FROM relationships WHERE relationship = ?", (relationship,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM relationships").fetchall()
        return [dict(r) for r in rows]


def list_issues():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM extraction_issues").fetchall()
        return [dict(r) for r in rows]
