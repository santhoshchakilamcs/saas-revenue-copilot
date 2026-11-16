"""SQLite store for agent answers (audit trail) and the human review queue."""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")


def redact(text: str) -> str:
    """Strip emails and phone numbers before anything is written to the audit log."""
    return PHONE_RE.sub("[PHONE]", EMAIL_RE.sub("[EMAIL]", text))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str = ":memory:"):
        self._lock = threading.Lock()
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.con.executescript(
            """
            CREATE TABLE IF NOT EXISTS answers (
                id TEXT PRIMARY KEY, created_at TEXT, username TEXT, question TEXT, answer TEXT,
                value REAL, plan TEXT, confidence REAL, backend TEXT, flagged INTEGER);
            CREATE TABLE IF NOT EXISTS reviews (
                id TEXT PRIMARY KEY, answer_id TEXT REFERENCES answers(id), created_at TEXT,
                flagged_by TEXT, reason TEXT, status TEXT, resolved_by TEXT, resolved_at TEXT,
                verdict TEXT, corrected_answer TEXT, note TEXT);
            """
        )

    def save_answer(self, username: str, out: dict, flagged: bool) -> str:
        aid = uuid.uuid4().hex[:12]
        with self._lock:
            self.con.execute(
                "INSERT INTO answers VALUES (?,?,?,?,?,?,?,?,?,?)",
                (aid, _now(), username, redact(out["question"]), out["answer"], out["value"],
                 json.dumps(out["plan"]), out["confidence"], out["backend"], int(flagged)),
            )
            self.con.commit()
        return aid

    def get_answer(self, aid: str) -> dict | None:
        r = self.con.execute("SELECT * FROM answers WHERE id = ?", (aid,)).fetchone()
        return dict(r) if r else None

    def flag(self, answer_id: str, by: str, reason: str) -> str:
        rid = uuid.uuid4().hex[:12]
        with self._lock:
            self.con.execute(
                "INSERT INTO reviews (id, answer_id, created_at, flagged_by, reason, status) VALUES (?,?,?,?,?, 'open')",
                (rid, answer_id, _now(), by, redact(reason)),
            )
            self.con.execute("UPDATE answers SET flagged = 1 WHERE id = ?", (answer_id,))
            self.con.commit()
        return rid

    def open_review_for(self, answer_id: str) -> dict | None:
        r = self.con.execute("SELECT * FROM reviews WHERE answer_id = ? AND status = 'open'", (answer_id,)).fetchone()
        return dict(r) if r else None

    def list_reviews(self, status: str | None = "open") -> list[dict]:
        q = ("SELECT r.*, a.question, a.answer, a.confidence FROM reviews r JOIN answers a ON a.id = r.answer_id"
             + (" WHERE r.status = ?" if status else "") + " ORDER BY r.created_at")
        return [dict(r) for r in self.con.execute(q, (status,) if status else ()).fetchall()]

    def get_review(self, rid: str) -> dict | None:
        r = self.con.execute("SELECT * FROM reviews WHERE id = ?", (rid,)).fetchone()
        return dict(r) if r else None

    def resolve(self, rid: str, by: str, verdict: str, corrected: str | None, note: str | None) -> None:
        with self._lock:
            self.con.execute(
                "UPDATE reviews SET status='resolved', resolved_by=?, resolved_at=?, verdict=?, corrected_answer=?, note=? WHERE id=?",
                (by, _now(), verdict, corrected, redact(note or ""), rid),
            )
            self.con.commit()
