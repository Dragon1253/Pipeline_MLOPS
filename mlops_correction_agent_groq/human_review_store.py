"""Durable, single-use human review decisions for the V4 pipeline."""
import hashlib
import hmac
import json
import os
from contextlib import contextmanager
from pathlib import Path
import secrets
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4


DEFAULT_DATABASE_PATH = Path(__file__).with_name("artifacts_v4") / "human_reviews.sqlite3"


def review_database_path():
    return Path(os.getenv("HUMAN_REVIEW_DB", DEFAULT_DATABASE_PATH))


class ReviewNotFound(Exception):
    pass


class InvalidReviewToken(Exception):
    pass


class ReviewAlreadyResolved(Exception):
    pass


class HumanReviewStore:
    def __init__(self, database_path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS human_reviews (
                    review_id TEXT PRIMARY KEY,
                    token_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    decision TEXT,
                    decided_at TEXT,
                    execution_json TEXT
                )"""
            )

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _token_hash(token):
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create_review(self, *, request, incident, correction_mode, run_dir,
                      pipeline_result_file):
        review_id = uuid4().hex
        token = secrets.token_urlsafe(32)
        payload = {
            "request": request,
            "incident": incident,
            "correction_mode": correction_mode,
            "run_dir": str(Path(run_dir).resolve()),
            "pipeline_result_file": str(Path(pipeline_result_file).resolve()),
            "created_at": self._now(),
        }
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO human_reviews
                   (review_id, token_hash, status, payload_json)
                   VALUES (?, ?, 'pending', ?)""",
                (review_id, self._token_hash(token), json.dumps(payload)),
            )
        return {"review_id": review_id, "token": token}

    def get_review(self, review_id, token):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM human_reviews WHERE review_id = ?",
                (review_id,),
            ).fetchone()
        if row is None:
            raise ReviewNotFound(review_id)
        if not hmac.compare_digest(row["token_hash"], self._token_hash(token)):
            raise InvalidReviewToken(review_id)
        return self._to_dict(row)

    def resolve(self, review_id, token, decision):
        if decision not in {"approved", "rejected"}:
            raise ValueError("La décision doit être approved ou rejected.")

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM human_reviews WHERE review_id = ?",
                (review_id,),
            ).fetchone()
            if row is None:
                raise ReviewNotFound(review_id)
            if not hmac.compare_digest(row["token_hash"], self._token_hash(token)):
                raise InvalidReviewToken(review_id)
            if row["status"] != "pending":
                raise ReviewAlreadyResolved(review_id)

            connection.execute(
                """UPDATE human_reviews
                   SET status = ?, decision = ?, decided_at = ?
                   WHERE review_id = ? AND status = 'pending'""",
                (decision, decision, self._now(), review_id),
            )
            resolved = connection.execute(
                "SELECT * FROM human_reviews WHERE review_id = ?",
                (review_id,),
            ).fetchone()
        return self._to_dict(resolved)

    def claim_execution(self, review_id):
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE human_reviews SET status = 'executing'
                   WHERE review_id = ? AND status = 'approved'""",
                (review_id,),
            )
            return cursor.rowcount == 1

    def complete_execution(self, review_id, *, status, result):
        if status not in {"completed", "blocked", "failed"}:
            raise ValueError("Statut d'exécution invalide.")
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE human_reviews
                   SET status = ?, execution_json = ?
                   WHERE review_id = ? AND status = 'executing'""",
                (status, json.dumps(result, ensure_ascii=False, default=str), review_id),
            )
            if cursor.rowcount != 1:
                raise ReviewAlreadyResolved(review_id)

    @staticmethod
    def _to_dict(row):
        return {
            "review_id": row["review_id"],
            "status": row["status"],
            "decision": row["decision"],
            "decided_at": row["decided_at"],
            "execution": json.loads(row["execution_json"])
            if row["execution_json"] else None,
            **json.loads(row["payload_json"]),
        }
