"""State protocol and SQLite backend for task routing state.

Stores classifications, aliases, concrete selected models, reason codes,
counters, timestamps, and non-content metrics. Never stores prompts,
tool arguments, tool outputs, model text, credentials, or reasoning.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class TaskState(ABC):
    """Abstract task state store.

    Implementations must be thread-safe for concurrent access.
    """

    @abstractmethod
    def get_task(self, task_id: str) -> dict[str, Any] | None:
        """Get task state by ID. Returns None if not found."""

    @abstractmethod
    def create_task(
        self,
        task_id: str,
        session_id: str | None = None,
        task_class: str | None = None,
        primary_alias: str | None = None,
        escalation_alias: str | None = None,
        concrete_model: str | None = None,
        reason_code: str | None = None,
    ) -> None:
        """Create a new task state record."""

    @abstractmethod
    def update_model(
        self, task_id: str, concrete_model: str, alias: str | None = None
    ) -> None:
        """Update the pinned concrete model for a task."""

    @abstractmethod
    def increment_counter(self, task_id: str, counter_name: str) -> int:
        """Atomically increment a named counter. Returns the new value."""

    @abstractmethod
    def get_counter(self, task_id: str, counter_name: str) -> int:
        """Get current counter value."""

    @abstractmethod
    def touch(self, task_id: str) -> None:
        """Update the last_activity timestamp."""

    @abstractmethod
    def is_expired(self, task_id: str, ttl_seconds: int) -> bool:
        """Check if a task has exceeded its TTL since last activity."""

    @abstractmethod
    def delete_task(self, task_id: str) -> None:
        """Remove a task and all its counters."""

    @abstractmethod
    def close(self) -> None:
        """Close the state store."""


class SqliteTaskState(TaskState):
    """SQLite-backed task state store.

    Thread-safe with WAL mode and connection-per-thread.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(Path(db_path).expanduser())
        self._local = threading.local()
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._db_path, timeout=5)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        result: sqlite3.Connection = self._local.conn
        return result

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                session_id TEXT,
                task_class TEXT,
                primary_alias TEXT,
                escalation_alias TEXT,
                concrete_model TEXT,
                reason_code TEXT,
                created_at REAL NOT NULL,
                last_activity REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS counters (
                task_id TEXT NOT NULL,
                counter_name TEXT NOT NULL,
                value INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (task_id, counter_name),
                FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
            );
        """)
        conn.commit()

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def create_task(
        self,
        task_id: str,
        session_id: str | None = None,
        task_class: str | None = None,
        primary_alias: str | None = None,
        escalation_alias: str | None = None,
        concrete_model: str | None = None,
        reason_code: str | None = None,
    ) -> None:
        now = time.time()
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO tasks
               (task_id, session_id, task_class, primary_alias, escalation_alias,
                concrete_model, reason_code, created_at, last_activity)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id, session_id, task_class, primary_alias,
                escalation_alias, concrete_model, reason_code, now, now,
            ),
        )
        conn.commit()

    def update_model(
        self, task_id: str, concrete_model: str, alias: str | None = None
    ) -> None:
        conn = self._get_conn()
        if alias:
            conn.execute(
                "UPDATE tasks SET concrete_model = ?, primary_alias = ?, last_activity = ? WHERE task_id = ?",
                (concrete_model, alias, time.time(), task_id),
            )
        else:
            conn.execute(
                "UPDATE tasks SET concrete_model = ?, last_activity = ? WHERE task_id = ?",
                (concrete_model, time.time(), task_id),
            )
        conn.commit()

    def increment_counter(self, task_id: str, counter_name: str) -> int:
        conn = self._get_conn()
        with self._lock:
            conn.execute(
                """INSERT INTO counters (task_id, counter_name, value)
                   VALUES (?, ?, 1)
                   ON CONFLICT(task_id, counter_name) DO UPDATE SET value = value + 1""",
                (task_id, counter_name),
            )
            conn.commit()
            row = conn.execute(
                "SELECT value FROM counters WHERE task_id = ? AND counter_name = ?",
                (task_id, counter_name),
            ).fetchone()
            return row["value"] if row else 0

    def get_counter(self, task_id: str, counter_name: str) -> int:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value FROM counters WHERE task_id = ? AND counter_name = ?",
            (task_id, counter_name),
        ).fetchone()
        return row["value"] if row else 0

    def touch(self, task_id: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE tasks SET last_activity = ? WHERE task_id = ?",
            (time.time(), task_id),
        )
        conn.commit()

    def is_expired(self, task_id: str, ttl_seconds: int) -> bool:
        task = self.get_task(task_id)
        if task is None:
            return True
        last_activity: float = task["last_activity"]
        return (time.time() - last_activity) > ttl_seconds

    def delete_task(self, task_id: str) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM counters WHERE task_id = ?", (task_id,))
        conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
        conn.commit()

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
