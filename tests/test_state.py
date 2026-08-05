"""Tests for the SQLite state backend."""

from __future__ import annotations

import os
import tempfile
import threading
import time

import pytest

from hermes_smart_router.state import SqliteTaskState


@pytest.fixture
def state():
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)
    s = SqliteTaskState(db_path)
    yield s
    s.close()
    os.unlink(db_path)


class TestSqliteTaskState:
    """Tests for the SQLite task state backend."""

    def test_create_and_get_task(self, state: SqliteTaskState) -> None:
        state.create_task(
            task_id="task-1",
            session_id="session-1",
            task_class="software_engineering",
            primary_alias="glm",
            escalation_alias="opus",
            concrete_model="zai/glm-5.2",
            reason_code="classifier",
        )
        task = state.get_task("task-1")
        assert task is not None
        assert task["task_id"] == "task-1"
        assert task["session_id"] == "session-1"
        assert task["task_class"] == "software_engineering"
        assert task["primary_alias"] == "glm"
        assert task["concrete_model"] == "zai/glm-5.2"

    def test_get_nonexistent_task(self, state: SqliteTaskState) -> None:
        task = state.get_task("nonexistent")
        assert task is None

    def test_update_model(self, state: SqliteTaskState) -> None:
        state.create_task(task_id="task-1", primary_alias="glm")
        state.update_model("task-1", concrete_model="anthropic/claude-opus-5", alias="opus")
        task = state.get_task("task-1")
        assert task is not None
        assert task["concrete_model"] == "anthropic/claude-opus-5"
        assert task["primary_alias"] == "opus"

    def test_increment_counter(self, state: SqliteTaskState) -> None:
        state.create_task(task_id="task-1")
        val = state.increment_counter("task-1", "failed_tool_calls")
        assert val == 1
        val = state.increment_counter("task-1", "failed_tool_calls")
        assert val == 2
        val = state.increment_counter("task-1", "failed_tool_calls")
        assert val == 3

    def test_get_counter_default(self, state: SqliteTaskState) -> None:
        val = state.get_counter("task-1", "nonexistent")
        assert val == 0

    def test_touch_updates_activity(self, state: SqliteTaskState) -> None:
        state.create_task(task_id="task-1")
        task_before = state.get_task("task-1")
        assert task_before is not None
        time.sleep(0.01)
        state.touch("task-1")
        task_after = state.get_task("task-1")
        assert task_after is not None
        assert task_after["last_activity"] > task_before["last_activity"]

    def test_is_expired(self, state: SqliteTaskState) -> None:
        state.create_task(task_id="task-1")
        # Should not be expired with large TTL
        assert not state.is_expired("task-1", 3600)
        # Should be expired with zero TTL (last_activity is in the past)
        # Actually with zero TTL, time.time() - last_activity > 0 is true
        assert state.is_expired("task-1", 0)

    def test_delete_task(self, state: SqliteTaskState) -> None:
        state.create_task(task_id="task-1")
        state.increment_counter("task-1", "counter-1")
        state.delete_task("task-1")
        assert state.get_task("task-1") is None
        assert state.get_counter("task-1", "counter-1") == 0

    def test_concurrent_increment(self, state: SqliteTaskState) -> None:
        """Test thread-safe counter increment."""
        state.create_task(task_id="task-1")
        n_threads = 10
        results: list[int] = []

        def increment() -> None:
            val = state.increment_counter("task-1", "concurrent")
            results.append(val)

        threads = [threading.Thread(target=increment) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert state.get_counter("task-1", "concurrent") == n_threads
        # Each thread should have seen a unique value
        assert sorted(results) == list(range(1, n_threads + 1))

    def test_task_ttl_expiry(self, state: SqliteTaskState) -> None:
        state.create_task(task_id="task-1")
        assert not state.is_expired("task-1", 3600)
        # Touch to reset
        state.touch("task-1")
        assert not state.is_expired("task-1", 3600)

    def test_multiple_tasks_isolation(self, state: SqliteTaskState) -> None:
        state.create_task(task_id="task-1", primary_alias="glm")
        state.create_task(task_id="task-2", primary_alias="opus")
        state.increment_counter("task-1", "failures")
        state.increment_counter("task-1", "failures")
        state.increment_counter("task-2", "failures")

        assert state.get_counter("task-1", "failures") == 2
        assert state.get_counter("task-2", "failures") == 1
