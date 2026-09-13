"""A continuously active turn still times out; a completed turn cannot cancel a later one."""
import threading
import time
from types import SimpleNamespace

from agent.run_deadline import run_with_deadline


def test_deadline_interrupts_active_work_and_reports_failure():
    interrupted = threading.Event()
    calls = []

    def interrupt(message, **kwargs):
        calls.append((message, kwargs))
        interrupted.set()

    agent = SimpleNamespace(run_budget_seconds=0.05, interrupt=interrupt)

    def continuously_active():
        assert interrupted.wait(3), "deadline did not interrupt active work"
        return {"completed": True, "final_response": "must not pass", "api_calls": 2}

    result = run_with_deadline(agent, continuously_active)
    assert result["failed"] and not result["completed"]
    assert result["failure_reason"] == "run_budget_exhausted"
    assert result["api_calls"] == 2
    assert calls[0][1]["hard_cancel"] is True


def test_completed_turn_cancels_its_timer_before_returning():
    interrupted = threading.Event()
    agent = SimpleNamespace(run_budget_seconds=0.02, interrupt=lambda *a, **k: interrupted.set())
    expected = {"completed": True, "final_response": "accepted"}
    assert run_with_deadline(agent, lambda: expected) is expected
    time.sleep(0.06)
    assert not interrupted.is_set()
