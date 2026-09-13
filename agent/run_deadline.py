"""A turn deadline independent of provider activity, with cancellation scoped to that turn."""

from __future__ import annotations

import math
import threading
import time
from typing import Callable

from agent.interrupt_control import InterruptControlMixin


def run_with_deadline(agent, operation: Callable[[], dict]) -> dict:
    budget = getattr(agent, "run_budget_seconds", None)
    if not isinstance(budget, (int, float)) or isinstance(budget, bool) or not math.isfinite(budget) or budget <= 0:
        return operation()

    # Internal cancellation must not pass new keywords through legacy public overrides.
    if isinstance(agent, InterruptControlMixin):
        interrupt = InterruptControlMixin.interrupt.__get__(agent)
        clear = InterruptControlMixin.clear_interrupt.__get__(agent)
    else:
        interrupt = agent.interrupt
        clear = getattr(agent, "clear_interrupt", None)

    lock = threading.Lock()
    finished = False
    expired = False
    # A fresh string identity distinguishes this timer from external interrupt publications.
    message = " ".join(("Run time budget exhausted;", "active work was interrupted."))

    def expire():
        nonlocal expired
        # Joining this critical section before returning prevents a late timer from stopping the next turn.
        with lock:
            if finished:
                return
            expired = True
            interrupt(message, hard_cancel=True, tool_reason="run time budget exhausted",
                      unless_interrupted=True)

    started = getattr(agent, "_run_budget_started_at", None)
    elapsed = max(0.0, time.time() - started) if isinstance(started, (int, float)) else 0.0
    timer = threading.Timer(max(0.0, budget - elapsed), expire)
    timer.daemon = True
    timer.start()
    try:
        result = operation()
    finally:
        with lock:
            finished = True
        timer.cancel()
        timer.join()
        if expired and callable(clear):
            clear(expected_message=message)

    if expired:
        result.update(completed=False, failed=True, interrupted=True,
                      failure_reason="run_budget_exhausted", turn_exit_reason="run_budget_exhausted",
                      error=message, final_response=message)
    return result
