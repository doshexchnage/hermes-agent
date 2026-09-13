"""A turn deadline independent of provider activity, with cancellation scoped to that turn."""

from __future__ import annotations

import math
import threading
import time
from typing import Callable


def run_with_deadline(agent, operation: Callable[[], dict]) -> dict:
    budget = getattr(agent, "run_budget_seconds", None)
    if not isinstance(budget, (int, float)) or isinstance(budget, bool) or not math.isfinite(budget) or budget <= 0:
        return operation()

    lock = threading.Lock()
    finished = False
    expired = False
    message = "Run time budget exhausted; active work was interrupted."

    def expire():
        nonlocal expired
        # Joining this critical section before returning prevents a late timer from stopping the next turn.
        with lock:
            if finished:
                return
            expired = True
            agent.interrupt(message, hard_cancel=True, tool_reason="run time budget exhausted")

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

    if expired:
        result.update(completed=False, failed=True, interrupted=True,
                      failure_reason="run_budget_exhausted", error=message, final_response=message)
    return result
