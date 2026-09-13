"""One-shot callers receive results in-turn; interactive callers keep asynchronous dispatch."""
from types import SimpleNamespace

import pytest

from run_agent import AIAgent
from tools.delegate_tool import _model_background_value


@pytest.mark.parametrize("oneshot,depth,background", [(True, 0, False), (False, 0, True), (False, 1, False)])
def test_model_dispatch_preserves_surface_completion_contract(monkeypatch, oneshot, depth, background):
    parent = SimpleNamespace(_single_query_mode=oneshot, _delegate_depth=depth)
    calls = []

    def dispatch(**kwargs):
        calls.append(kwargs)
        return "pending handle" if kwargs["background"] else "verified child result"

    monkeypatch.setattr("tools.delegate_tool.delegate_task", dispatch)
    result = AIAgent._dispatch_delegate_task(parent, {"goal": "bounded task", "background": not background})
    assert calls[0]["background"] is background
    assert calls[0]["parent_agent"] is parent
    assert result == ("pending handle" if background else "verified child result")
    assert _model_background_value({}, parent) is background
