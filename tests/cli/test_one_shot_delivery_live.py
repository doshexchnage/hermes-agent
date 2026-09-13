"""Real CLI/provider transport: child handback and cancellation on an active stream."""
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


def _completed_child(content):
    try:
        result = json.loads(content)
    except (ValueError, TypeError):
        return False
    return any(entry.get('status') == 'completed' and entry.get('summary') == 'CHILD_ACCEPTED'
               for entry in result.get('results', []))


@contextmanager
def provider_fixture(mode):
    observations = []
    cancelled = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if "model" not in body:
                self.send_error(404, "Only model inference is served by this fixture")
                return
            observations.append(body)
            worker = body['model'] == 'fixture-worker'
            tools = [m for m in body['messages'] if m['role'] == 'tool']
            if mode == 'busy' or (mode == 'busy-child' and worker):
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                try:
                    until = time.monotonic() + 15
                    while time.monotonic() < until:
                        chunk = {'id': 'busy', 'object': 'chat.completion.chunk', 'choices': [
                            {'index': 0, 'delta': {'reasoning_content': 'Still active. '}, 'finish_reason': None}]}
                        self.wfile.write(('data: ' + json.dumps(chunk) + '\n\n').encode())
                        self.wfile.flush()
                        time.sleep(0.02)
                except (BrokenPipeError, ConnectionResetError):
                    cancelled.set()
                return
            if worker:
                message = {'role': 'assistant', 'content': 'CHILD_ACCEPTED'}
            elif not tools:
                message = {'role': 'assistant', 'content': None, 'tool_calls': [
                    {'id': 'bounded-worker', 'type': 'function', 'function': {
                        'name': 'delegate_task', 'arguments': json.dumps({'goal': 'Return CHILD_ACCEPTED, no tools.'})}}]}
            else:
                accepted = any(_completed_child(m["content"]) for m in tools)
                message = {'role': 'assistant', 'content': 'HANDOFF_ACCEPTED' if accepted else 'HANDOFF_MISSING'}
            payload = {'id': 'fixture', 'object': 'chat.completion', 'model': body['model'], 'choices': [
                {'index': 0, 'message': message, 'finish_reason': 'tool_calls' if message.get('tool_calls') else 'stop'}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}
            if body.get('stream'):
                payload['object'] = 'chat.completion.chunk'
                payload['choices'][0]['delta'] = payload['choices'][0].pop('message')
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.wfile.write(('data: ' + json.dumps(payload) + '\n\ndata: [DONE]\n\n').encode())
            else:
                encoded = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', observations, cancelled
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def run_cli(tmp_path, url, quiet, budget):
    home = tmp_path / 'home'
    home.mkdir()
    config = {
        'model': {'default': 'fixture-lead', 'provider': 'custom', 'base_url': url},
        'delegation': {'model': 'fixture-worker', 'provider': 'custom', 'base_url': url,
                       'api_key': 'fixture-not-a-secret', 'max_iterations': 2, 'max_concurrent_children': 1},
        'agent': {'max_turns': 3}, 'display': {'streaming': True},
    }
    import yaml
    (home / 'config.yaml').write_text(yaml.safe_dump(config))
    env = {'PATH': os.environ['PATH'], 'HOME': str(tmp_path), 'HERMES_HOME': str(home),
           'PYTHONDONTWRITEBYTECODE': '1', 'OPENAI_API_KEY': 'fixture-not-a-secret', 'TERM': 'dumb'}
    command = [sys.executable, str(Path(__file__).resolve().parents[2] / 'hermes'), 'chat', '--oneshot',
               '--ignore-rules', '--toolsets', 'delegation', '--max-turns', '3', '--run-budget', str(budget),
               '-q', 'Delegate exactly one child, then use its returned answer.']
    if quiet:
        command.append('-Q')
    return subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=25)


@pytest.mark.parametrize('quiet', [False, True])
def test_one_shot_delivers_child_result_in_same_turn(tmp_path, quiet):
    with provider_fixture('handoff') as (url, observations, _):
        result = run_cli(tmp_path, url, quiet, 10)
    assert result.returncode == 0, result.stderr + result.stdout
    assert 'HANDOFF_ACCEPTED' in result.stdout, result.stderr + result.stdout
    assert any(body['model'] == 'fixture-worker' for body in observations)
    assert any(_completed_child(m['content']) for body in observations if body['model'] == 'fixture-lead' for m in body['messages'] if m['role'] == 'tool')


@pytest.mark.parametrize('mode', ['busy', 'busy-child'])
@pytest.mark.parametrize('quiet', [False, True])
def test_budget_cancels_active_provider_and_child_streams(tmp_path, mode, quiet):
    with provider_fixture(mode) as (url, observations, cancelled):
        started = time.monotonic()
        result = run_cli(tmp_path, url, quiet, 1)
        elapsed = time.monotonic() - started
        assert cancelled.wait(3), result.stderr + result.stdout
    assert result.returncode != 0
    assert 'budget exhausted' in result.stdout.lower(), result.stderr + result.stdout
    assert elapsed < 12
    if mode == 'busy-child':
        assert any(body['model'] == 'fixture-worker' for body in observations)


@pytest.mark.parametrize('scenario', ['late-finalizer', 'late-child', 'late-external-stop', 'early-external-stop'])
def test_deadline_turn_boundaries(tmp_path, scenario):
    """Real agents and HTTP transport; pause only at the two cancellation race boundaries."""
    home = tmp_path / 'home'
    home.mkdir()
    import yaml
    with provider_fixture('handoff') as (url, observations, _):
        (home / 'config.yaml').write_text(yaml.safe_dump({
            'delegation': {'model': 'fixture-worker', 'provider': 'custom', 'base_url': url,
                           'api_key': 'fixture-not-a-secret', 'max_iterations': 2},
        }))
        script = r'''
import sys, threading, time
from run_agent import AIAgent
from hermes_state import SessionDB
from pathlib import Path
from tools import delegate_tool
from tools.interrupt import is_interrupted
url, scenario, home = sys.argv[1:]
db = SessionDB(Path(home) / 'state.db')
agent = AIAgent(api_key='fixture-not-a-secret', provider='custom', base_url=url,
    model='fixture-worker' if scenario != 'late-child' else 'fixture-lead',
    enabled_toolsets=[] if scenario != 'late-child' else ['delegation'],
    quiet_mode=True, skip_context_files=True, skip_memory=True, skip_background_review=True,
    max_iterations=3, session_db=db)
agent.run_budget_seconds = 0.3
expired = threading.Event()
original_interrupt = agent.interrupt
from agent.interrupt_control import InterruptControlMixin
original_control_interrupt = InterruptControlMixin.interrupt
def observed_interrupt(target, *args, **kwargs):
    result = original_control_interrupt(target, *args, **kwargs)
    if target is agent and kwargs.get('unless_interrupted'):
        expired.set()
    return result
InterruptControlMixin.interrupt = observed_interrupt
if scenario == 'late-child':
    attach = delegate_tool._attach_child
    def delayed_attach(parent, child):
        assert expired.wait(3), 'deadline never fired before attachment'
        attach(parent, child)
    delegate_tool._attach_child = delayed_attach
else:
    sync = agent._sync_external_memory_for_turn
    def delayed_sync(**kwargs):
        assert not agent._interrupt_requested, 'did not reach post-clear finalization'
        if scenario == 'early-external-stop':
            original_interrupt('user stop', hard_cancel=True)
        assert expired.wait(3), 'deadline never fired in finalization'
        if scenario == 'late-external-stop':
            agent.interrupt('user stop', hard_cancel=True)
    agent._sync_external_memory_for_turn = delayed_sync
result = agent.run_conversation('Return the accepted answer; delegate if available.')
assert result['failed'] and result['failure_reason'] == 'run_budget_exhausted', result
assert result['turn_exit_reason'] == 'run_budget_exhausted', result
if scenario in ('late-external-stop', 'early-external-stop'):
    assert agent._interrupt_requested and agent._interrupt_message == 'user stop'
else:
    assert not agent._interrupt_requested, 'deadline interrupt leaked into cached next turn'
    assert not agent._hard_interrupt_requested.is_set()
    assert not is_interrupted()
    if scenario == 'late-finalizer':
        agent._sync_external_memory_for_turn = sync
        agent.run_budget_seconds = None
        second = agent.run_conversation('Return the accepted answer again.')
        assert second['completed'] and second['final_response'] == 'CHILD_ACCEPTED', second
agent.clear_interrupt()
agent.close()
db.close()
'''
        env = {'PATH': os.environ['PATH'], 'HOME': str(tmp_path), 'HERMES_HOME': str(home),
               'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(Path(__file__).resolve().parents[2])}
        result = subprocess.run([sys.executable, '-c', script, url, scenario, str(home)],
                                cwd=tmp_path, env=env, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    if scenario == 'late-child':
        assert not any(body['model'] == 'fixture-worker' for body in observations), 'cancelled child started inference'
    if scenario == 'late-finalizer':
        assert any(m.get('content') == 'Return the accepted answer again.'
                   for body in observations for m in body['messages'] if m['role'] == 'user')


def test_redirect_cannot_resume_inference_after_run_deadline(tmp_path):
    home = tmp_path / 'home'
    home.mkdir()
    script = r'''
import threading, sys, time
from run_agent import AIAgent
agent = AIAgent(api_key='fixture-not-a-secret', provider='custom', base_url=sys.argv[1],
    model='fixture-worker', enabled_toolsets=[], quiet_mode=True, skip_context_files=True,
    skip_memory=True, skip_background_review=True, max_iterations=3)
agent.run_budget_seconds = 0.4
expired = threading.Event()
original_interrupt = agent.interrupt
original_clear = agent.clear_interrupt
from agent.interrupt_control import InterruptControlMixin
original_control_interrupt = InterruptControlMixin.interrupt
def observed_interrupt(target, *a, **kw):
    result = original_control_interrupt(target, *a, **kw)
    if target is agent and kw.get('unless_interrupted'):
        print('DEADLINE_INTERRUPT_ACCEPTED', result, flush=True)
        expired.set()
    return result
def delayed_clear(*a, **kw):
    if kw.get('preserve_redirect'):
        assert expired.wait(3), 'deadline never arrived while redirect was pending'
    return original_clear(*a, **kw)
InterruptControlMixin.interrupt = observed_interrupt
agent.clear_interrupt = delayed_clear
def steer():
    assert agent._model_request_active.wait(3)
    time.sleep(0.1)
    assert agent.redirect('Continue using the correction')
rescued = threading.Event()
def rescue():
    assert expired.wait(3)
    time.sleep(1.0)
    rescued.set()
    original_interrupt('test rescue stop', hard_cancel=True)
threading.Thread(target=steer, daemon=True).start()
threading.Thread(target=rescue, daemon=True).start()
result = agent.run_conversation('Keep working')
assert not rescued.is_set(), 'expired deadline needed a second stop to settle the turn'
assert result.get('failure_reason') == 'run_budget_exhausted', result
assert result.get('turn_exit_reason') == 'run_budget_exhausted', result
assert result.get('failed') and not result.get('completed'), result
agent.close()
'''
    with provider_fixture('busy') as (url, observations, cancelled):
        env = {'PATH': os.environ['PATH'], 'HOME': str(tmp_path), 'HERMES_HOME': str(home),
               'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(Path(__file__).resolve().parents[2])}
        result = subprocess.run([sys.executable, '-c', script, url], cwd=tmp_path, env=env,
                                text=True, capture_output=True, timeout=12)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(observations) == 1, f'Inference resumed after expired deadline: {len(observations)} HTTP requests\n{result.stdout}\n{result.stderr}'


def test_deadline_cancels_legacy_agent_override(tmp_path):
    home = tmp_path / 'home'
    home.mkdir()
    script = r'''
import threading, sys
from run_agent import AIAgent
class LegacyAgent(AIAgent):
    def interrupt(self, message=None):
        raise AssertionError('deadline dispatched through the legacy override')
    def clear_interrupt(self):
        return super().clear_interrupt()
agent = LegacyAgent(api_key='fixture-not-a-secret', provider='custom', base_url=sys.argv[1],
    model='fixture-worker', enabled_toolsets=[], quiet_mode=True, skip_context_files=True,
    skip_memory=True, skip_background_review=True, max_iterations=3, run_budget_seconds=0.3)
rescued = threading.Event()
def rescue():
    rescued.set()
    agent.hard_interrupt('test rescue stop')
timer = threading.Timer(2, rescue)
timer.daemon = True
timer.start()
try:
    result = agent.run_conversation('Keep working')
    assert not rescued.is_set(), 'deadline failed to cancel a legacy subclass'
    assert result.get('failed') and not result.get('completed'), result
    assert result.get('failure_reason') == 'run_budget_exhausted', result
finally:
    timer.cancel()
    timer.join()
    agent.close()
'''
    with provider_fixture('busy') as (url, observations, cancelled):
        env = {'PATH': os.environ['PATH'], 'HOME': str(tmp_path), 'HERMES_HOME': str(home),
               'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(Path(__file__).resolve().parents[2])}
        result = subprocess.run([sys.executable, '-c', script, url], cwd=tmp_path, env=env,
                                text=True, capture_output=True, timeout=8)
        assert cancelled.wait(3), result.stdout + result.stderr
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(observations) == 1
