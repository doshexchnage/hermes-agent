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
