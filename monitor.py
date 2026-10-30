"""Read-only local experiment monitor. Never exposes arbitrary filesystem paths."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import time
from urllib.parse import parse_qs,urlsplit

ROOT = Path(__file__).resolve().parent


def run_directory():
    return Path((ROOT / 'latest_run.txt').read_text(encoding='utf-8').strip())


def timeline_events(out, completed_step):
    events = []
    for name, kind in (('switches.jsonl', 'learner_switch'),
                       ('plateau_events.jsonl', 'mutation'), ('copies.jsonl', 'opponent_copy'),
                       ('alien_sessions.jsonl', 'alien_learning'),
                       ('sharing_sessions.jsonl', 'shared_learning')):
        path = out/name
        if not path.exists():
            continue
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event['step'] > completed_step:
                continue
            if name == 'plateau_events.jsonl' and event.get('type') != 'mutation':
                continue
            events.append({**event, 'type': kind})
    # At a single update a pre-training switch can precede an after-training
    # copy/switch. Preserve that order when reconstructing the colored spans.
    return sorted(events, key=lambda e: (e['step'], e.get('timing') != 'before_update'))


def process_is_alive(pid):
    """Query process state without signaling anything on Windows."""
    if not pid or pid <= 0:
        return False
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return None if ctypes.get_last_error() == 5 else False
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                return None
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return None


def runtime_health(state, fallback_pid=None, fallback_heartbeat=None):
    """Keep an old 'running' snapshot from impersonating a live trainer."""
    state = dict(state)
    age = max(0., time.time() - state.get('heartbeat_unix', fallback_heartbeat or time.time()))
    state['heartbeat_age_seconds'] = round(age, 1)
    state['stale'] = state.get('status') in ('running','waiting_storage') and age > 60
    state['trainer_alive'] = process_is_alive(state.get('trainer_pid', fallback_pid))
    if state.get('status') in ('running','waiting_storage') and state['trainer_alive'] is False:
        state['recorded_status'] = state['status']
        state['status'] = 'failed'
        state['phase'] = 'process_exited'
        state['error'] = ('学習プロセスが終了しています。最後に保存された進捗を表示しています。'
                          '詳細は実験フォルダの stderr.log を確認してください。')
    return state


def read_run_state(out,fallback_pid=None):
    progress_path=out/'progress.json'
    state=json.loads(progress_path.read_text(encoding='utf-8'))
    state=runtime_health(state,fallback_pid,progress_path.stat().st_mtime)
    history=[]
    if (out/'evaluations.jsonl').exists():
        for line in (out/'evaluations.jsonl').read_text(encoding='utf-8').splitlines():
            try:history.append(json.loads(line))
            except json.JSONDecodeError:pass
    state['evaluation_history']=history
    state['timeline_events']=timeline_events(out,state['step'])
    state['last_games']={}
    for family in state['models']:
        path=out/f'last_game_{family}.json'
        if path.exists():state['last_games'][family]=json.loads(path.read_text(encoding='utf-8'))
    return state


def active_comparison():
    pointer=ROOT/'latest_comparison.txt'
    if not pointer.exists():return None
    folder=Path(pointer.read_text(encoding='utf-8-sig').strip())
    if not (folder/'comparison.json').exists():return None
    manifest=json.loads((folder/'comparison.json').read_text(encoding='utf-8-sig'))
    paths=[Path(v['path']) for v in manifest['runs'].values()]
    return manifest if run_directory() in paths else None


def comparison_payload(manifest):
    runs={}
    for reward,entry in manifest['runs'].items():
        path=Path(entry['path'])
        if not (path/'progress.json').exists():
            runs[reward]={'status':'initializing','step':0,'evaluation_history':[],
                         'algorithm':entry.get('algorithm'),
                         'label':entry.get('label','石差' if reward=='margin' else '勝+1・負−1・分0')};continue
        state=read_run_state(path,entry['pid'])
        runs[reward]={k:state.get(k) for k in ('status','step','phase','elapsed_seconds','controller',
                      'unfreezing','evaluation','evaluation_history','training_games','error','algorithm','model_count')}
        runs[reward]['label']=entry.get('label','石差' if reward=='margin' else '勝+1・負−1・分0')
    maps=[{r['step']:r for r in run['evaluation_history']} for run in runs.values()]
    common=set.intersection(*(set(m) for m in maps)) if maps else set()
    matched=max(common) if common else None
    for run,mapping in zip(runs.values(),maps):
        run['matched_evaluation']=mapping.get(matched)
        run['baseline_evaluation']=run['evaluation_history'][0] if run['evaluation_history'] else None
    return {'runs':runs,'mode':manifest.get('mode','reward_comparison'),
            'models':manifest.get('models',20),'matched_step':matched,'eval_games':manifest['eval_games'],
            'initial_weights_verified':manifest.get('initial_weights_verified',False)}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = self.path.split('?')[0]
        try:
            if route in ('/', '/index.html'):
                content = (ROOT / 'monitor.html').read_bytes()
                content_type = 'text/html; charset=utf-8'
            elif route in ('/monitor.js','/individual-monitor.js','/reward-comparison.js'):
                content = (ROOT / route.lstrip('/')).read_bytes()
                content_type = 'text/javascript; charset=utf-8'
            elif route == '/api/state':
                out = run_directory()
                manifest=active_comparison()
                reward=parse_qs(urlsplit(self.path).query).get('reward',[next(iter(manifest['runs'])) if manifest else 'margin'])[0]
                if manifest and reward in manifest['runs']:out=Path(manifest['runs'][reward]['path'])
                pid_path = ROOT / 'training.pid'
                fallback_pid = int(pid_path.read_text().strip()) if pid_path.exists() else None
                if manifest and reward in manifest['runs']:fallback_pid=manifest['runs'][reward]['pid']
                state=read_run_state(out,fallback_pid)
                if manifest:state['comparison']=comparison_payload(manifest)
                content = json.dumps(state, ensure_ascii=False).encode('utf-8')
                content_type = 'application/json; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(content)
        except (OSError, ValueError) as error:
            self.send_error(503, str(error))

    def log_message(self, *_):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    print(f'http://127.0.0.1:{args.port}/', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
