"""Regression tests for the Windows file-lock crash and stale running badge."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from monitor import process_is_alive, runtime_health
from runtime_io import atomic_json, replace_with_retry


class StorageTest(unittest.TestCase):
    def test_disk_full_during_json_write_keeps_old_snapshot_and_removes_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'progress.json'
            atomic_json(path,{'step':1})
            real=tempfile.NamedTemporaryFile
            def disk_full(*args,**kwargs):
                f=real(*args,**kwargs);write=f.write
                def fail(payload):
                    write(payload[:4]);raise OSError(28,'No space left on device')
                f.write=fail
                return f
            with patch('runtime_io.tempfile.NamedTemporaryFile',side_effect=disk_full),self.assertRaises(OSError):
                atomic_json(path,{'step':20})
            self.assertEqual(json.loads(path.read_text()),{'step':1})
            self.assertEqual(list(Path(directory).glob('*.tmp')),[])

    def test_transient_permission_errors_retry_then_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'progress.json'
            atomic_json(path, {'step': 1})
            real_replace = os.replace
            calls = []

            def locked_twice(source, destination):
                calls.append(source)
                if len(calls) <= 2:
                    raise PermissionError('Windows file temporarily locked')
                return real_replace(source, destination)

            with patch('runtime_io.os.replace', side_effect=locked_twice), patch('runtime_io.time.sleep'):
                atomic_json(path, {'step': 2})
            self.assertEqual(len(calls), 3)
            self.assertEqual(json.loads(path.read_text()), {'step': 2})
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])

    def test_permanent_error_preserves_last_complete_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'progress.json'
            atomic_json(path, {'step': 1})
            with patch('runtime_io.os.replace', side_effect=PermissionError('locked')), patch('runtime_io.time.sleep'):
                with self.assertRaises(PermissionError):
                    atomic_json(path, {'step': 2})
            self.assertEqual(json.loads(path.read_text()), {'step': 1})
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])

    @unittest.skipUnless(os.name == 'nt', 'Windows sharing-lock regression')
    def test_real_windows_read_handle_delays_replacement_without_crashing(self):
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'progress.json'
            atomic_json(path, {'step': 1})
            handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 0x80, None)
            self.assertNotEqual(handle, ctypes.c_void_p(-1).value)
            timer = threading.Timer(.12, lambda: kernel.CloseHandle(handle))
            timer.start()
            started = time.monotonic()
            try:
                atomic_json(path, {'step': 2})
            finally:
                timer.join()
            self.assertGreaterEqual(time.monotonic() - started, .1)
            self.assertEqual(json.loads(path.read_text()), {'step': 2})


class HealthTest(unittest.TestCase):
    def test_actual_live_and_exited_process_detection(self):
        self.assertTrue(process_is_alive(os.getpid()))
        child = subprocess.Popen([sys.executable, '-c', 'pass'])
        child.wait(timeout=10)
        self.assertFalse(process_is_alive(child.pid))

    def test_exited_trainer_overrides_old_running_snapshot(self):
        original = {'status': 'running', 'phase': 'selfplay_taylor', 'step': 79,
                    'trainer_pid': 123, 'heartbeat_unix': time.time()}
        with patch('monitor.process_is_alive', return_value=False):
            state = runtime_health(original)
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['phase'], 'process_exited')
        self.assertEqual(original['status'], 'running')

    def test_long_live_phase_warns_without_claiming_process_died(self):
        with patch('monitor.process_is_alive', return_value=True):
            state = runtime_health({'status': 'running', 'trainer_pid': 123,
                                    'heartbeat_unix': time.time() - 120})
        self.assertEqual(state['status'], 'running')
        self.assertTrue(state['stale'])

    def test_completed_run_remains_completed_after_process_exit(self):
        with patch('monitor.process_is_alive', return_value=False):
            self.assertEqual(runtime_health({'status': 'completed'})['status'], 'completed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
