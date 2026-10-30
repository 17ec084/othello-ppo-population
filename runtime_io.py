"""Atomic file updates that tolerate brief Windows reader/antivirus locks."""
import os
from pathlib import Path
import tempfile
import time
import json


def replace_with_retry(source, destination, attempts=14):
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(.01 * 2 ** attempt, .25))


def atomic_json(path, value):
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as file:
            temp = Path(file.name)
            file.write(payload)
        replace_with_retry(temp, path)
    finally:
        try:
            if temp is not None:temp.unlink(missing_ok=True)
        except OSError:
            pass
