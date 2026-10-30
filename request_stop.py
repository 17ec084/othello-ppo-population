"""Request a checkpoint and stop after the pending round/save has completed."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
run=Path((ROOT/'latest_run.txt').read_text(encoding='utf-8-sig').strip())
(run/'STOP').touch()
print('Stop requested:',run)
print('Wait until progress.json has status=stopped AND phase=stopped before powering off.')
