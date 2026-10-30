"""Portable launcher for the step-4085 experiment, with a small CPU smoke run."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true',help='Small CPU validation; not the article conditions')
    parser.add_argument('--updates',type=int,default=None,help='Total completed rounds, not additional rounds; 0 means unlimited')
    parser.add_argument('--out',type=Path,default=None)
    parser.add_argument('--resume',type=Path,default=None,help='Existing run directory')
    parser.add_argument('--device',choices=['cpu','cuda','auto'],default=None)
    args=parser.parse_args()
    if args.smoke and args.resume:parser.error('--smoke and --resume cannot be combined')
    if args.updates is not None and args.updates<0:parser.error('--updates must be >= 0')
    if args.resume:
        run=args.resume.resolve()
        config=json.loads((run/'config.json').read_text(encoding='utf-8'))
        if config.get('protocol')!='individual-pool-winrate-full-v1':parser.error('Different training protocol')
        checkpoint=run/'latest.pt.gz'
        if not checkpoint.exists() or not (run/'initial.pt').exists():parser.error('Both initial.pt and latest.pt.gz are required')
        command=[sys.executable,'-u',str(ROOT/'individual_train.py'),'--selection','win_rate',
                 '--reward',config['reward']['mode'],'--compact-checkpoints','--publish',
                 '--out',str(run),'--resume',str(checkpoint)]
        keys=['models','games-per-opponent','win-threshold','initial-percent','algorithm','record-games',
              'record-every','patience','eval-games','eval-every','save-every','width','layers','heads',
              'grad-chunk','seed','updates','lr','ppo-epochs','ppo-minibatch','ppo-clip','gae-lambda',
              'value-coef','target-kl','min-free-gib','device']
        for key in keys:
            value=config[key.replace('-','_')]
            if key=='updates' and args.updates is not None:value=args.updates
            if key=='device' and args.device is not None:value=args.device
            command += ['--'+key,str(value)]
        (run/'STOP').unlink(missing_ok=True)
    else:
        run=(args.out or ROOT/'runs'/('smoke' if args.smoke else 'reproduction-4085')).resolve()
        if (run/'config.json').exists():parser.error('Run already exists; choose --out or use --resume')
        free=shutil.disk_usage(ROOT).free
        if free<(256*1024**2 if args.smoke else 4*1024**3):parser.error('Free space needed: smoke 256 MiB; full run 4 GiB')
        command=[sys.executable,'-u',str(ROOT/'individual_train.py'),'--selection','win_rate','--algorithm','ppo',
            '--reward','signed_win','--models','3' if args.smoke else '20',
            '--games-per-opponent','4' if args.smoke else '128','--win-threshold','0.55',
            '--initial-percent','100','--width','16' if args.smoke else '96',
            '--layers','1' if args.smoke else '4','--heads','2' if args.smoke else '4',
            '--eval-games','2' if args.smoke else '128','--eval-every','25',
            '--ppo-epochs','2' if args.smoke else '4','--ppo-minibatch','64' if args.smoke else '512',
            '--grad-chunk','32' if args.smoke else '256','--lr','0.0001','--ppo-clip','0.2',
            '--gae-lambda','0.95','--value-coef','0.5','--target-kl','0.02',
            '--seed','20261028','--save-every','1','--min-free-gib','0.125' if args.smoke else '2',
            '--compact-checkpoints','--publish','--out',str(run),
            '--device',args.device or ('cpu' if args.smoke else 'auto'),
            '--updates',str(args.updates if args.updates is not None else (2 if args.smoke else 4085))]
    print('Run:',run,flush=True)
    return subprocess.call(command,cwd=ROOT)


if __name__=='__main__':raise SystemExit(main())
