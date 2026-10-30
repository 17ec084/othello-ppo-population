"""Run paired REINFORCE/PPO experiments in one process, one step per method."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import traceback
from zoneinfo import ZoneInfo

from individual_train import experiment,parse_args
from runtime_io import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True);parser.add_argument('--seed',type=int,default=20261023)
    parser.add_argument('--updates',type=int,default=0);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    args=parser.parse_args();root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve().parent
    if args.resume:
        manifest=json.loads((root/'comparison.json').read_text(encoding='utf-8'))
        for entry in manifest['runs'].values():
            if not (Path(entry['path'])/'latest.pt.gz').exists():raise FileNotFoundError('A paired checkpoint is missing')
    elif (root/'comparison.json').exists():raise FileExistsError('Use a new comparison folder')
    else:
        manifest={'mode':'algorithm_comparison','reward':'margin','seed':args.seed,'models':20,'games':128,
            'eval_games':128,'initial_weights_verified':False,'single_process':True,
            'created_at':datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
            'runs':{method:{'path':str(root/method),'pid':os.getpid(),'algorithm':method,
                           'label':'従来法' if method=='reinforce' else 'PPO＋価値関数','reward':'margin'}
                    for method in ('reinforce','ppo')}}
    for entry in manifest['runs'].values():entry['pid']=os.getpid()
    atomic_json(root/'comparison.json',manifest)
    (source/'latest_comparison.txt').write_text(str(root),encoding='utf-8')
    (source/'latest_run.txt').write_text(manifest['runs']['reinforce']['path'],encoding='utf-8')
    (source/'training.pid').write_text(str(os.getpid()),encoding='utf-8')
    atomic_json(source/'training-pids.json',{'pids':[os.getpid()]})
    generators={}
    try:
        for method,entry in manifest['runs'].items():
            folder=Path(entry['path']);folder.mkdir(parents=True,exist_ok=True)
            snapshot=folder/'source';snapshot.mkdir(exist_ok=True)
            for path in source.glob('*.py'):
                if not path.name.startswith('test_'):(snapshot/path.name).write_bytes(path.read_bytes())
            if args.resume:
                config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
                cli=['--resume',str(folder/'latest.pt.gz')]
                for name in ('models','games','record_games','record_every','patience','eval_games','eval_every',
                             'save_every','width','layers','heads','grad_chunk','seed','initial_percent','lr',
                             'reward','algorithm','ppo_epochs','ppo_minibatch','ppo_clip','gae_lambda',
                             'value_coef','target_kl','initial_checkpoint'):
                    cli+=['--'+name.replace('_','-'),str(config[name] if name!='reward' else config[name]['mode'])]
                (folder/'STOP').unlink(missing_ok=True)
            else:
                cli=['--seed',str(args.seed),'--reward','margin','--algorithm',method,
                     '--initial-checkpoint',str(root/'initial.pt')]
            cli+=['--out',str(folder),'--device',args.device,'--updates',str(args.updates),'--compact-checkpoints']
            generators[method]=experiment(parse_args(cli))
        initial=[next(g) for g in generators.values()]
        manifest['initial_weights_verified']=len({r['initial_policy_sha256'] for r in initial})==1
        if not manifest['initial_weights_verified']:raise AssertionError('Initial policies differ')
        manifest['initial_policy_sha256']=initial[0]['initial_policy_sha256']
        atomic_json(root/'comparison.json',manifest)
        while generators:
            for method in list(generators):
                for other,entry in manifest['runs'].items():
                    path=Path(entry['path'])/'progress.json'
                    if other!=method and path.exists():
                        state=json.loads(path.read_text(encoding='utf-8'))
                        if state.get('status')=='running':
                            import time
                            state.update(phase='waiting_pair',heartbeat_unix=time.time())
                            atomic_json(path,state)
                try:
                    progress=next(generators[method])
                    print(json.dumps({'method':method,**progress}),flush=True)
                except StopIteration:del generators[method]
        manifest['status']='stopped_or_completed'
        atomic_json(root/'comparison.json',manifest)
    except BaseException:
        # Failure of either method halts the pair. Do not let the survivor keep
        # consuming resources while the dashboard suggests a live comparison.
        for entry in manifest['runs'].values():(Path(entry['path'])/'STOP').touch()
        manifest.update(status='failed',error=traceback.format_exc())
        try:atomic_json(root/'comparison.json',manifest)
        except OSError:pass
        raise


if __name__=='__main__':main()
