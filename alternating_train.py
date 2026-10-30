"""Alternating Transformer training with anytime-valid phase switches.

Default updates=0 means no update limit. Only the active learner is updated.
Training reward is raw terminal stone margin. Independent, color-balanced
evaluation games provide one-sided phase-level betting evidence for switching.
"""
import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from curriculum import ProgressiveUnfreezer
from model import ModelConfig, OthelloGPT
from adaptive import AdaptiveController, copy_opponent
from population_control import PopulationController
from entropy import probe_positions, measure_entropy
from runtime_io import atomic_json, replace_with_retry
from train import evaluate, play_batch, update_policy


def set_roles(models, learner):
    for agent, model in enumerate(models):
        model.requires_grad_(agent == learner)
        if agent != learner:
            for parameter in model.parameters():
                parameter.grad = None


def main(entropy_mode=False):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--updates', type=int, default=0, help='0 = unlimited, only STOP ends the experiment')
    p.add_argument('--games-per-batch', type=int, default=128)
    p.add_argument('--test-games', type=int, default=200 if entropy_mode else 64)
    p.add_argument('--eval-games', type=int, default=64)
    p.add_argument('--eval-every', type=int, default=25)
    p.add_argument('--save-every', type=int, default=10)
    p.add_argument('--alpha', type=float, default=.05)
    p.add_argument('--initial-percent', type=float, default=.1)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--width', type=int, default=96)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--grad-chunk', type=int, default=256)
    p.add_argument('--entropy-coef', type=float, default=.5 if entropy_mode else 0.)
    p.add_argument('--entropy-positions', type=int, default=128)
    p.add_argument('--seed', type=int, default=20261023 if entropy_mode else 20261022)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--out', default='')
    p.add_argument('--resume', default='')
    p.add_argument('--publish', action='store_true')
    args = p.parse_args()
    entropy_mode=entropy_mode or args.entropy_coef>0
    protocol='alternating-entropy-margin-v1' if entropy_mode else 'alternating-anytime-copy-v3'
    if (args.updates < 0 or min(args.eval_every, args.save_every, args.grad_chunk) < 1
            or args.patience < 1 or args.entropy_positions<1
            or not np.isfinite(args.entropy_coef) or args.entropy_coef<0
            or not 0 < args.alpha < 1 or args.lr <= 0 or not .1 <= args.initial_percent <= 100
            or abs(args.initial_percent*10 - round(args.initial_percent*10)) > 1e-9
            or any(n < 2 or n % 2 for n in (args.games_per_batch, args.test_games, args.eval_games))):
        p.error('Invalid configuration (games must be even; initial percent in 0.1 increments)')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    if device == 'auto':
        device = 'cpu'
    out = Path(args.out or ('runs/alternating-' + datetime.now().strftime('%Y%m%d-%H%M%S'))).resolve()
    out.mkdir(parents=True, exist_ok=True)
    args.out = str(out)
    if (out/'config.json').exists() and not args.resume:
        p.error('Choose a new output directory or explicitly resume')
    models = [OthelloGPT(ModelConfig(args.width, args.layers, args.heads)).to(device) for _ in range(2)]
    initial_models = [copy.deepcopy(model).eval().requires_grad_(False) for model in models]
    optimizers = [torch.optim.Adam(model.parameters(), lr=args.lr) for model in models]
    baseline = [[0., 0.], [0., 0.]]
    mask = ProgressiveUnfreezer(models[0], args.seed+500, initial_percent=args.initial_percent,
                               min_hold_updates=0, granularity=10)
    controller = (PopulationController if entropy_mode else AdaptiveController)(args.alpha, args.patience)
    train_rng = np.random.default_rng(args.seed)
    test_rng = np.random.default_rng(args.seed+700000)
    step, training_games, test_games, elapsed_prior = 0, 0, 0, 0.
    learner_steps=[0,0]
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved.get('protocol') != protocol:
            p.error('This checkpoint belongs to a different experiment protocol')
        old = saved['args']
        for key in ('width','layers','heads','lr','seed','games_per_batch','test_games','alpha','initial_percent',
                    'patience'):
            if old[key] != getattr(args, key):
                p.error(f'Resume must preserve {key}={old[key]}')
        if old.get('entropy_coef',0.)!=args.entropy_coef or old.get('entropy_positions',128)!=args.entropy_positions:
            p.error('Resume must preserve entropy reward and diagnostic configuration')
        if Path(old['out']).resolve() != out:
            p.error('Resume must use the original output directory')
        for agent in (0, 1):
            models[agent].load_state_dict(saved['models'][agent])
            optimizers[agent].load_state_dict(saved['optimizers'][agent])
        baseline = saved['baselines']
        mask.load_state_dict(saved['mask'])
        controller.load_state_dict(saved['controller'])
        train_rng.bit_generator.state = saved['train_rng']
        test_rng.bit_generator.state = saved['test_rng']
        step, training_games, test_games = saved['step'], saved['training_games'], saved['test_games']
        elapsed_prior = saved['elapsed']
        learner_steps=saved.get('learner_steps',[0,0])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device == 'cuda' and saved['cuda_rng']:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
        # Keep append-only diagnostics consistent with the last saved model.
        for name in ('metrics.jsonl','evaluations.jsonl','switches.jsonl','copies.jsonl'):
            path = out/name
            if path.exists():
                rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
                path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows if r['step'] <= step),encoding='utf-8')
    else:
        torch.save({'transformer':[m.state_dict() for m in initial_models]},out/'initial.pt')
    set_roles(models,controller.learner)
    metadata = {**vars(args), 'protocol':protocol, 'updates_limit':args.updates or None,
                'reward':{'mode':'margin','formula':'own_stones - opponent_stones'},
                'device_used':device,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,
                'models':{'transformer':{**models[0].description(),'all_parameters_trainable':False}},'torch':torch.__version__,
                'unfreezing':'0.1 percentage points only on B-to-A switches; same scalar mask for A and B',
                'plateau_recovery':{'patience':args.patience,'mutations_enabled':False,
                                    'timeout_timing':f'before training at t+{args.patience}',
                                    'forced_switch':'keep current weights; no rollback; not a significance decision'},
                'inferiority_copy':{'threshold':args.alpha,'scope':'all model weights, including frozen entries',
                                    'after_copy':'same learner; reset both tests, plateau clock, Adam and reward baseline'},
                'test':{'alpha':args.alpha,'type':'one-sided bounded mixture betting e-process',
                        'unit':'two games with colors reversed; draw=half score',
                        'null':'conditional expected learner pair score <= 0.5 throughout this phase',
                        'changing_learner':'predictable learning between held-out evaluation batches',
                        'sampling':'fresh randomness, standard board, temperature 1',
                        'inferiority':'separate one-sided test on negated learner payoff',
                        'reset':'on learner switch or copy; alpha per direction per epoch, not global familywise',
                        'source':'https://arxiv.org/abs/2210.01948'}}
    if entropy_mode:
        metadata['entropy_reward']={'coefficient':args.entropy_coef,
            'formula':'terminal stone margin + coefficient * sum_t H(legal policy at s_t)/log(number of legal moves)',
            'forced_moves':'zero bonus when only one legal move; pass is not a sampled action',
            'gradient':'future entropy rewards in REINFORCE returns plus explicit current entropy derivative',
            'scope':'active learner only; no population, sharing, or alien-learning loops',
            'source':'https://proceedings.mlr.press/v97/ahmed19a.html'}
        metadata['test'].update(type='one-sided Hoeffding exponential mixture on held-out win signs',
                                unit='independent color-balanced pairs, batch update; win +1, draw 0, loss -1')
    atomic_json(out/'config.json',metadata)
    started=time.monotonic()
    state={'status':'running','step':step,'updates':args.updates or None,'phase':'initializing',
           'run_dir':str(out),'models':metadata['models'],'reward':metadata['reward'],
           'device':metadata['gpu'] or device,'evaluation':None,'latest_training':{}}
    if entropy_mode:
        entropy_tokens,entropy_masks=probe_positions(args.seed+9000,args.entropy_positions)
        entropy_stats=[measure_entropy(m,entropy_tokens,entropy_masks) for m in models]
        state['entropy_reward']=metadata['entropy_reward']
        state['batch_games']=args.games_per_batch
        state['test_batch_games']=args.test_games
    if args.resume and (out/'evaluations.jsonl').exists():
        lines=(out/'evaluations.jsonl').read_text(encoding='utf-8').splitlines()
        if lines: state['evaluation']=json.loads(lines[-1])

    def append(name,row):
        with (out/name).open('a',encoding='utf-8') as file:
            file.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

    def publish(phase):
        state.update(phase=phase,step=step,elapsed_seconds=round(elapsed_prior+time.monotonic()-started,2),
                     trainer_pid=os.getpid(),heartbeat_unix=time.time(),
                     training_games={'transformer':training_games},test_games=test_games,
                     alternating=controller.summary(),unfreezing={'transformer':mask.summary()},
                     plateau=controller.plateau_summary(step))
        if entropy_mode:
            state.update(entropy={name:entropy_stats[i] for i,name in enumerate('AB')},
                experience={name:{'competition_steps':learner_steps[i],
                                  'training_games_per_individual':learner_steps[i]*args.games_per_batch}
                            for i,name in enumerate('AB')})
        atomic_json(out/'progress.json',state)

    def checkpoint():
        data={'protocol':protocol,'args':vars(args),'step':step,'learner_steps':learner_steps,
              'elapsed':elapsed_prior+time.monotonic()-started,'training_games':training_games,
              'test_games':test_games,'models':[m.state_dict() for m in models],
              'optimizers':[o.state_dict() for o in optimizers],'baselines':baseline,
              'mask':mask.state_dict(),'controller':controller.state_dict(),
              'train_rng':train_rng.bit_generator.state,'test_rng':test_rng.bit_generator.state,
              'torch_rng':torch.get_rng_state(),
              'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else []}
        torch.save(data,out/'latest.tmp')
        replace_with_retry(out/'latest.tmp',out/'latest.pt')

    def diagnostic_evaluation():
        publish('evaluation_fixed')
        result=evaluate({'transformer':{'models':models}}, {'transformer':initial_models},
                        args.eval_games,args.seed+900000,device,step)
        state['evaluation']=result
        append('evaluations.jsonl',result)
        publish('evaluation_complete')

    try:
        publish('initializing')
        # Only production launches select the monitor; tests cannot replace it.
        if args.publish:
            Path(__file__).with_name('latest_run.txt').write_text(str(out),encoding='utf-8')
        if not args.resume:
            checkpoint()
            diagnostic_evaluation()
        while args.updates==0 or step < args.updates:
            if (out/'STOP').exists():
                state['status']='stopped'
                break
            next_step=step+1
            before_event=controller.before_step(next_step,mask)
            if before_event:
                set_roles(models,controller.learner)
                append('switches.jsonl',before_event)
            learner=controller.learner
            publish('selfplay_transformer')
            match=play_batch(models,args.games_per_batch,train_rng,device,collect=True,record=True,reward_mode='margin')
            publish('update_transformer')
            update=update_policy(models[learner],optimizers[learner],match['records'][learner],
                                 match['rewards'][learner],match['black_agents'],learner,
                                 baseline[learner],args.grad_chunk,unfreezer=mask,entropy_coef=args.entropy_coef)
            training_games+=args.games_per_batch
            learner_steps[learner]+=1
            # Held-out trajectories: never used for gradients or baselines.
            publish('evaluation_switch')
            match_test=play_batch(models,args.test_games,test_rng,device,reward_mode='margin')
            test_games+=args.test_games
            outcomes=match_test['rewards'][0]
            event=controller.observe(next_step,outcomes.reshape(-1,2) if entropy_mode else outcomes,mask)
            if event:
                if event['type']=='opponent_copy':
                    copy_opponent(models,optimizers,baseline,learner)
                    append('copies.jsonl',event)
                else:
                    set_roles(models,controller.learner)
                    append('switches.jsonl',event)
            if entropy_mode:
                entropy_stats=[measure_entropy(m,entropy_tokens,entropy_masks) for m in models]
            step=next_step
            row={'step':step,'learner':'AB'[learner],'a_wins':match['a_wins'],'b_wins':match['b_wins'],
                 'draws':match['draws'],'games':match['games'],
                 'a_mean_reward':float(match['rewards'][0].mean()),
                 'b_mean_reward':float(match['rewards'][1].mean()),'agents':{'AB'[learner]:update},
                 'active_percent':mask.summary()['percent'],'test':controller.test.summary(),
                 'loss_test':controller.loss_test.summary(),
                 'switch':event if event and event['type']=='learner_switch' else before_event,
                 'copy':event if event and event['type']=='opponent_copy' else None,
                 'step_events':[e for e in (before_event,event) if e],
                 'plateau':controller.plateau_summary(step),
                 'training_games':training_games,'test_games':test_games}
            if entropy_mode:
                row['entropy']={name:entropy_stats[i] for i,name in enumerate('AB')}
            append('metrics.jsonl',row)
            state['latest_training']={'transformer':row}
            atomic_json(out/'last_game_transformer.json',{'step':step,'family':'transformer','moves':match['trace'],
                'black_stones':int(int(match['black'][0]).bit_count()),'white_stones':int(int(match['white'][0]).bit_count())})
            if event or before_event or step%args.save_every==0:
                checkpoint()
            publish('batch_complete')
            print(json.dumps({'event':'training','step':step,'learner':'AB'[learner],
                              'next_learner':'AB'[controller.learner],'percent':mask.summary()['percent'],
                              'p':controller.test.p_value,'loss_p':controller.loss_test.p_value,
                              'switched':bool(row['switch']),'copied':bool(row['copy']),
                              'plateau_t':controller.t}),flush=True)
            if step%args.eval_every==0 and not (out/'STOP').exists():
                diagnostic_evaluation()
        else:
            state['status']='completed'
        checkpoint()
        publish(state['status'])
    except KeyboardInterrupt:
        state.update(status='interrupted',error='Resume latest.pt; use STOP for a graceful checkpoint.')
        publish('interrupted')
    except Exception:
        state.update(status='failed',error=traceback.format_exc())
        try: publish('failed')
        except OSError: print(state['error'],file=sys.stderr,flush=True)
        raise


if __name__=='__main__':
    main()
