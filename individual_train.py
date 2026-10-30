"""Twenty independent models: one learner, nineteen frozen random opponents."""
import argparse
import copy
import errno
import hashlib
from datetime import datetime
import json
import os
import shutil
from pathlib import Path
import time
import traceback
import sys

import numpy as np
import torch

from curriculum import ProgressiveUnfreezer
from individual_control import IndividualController,WinRateController
from model import ModelConfig,OthelloGPT
from othello import U64,initial_boards,legal_moves,row_tokens,bits_to_mask,apply_moves,stone_counts
from runtime_io import atomic_json,replace_with_retry
from train import play_batch,policy_logits,update_policy
from ppo import value_head,update_ppo,rollout_digest
import compact_checkpoint

PROTOCOL='individual-pool-no-rollback-v1'
WIN_RATE_PROTOCOL='individual-pool-winrate-full-v1'


def balanced_opponents(size,learner,games_per_opponent,rng):
    if size<2 or not 0<=learner<size or games_per_opponent<2 or games_per_opponent%2:
        raise ValueError('Use an even number of games per other individual')
    others=np.flatnonzero(np.arange(size)!=learner)
    # Even/odd game indices are black/white. Shuffle within each color so
    # every opponent gets exactly half of its games in each role.
    result=np.empty((size-1)*games_per_opponent,dtype=int)
    for color in (0,1):result[color::2]=rng.permutation(np.repeat(others,games_per_opponent//2))
    return result


def save_checkpoint(path,data):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    try:
        torch.save(data,temporary)
        replace_with_retry(temporary,path)
    finally:
        try:temporary.unlink(missing_ok=True)
        except OSError:pass


def training_rewards(margins,mode):
    margins=np.asarray(margins)
    if not np.isfinite(margins).all():raise ValueError('Rewards must be finite')
    if mode=='margin':return margins.copy()
    if mode=='signed_win':return np.sign(margins)
    raise ValueError('Unknown reward mode')


def play_individual(models,learner,games,rng,collect=False,record=False,opponents=None,heartbeat=None):
    """The opponent is chosen once per game and remains fixed during that game."""
    if games<2 or games%2:raise ValueError('Even games required')
    available=np.flatnonzero(np.arange(len(models))!=learner)
    opponents=rng.choice(available,games) if opponents is None else np.asarray(opponents,dtype=int)
    if opponents.shape!=(games,) or (opponents==learner).any() or (opponents<0).any() or (opponents>=len(models)).any():
        raise ValueError('Opponents must be other model IDs')
    device=next(models[learner].parameters()).device
    black,white=initial_boards(games)
    black_agents=np.arange(games)%2
    done=np.zeros(games,dtype=bool)
    records=[];trace=[]
    for turn in range(122):
        if heartbeat and turn%10==0:heartbeat()
        color=turn%2
        own,opp=(black,white) if color==0 else (white,black)
        legal=legal_moves(own,opp)
        done|=(legal==0)&(legal_moves(opp,own)==0)
        if done.all():break
        players=black_agents if color==0 else 1-black_agents
        playable=(~done)&(legal!=0)
        moves=np.zeros(games,dtype=np.uint64)
        if record and not done[0] and not playable[0]:trace.append({'color':color,'agent':int(players[0]),'action':None})
        for side in (0,1):
            ids=np.flatnonzero(playable&(players==side))
            if not len(ids):continue
            tokens=row_tokens(own[ids],opp[ids]);masks=bits_to_mask(legal[ids])
            if side==0:scores=policy_logits(models[learner],tokens,device)
            else:
                scores=np.empty((len(ids),64),dtype=np.float32)
                for index in np.unique(opponents[ids]):
                    subset=np.flatnonzero(opponents[ids]==index)
                    scores[subset]=policy_logits(models[int(index)],tokens[subset],device)
            actions=np.argmax(np.where(masks,scores+rng.gumbel(size=scores.shape),-np.inf),axis=-1)
            selected=U64(1)<<actions.astype(np.uint64)
            if np.any((selected&legal[ids])!=selected):raise AssertionError('Illegal action')
            moves[ids]=selected
            if collect and side==0:records.append((tokens.copy(),masks.copy(),actions.copy(),ids.copy()))
            if record and 0 in ids:
                k=int(np.flatnonzero(ids==0)[0]);trace.append({'color':color,'agent':side,'action':int(actions[k])})
        ids=np.flatnonzero(playable)
        own[ids],opp[ids]=apply_moves(own[ids],opp[ids],moves[ids])
        if np.any(black&white):raise AssertionError('Overlapping stones')
    else:raise AssertionError('Turn limit exceeded')
    margins=np.where(black_agents==0,stone_counts(black)-stone_counts(white),stone_counts(white)-stone_counts(black))
    return {'records':records,'margins':margins,'black_agents':black_agents,'opponents':opponents,
            'games':games,'wins':int((margins>0).sum()),'draws':int((margins==0).sum()),
            'losses':int((margins<0).sum()),'mean_reward':float(margins.mean()),'trace':trace,
            'black_stones':int(int(black[0]).bit_count()),'white_stones':int(int(white[0]).bit_count())}


def evaluation_aggregate(individuals):
    aggregate={}
    for name in ('random','position','initial'):
        aggregate[name]={k:sum(row[name][k] for row in individuals) for k in ('wins','draws','losses','games')}
        aggregate[name]['win_rate']=aggregate[name]['wins']/aggregate[name]['games']
    return aggregate


def merge_evaluation(previous,fresh):
    """Only the learner changes. Carry unchanged individuals' exact last values.

    evaluated_step distinguishes an actual new measurement from a carried value.
    """
    rows={row['id']:copy.deepcopy(row) for row in previous['individuals']}
    for row in fresh['individuals']:rows[row['id']]=row
    result={**fresh,'individuals':[rows[i] for i in sorted(rows)]}
    result['aggregate']=evaluation_aggregate(result['individuals'])
    return result


def diagnostic(models,initial,config,games,seed,step,heartbeat=None,indices=None):
    reference=copy.deepcopy(models[0])
    result=[]
    selected=list(range(len(models))) if indices is None else list(indices)
    for i in selected:
        model=models[i]
        scores={}
        reference.load_state_dict(initial[i])
        for j,(name,opponent) in enumerate((('random','random'),('position','position'),('initial',reference))):
            if heartbeat:heartbeat()
            match=play_batch([model,opponent],games,np.random.default_rng(seed+i*1009+j*101),
                             next(model.parameters()).device,reward_mode='margin')
            margins=match['rewards'][0]
            scores[name]={'wins':int((margins>0).sum()),'draws':int((margins==0).sum()),
                          'losses':int((margins<0).sum()),'games':games,'win_rate':float((margins>0).mean()),
                          'mean_margin_diagnostic_only':float(margins.mean())}
        result.append({'id':i,'evaluated_step':step,**scores})
    return {'step':step,'individuals':result,'aggregate':evaluation_aggregate(result),'measured_ids':selected}


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name,default in (('models',20),('games',128),('record-games',1000),('record-every',10),
            ('patience',200),('eval-games',128),('eval-every',25),('save-every',10),
            ('width',96),('layers',4),('heads',4),('grad-chunk',16),('seed',20261023),('updates',0)):
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--initial-percent',type=float,default=.1)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--reward',choices=['margin','signed_win'],default='margin')
    p.add_argument('--algorithm',choices=['reinforce','ppo'],default='reinforce')
    p.add_argument('--selection',choices=['record','win_rate'],default='record')
    p.add_argument('--games-per-opponent',type=int,default=128)
    p.add_argument('--win-threshold',type=float,default=.55)
    p.add_argument('--min-free-gib',type=float,default=.125)
    p.add_argument('--ppo-epochs',type=int,default=4)
    p.add_argument('--ppo-minibatch',type=int,default=256)
    p.add_argument('--ppo-clip',type=float,default=.2)
    p.add_argument('--gae-lambda',type=float,default=.95)
    p.add_argument('--value-coef',type=float,default=.5)
    p.add_argument('--target-kl',type=float,default=.02)
    p.add_argument('--initial-checkpoint',default='')
    p.add_argument('--compact-checkpoints',action='store_true')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--out',default='');p.add_argument('--resume',default='');p.add_argument('--publish',action='store_true')
    args=p.parse_args(argv)
    if (args.models<2 or args.updates<0 or any(getattr(args,k)<1 for k in
         ('record_every','patience','eval_every','save_every','grad_chunk'))
        or any(getattr(args,k)<2 or getattr(args,k)%2 for k in ('games','record_games','eval_games'))
        or args.lr<=0 or not .1<=args.initial_percent<=100
        or abs(args.initial_percent*10-round(args.initial_percent*10))>1e-9
        or args.ppo_epochs<1 or args.ppo_minibatch<1 or not 0<args.ppo_clip<1
        or not 0<=args.gae_lambda<=1 or args.value_coef<0 or args.target_kl<=0
        or args.games_per_opponent<2 or args.games_per_opponent%2 or not 0<args.win_threshold<1
        or not np.isfinite(args.min_free_gib) or args.min_free_gib<0):p.error('Invalid configuration')
    if args.selection=='win_rate':
        if args.initial_percent!=100:p.error('Win-rate selection requires all actor weights to be trainable from the start')
        args.games=(args.models-1)*args.games_per_opponent
    return args


def experiment(args):
    """Yield at completed-step boundaries so paired methods share one runtime."""
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    out=Path(args.out or 'runs/individual-pool-'+datetime.now().strftime('%Y%m%d-%H%M%S')).resolve();out.mkdir(parents=True,exist_ok=True)
    args.out=str(out)
    win_mode=args.selection=='win_rate'
    protocol=WIN_RATE_PROTOCOL if win_mode else PROTOCOL
    if (out/'config.json').exists() and not args.resume:raise ValueError('Choose a new output directory or resume')
    config=ModelConfig(args.width,args.layers,args.heads)
    template=OthelloGPT(config).to(device);description=template.description()
    description['all_parameters_trainable']=win_mode or args.initial_percent==100
    mask=(None if win_mode else ProgressiveUnfreezer(template,args.seed+500,initial_percent=args.initial_percent,granularity=10,min_hold_updates=0))
    del template
    initial_path=Path(args.initial_checkpoint).resolve() if args.initial_checkpoint else out/'initial.pt'
    initial=None
    if args.resume:
        initial=torch.load(initial_path,map_location='cpu',weights_only=False)['models']
        saved=(compact_checkpoint.load(args.resume,initial) if args.compact_checkpoints else
               torch.load(args.resume,map_location='cpu',weights_only=False))
    else:saved=None
    evaluation_changed=bool(saved and saved['args']['eval_games']!=args.eval_games)
    if saved:
        if saved['protocol']!=protocol:raise ValueError('Checkpoint protocol differs')
        defaults=vars(parse_args([]))
        for k,v in vars(args).items():
            if k not in ('updates','resume','device','publish','eval_games') and saved['args'].get(k,defaults.get(k))!=v:
                # Preserve the CLI's earlier validation behavior for callers.
                raise SystemExit(f'Resume must preserve {k}')
        if mask is not None:mask.load_state_dict(saved['mask'])
    models=[OthelloGPT(config).to(device) for _ in range(args.models)]
    critics=[value_head(args.width,device) for _ in models] if args.algorithm=='ppo' else []
    optimizers=[torch.optim.Adam(list(m.parameters())+(list(critics[i].parameters()) if critics else []),lr=args.lr)
                for i,m in enumerate(models)]
    rngs={name:np.random.default_rng(args.seed+offset) for name,offset in
          (('training',0),('record',700000),('selection',820000))}
    controller=(WinRateController(args.models,args.win_threshold,args.games_per_opponent,rngs['selection']) if win_mode
                else IndividualController(args.models,args.patience,rngs['selection']))
    ppo_rng=np.random.default_rng(args.seed+970000)
    step=training_games=record_games=0;elapsed_prior=0.
    baselines=[[0.,0.] for _ in models];learner_steps=[0]*args.models
    opponent_games=[0]*args.models;latest_training=None;evaluation=None
    if saved:
        for m,w in zip(models,saved['models']):m.load_state_dict(w)
        for critic,w in zip(critics,saved.get('critics',[])):critic.load_state_dict(w)
        for optimizer,state in zip(optimizers,saved['optimizers']):optimizer.load_state_dict(state)
        controller.load_state_dict(saved['controller'])
        for name,rng in rngs.items():rng.bit_generator.state=saved['rngs'][name]
        if 'ppo_rng' in saved:ppo_rng.bit_generator.state=saved['ppo_rng']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device=='cuda' and saved['cuda_rng']:torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        (step,training_games,record_games,elapsed_prior,baselines,learner_steps,opponent_games,
         latest_training,evaluation)=(saved[k] for k in ('step','training_games','record_games','elapsed',
         'baselines','learner_steps','opponent_games','latest_training','evaluation'))
        for name in ('metrics.jsonl','evaluations.jsonl','records.jsonl','record_evaluations.jsonl','switches.jsonl'):
            path=out/name
            if path.exists():
                rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x]
                path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows if r['step']<=step),encoding='utf-8')
        del saved
    else:
        if initial_path.exists():
            initial=torch.load(initial_path,map_location='cpu',weights_only=False)['models']
            if len(initial)!=args.models:raise ValueError('Initial pool size differs')
            for m,w in zip(models,initial):m.load_state_dict(w)
        else:
            initial=[{n:v.detach().cpu().clone() for n,v in m.state_dict().items()} for m in models]
            save_checkpoint(initial_path,{'protocol':protocol,'models':initial})
    digest=hashlib.sha256()
    for model in initial:
        for name,value in model.items():digest.update(name.encode());digest.update(value.numpy().tobytes())
    initial_hash=digest.hexdigest()
    if args.initial_checkpoint:atomic_json(out/'initial-reference.json',{'path':str(initial_path),'policy_sha256':initial_hash})
    metadata=vars(args)|{'protocol':protocol,'mode':'individual_pool','device_used':device,
        'reward':{'mode':args.reward,'formula':'own_stones - opponent_stones' if args.reward=='margin' else 'win +1; loss -1; draw 0'},'model':description,
        'selection':{'score':'(wins + 0.5 * draws) / games','record_every':args.record_every,
          'patience':args.patience,'rollback':False,'initial':'uniform among all models',
          'reselection':'uniform among all models except current learner; keep all weights and Adam states'},
        'matchmaking':'uniform random other model per game; learner black/white equally often',
        'sharing_enabled':False,'alien_learning_enabled':False,'entropy_reward_enabled':False,
        'p_values_enabled':False,'streaks_enabled':False,'updates_limit':args.updates or None,
        'algorithm':args.algorithm,'initial_policy_sha256':initial_hash,
        'ppo':({'epochs':args.ppo_epochs,'minibatch':args.ppo_minibatch,'clip':args.ppo_clip,
                'gamma':1.,'gae_lambda':args.gae_lambda,'value_coefficient':args.value_coef,'target_kl':args.target_kl,
                'critic':'linear state-value head sharing actor features; zero initialized; head fully trainable',
                'critic_parameters':args.width+1,'entropy_coefficient':0.} if critics else None),
        'checkpoint_format':compact_checkpoint.FORMAT if args.compact_checkpoints else 'full',
        'unfreezing':'initial 0.1%; add 0.1 percentage points to all models on every reselection; cap at 100%'}
    if win_mode:
        metadata.update(selection_mode='win_rate_threshold',
            fixed_evaluation_schedule='learner every step; all individuals every eval_every steps',
            selection={'score':'wins / games; draws are not wins','threshold':args.win_threshold,'strict':True,
                'check_every_round':True,'check_before_update':True,'timeout':None,'rollback':False,
                'initial':'uniform among all models','reselection':'uniform among all models except current learner; keep weights and Adam'},
            matchmaking=f'each other model exactly {args.games_per_opponent} games per round; half black, half white',
            unfreezing='none; all actor parameters trainable whenever this individual is selected',
            checkpoint_format=compact_checkpoint.FULL_FORMAT if args.compact_checkpoints else 'full')
    atomic_json(out/'config.json',metadata)
    started=time.monotonic();state={'status':'running','mode':'individual_pool','run_dir':str(out),
        'models':{'transformer':description},'model_count':args.models,'reward':metadata['reward'],
        'algorithm':args.algorithm,'ppo':metadata['ppo'],'initial_policy_sha256':initial_hash,
        'device':torch.cuda.get_device_name(0) if device=='cuda' else device,
        'updates':args.updates or None,'batch_games':args.games,'record_every':args.record_every,
        'eval_games_per_individual':args.eval_games,
        'record_evaluation_games':args.record_games,'sharing_enabled':False,'alien_learning_enabled':False}
    if win_mode:state.update(selection_mode='win_rate_threshold',games_per_opponent=args.games_per_opponent,
                             win_threshold=args.win_threshold)
    def capacity_summary():
        if mask is not None:return mask.summary()
        return {'percent':100.,'active_parameters':description['parameters'],'total_parameters':description['parameters'],
                'frozen_parameters':0,'events':[]}
    def append(name,row):
        with (out/name).open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    def publish(phase):
        state.update(phase=phase,step=step,elapsed_seconds=round(elapsed_prior+time.monotonic()-started,2),
            heartbeat_unix=time.time(),trainer_pid=os.getpid(),controller=controller.summary(),
            unfreezing=capacity_summary(),training_games=training_games,record_games=record_games,
            learner_steps=learner_steps,opponent_games=opponent_games,latest_training=latest_training,evaluation=evaluation)
        atomic_json(out/'progress.json',state)
    def checkpoint():
        prior_status=state['status']
        data={'protocol':protocol,'args':vars(args),'models':[m.state_dict() for m in models],
          'initial_policy_sha256':initial_hash,
          'optimizers':[o.state_dict() for o in optimizers],'mask':mask.state_dict() if mask is not None else None,
          'trained_model_ids':[i for i,n in enumerate(learner_steps) if n>0],'controller':controller.state_dict(),
          'critics':[c.state_dict() for c in critics],'ppo_rng':ppo_rng.bit_generator.state,
          'step':step,'training_games':training_games,'record_games':record_games,
          'baselines':baselines,'learner_steps':learner_steps,'opponent_games':opponent_games,
          'latest_training':latest_training,'evaluation':evaluation,'elapsed':elapsed_prior+time.monotonic()-started,
          'rngs':{n:r.bit_generator.state for n,r in rngs.items()},'torch_rng':torch.get_rng_state(),
          'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else []}
        if not args.compact_checkpoints:save_checkpoint(out/'latest.pt',data);return
        while True:
            try:
                compact_checkpoint.save(out/'latest.pt.gz',data,mask.masks if mask is not None else None,
                    reserve_bytes=min(int(args.min_free_gib*1024**3),128*1024**2)
                                  if (out/'STOP').exists() else int(args.min_free_gib*1024**3))
                state.update(status=prior_status,storage=None);return
            except OSError as error:
                if error.errno!=errno.ENOSPC:raise
                state.update(status='waiting_storage',storage=str(error))
                print(f'Waiting for checkpoint storage at step {step}: {error}',file=sys.stderr,flush=True)
                try:publish('waiting_storage')
                except OSError:pass
                # Keep the pending complete step and previous checkpoint. The
                # shared runner pauses too; it never silently drops a save.
                time.sleep(5)
    def assess():
        nonlocal record_games
        publish('evaluation_record')
        match=play_individual(models,controller.learner,args.record_games,rngs['record'],heartbeat=lambda:publish('evaluation_record'))
        record_games+=args.record_games
        improved=controller.observe(step,match['margins'])
        row={'step':step,'learner':controller.learner,'phase':controller.phase,'improved':improved,
             'evaluation':copy.deepcopy(controller.current),'opponent_counts':np.bincount(match['opponents'],minlength=args.models).tolist()}
        append('record_evaluations.jsonl',row)
        if improved:append('records.jsonl',row)
        return improved
    def diagnose(indices=None):
        nonlocal evaluation
        publish('evaluation_fixed')
        fresh=diagnostic(models,initial,config,args.eval_games,args.seed+900000,step,
                         lambda:publish('evaluation_fixed'),indices=indices)
        fresh['schedule']='learner_every_step' if win_mode else 'periodic_all'
        evaluation=merge_evaluation(evaluation,fresh) if indices is not None else fresh
        append('evaluations.jsonl',evaluation)
    def wait_for_capacity():
        reserve=int(args.min_free_gib*1024**3)
        while shutil.disk_usage(out).free<reserve:
            state.update(status='waiting_storage',storage='Free disk space is below the configured reserve')
            publish('waiting_storage')
            if (out/'STOP').exists():return False
            time.sleep(5)
        state.update(status='running',storage=None)
        return True
    def set_learner_scope():
        if not win_mode:return
        for i,model in enumerate(models):
            model.requires_grad_(i==controller.learner)
            if i!=controller.learner:
                for parameter in model.parameters():parameter.grad=None
        for i,critic in enumerate(critics):
            critic.requires_grad_(i==controller.learner)
            if i!=controller.learner:
                for parameter in critic.parameters():parameter.grad=None
    try:
        publish('initializing')
        if args.publish:Path(__file__).with_name('latest_run.txt').write_text(str(out),encoding='utf-8')
        set_learner_scope()
        if (not args.resume or evaluation_changed or
            (win_mode and (evaluation is None or evaluation.get('schedule')!='learner_every_step'))):diagnose()
        if win_mode:checkpoint();publish('ready')
        yield {'step':step,'initial_policy_sha256':initial_hash}
        while args.updates==0 or step<args.updates:
            if (out/'STOP').exists():state['status']='stopped';break
            if not wait_for_capacity():state['status']='stopped';break
            if not win_mode and controller.best is None:
                assess();checkpoint()
                if (out/'STOP').exists():state['status']='stopped';break
            learner=controller.learner
            publish('competition_play')
            opponents=balanced_opponents(args.models,learner,args.games_per_opponent,rngs['training']) if win_mode else None
            match=play_individual(models,learner,args.games,rngs['training'],collect=True,record=True,
                                  opponents=opponents,heartbeat=lambda:publish('competition_play'))
            passed=False;improved=False
            if win_mode:
                prior_records=controller.records
                passed=controller.observe(step+1,match['margins'],match['opponents'])
                improved=controller.records>prior_records
                row={'step':step+1,'learner':learner,'phase':controller.phase,'passed':passed,'improved':improved,
                     'evaluation':copy.deepcopy(controller.current),'from_training_rollout':True}
                append('record_evaluations.jsonl',row)
                if improved:append('records.jsonl',row)
            publish('competition_update')
            rewards=training_rewards(match['margins'],args.reward)
            if passed:
                update={'algorithm':args.algorithm,'optimizer_steps':0,'skipped':'threshold_met',
                        'mean_reward':float(rewards.mean()),'parameter_delta_l2':0.,'decisions':0}
            elif args.algorithm=='ppo':
                update=update_ppo(models[learner],critics[learner],optimizers[learner],match['records'],rewards,
                    ppo_rng,mask,epochs=args.ppo_epochs,minibatch=args.ppo_minibatch,chunk_size=args.grad_chunk,
                    clip=args.ppo_clip,lam=args.gae_lambda,value_coef=args.value_coef,target_kl=args.target_kl,
                    heartbeat=lambda:publish('competition_update'))
            else:
                update=update_policy(models[learner],optimizers[learner],match['records'],rewards,
                    match['black_agents'],0,baselines[learner],args.grad_chunk,unfreezer=mask)
            step+=1;training_games+=args.games;learner_steps[learner]+=int(not passed)
            counts=np.bincount(match['opponents'],minlength=args.models)
            opponent_games[:]=[a+int(b) for a,b in zip(opponent_games,counts)]
            if win_mode:
                event=controller.reselect(step,rngs['selection']) if passed else None
                if event:set_learner_scope()
            else:
                improved=assess() if step%args.record_every==0 or controller.expired(step) else False
                event=controller.reselect(step,mask,rngs['selection']) if controller.expired(step) else None
            if event:append('switches.jsonl',event)
            latest_training={'step':step,'learner':learner,'games':args.games,'wins':match['wins'],
                'draws':match['draws'],'losses':match['losses'],'mean_reward':float(rewards.mean()),
                'mean_margin':match['mean_reward'],'reward_mode':args.reward,
                'algorithm':args.algorithm,'rollout_sha256':rollout_digest(match['records']),
                'opponents':match['opponents'].tolist(),'opponent_counts':counts.tolist(),'update':update,
                'record_improved':improved,'switch':event,'active_percent':capacity_summary()['percent'],
                'threshold_passed':passed if win_mode else None}
            if win_mode:latest_training['per_opponent']=row['evaluation']['per_opponent']
            append('metrics.jsonl',latest_training)
            atomic_json(out/'last_game_transformer.json',{'step':step,'learner_id':learner,
                'black_id':learner,'white_id':int(match['opponents'][0]),'moves':match['trace'],
                'black_stones':match['black_stones'],'white_stones':match['white_stones']})
            if win_mode:
                diagnose(indices=None if step%args.eval_every==0 else [learner])
            if improved or event or step%args.save_every==0:publish('saving');checkpoint()
            publish('batch_complete')
            if not win_mode and step%args.eval_every==0 and not (out/'STOP').exists():diagnose()
            yield {'step':step,'initial_policy_sha256':initial_hash}
        else:state['status']='completed'
        publish('saving');checkpoint();publish(state['status'])
    except Exception:
        state.update(status='failed',error=traceback.format_exc())
        try:publish('failed')
        except OSError:pass
        raise


def main():
    for _ in experiment(parse_args()):pass


if __name__=='__main__':main()
