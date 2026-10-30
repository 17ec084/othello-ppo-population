"""Ten vs ten: random matches, shared policies, optional novelty, record rollback."""
import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from curriculum import ProgressiveUnfreezer
from model import ModelConfig,OthelloGPT
from population import (PopulationGroup,population_match,competition_update,population_diversity,
                        alien_update,random_game_pairings)
from population_train import common_positions,evaluate_population
from record_control import RecordController
from runtime_io import atomic_json,replace_with_retry
from sharing import evaluate_teachers,sharing_plan,sharing_update

PROTOCOL='population-record-per-game-v2'


def record_evaluation(groups,games_per_pair,rng,heartbeat=None):
    """All A/B pairs, balanced colors; fresh randomized cyclic match rounds."""
    size=groups[0].policy.size
    base=rng.permutation(size)
    margins=[]
    pairings=[]
    for offset in rng.permutation(size):
        if heartbeat:heartbeat()
        pairing=base[(np.arange(size)+offset)%size]
        match=population_match([g.policy for g in groups],size,games_per_pair,rng,
                               opponent_indices=pairing)
        margins.append(match['margins'])
        pairings.append(pairing.tolist())
    return np.concatenate(margins,axis=1),pairings


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name,default in (('population',10),('games-per-pair',128),('record-every',10),
            ('record-games-per-pair',10),('patience',200),('eval-games-per-model',16),
            ('eval-every',25),('save-every',10),('sharing-rounds',4),('sharing-positions',64),
            ('teacher-eval-every',25),('teacher-games-per-opponent',16),('teachers-per-student',2),
            ('diversity-positions',128),('width',96),('layers',4),('heads',4),('chunk',16),
            ('seed',20261023),('updates',0)):
        p.add_argument('--'+name,type=int,default=default)
    for name,default in (('initial-percent',.1),('lr',1e-4),('alien-lr',1e-4),
                         ('sharing-lr',1e-5),('teacher-fraction',.2)):
        p.add_argument('--'+name,type=float,default=default)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--out',default='')
    p.add_argument('--resume',default='')
    p.add_argument('--publish',action='store_true')
    p.add_argument('--alien-learning',action='store_true',help='Enable novelty updates (disabled by default)')
    args=p.parse_args()
    positive=('record_every','record_games_per_pair','patience','eval_games_per_model','eval_every',
              'save_every','sharing_rounds','sharing_positions','teacher_eval_every',
              'teacher_games_per_opponent','teachers_per_student','diversity_positions','chunk')
    if (args.population<2 or args.updates<0 or any(getattr(args,k)<1 for k in positive)
            or any(n<2 or n%2 for n in (args.games_per_pair,args.record_games_per_pair,
                       args.eval_games_per_model,args.teacher_games_per_opponent))
            or not 0<args.teacher_fraction<=1 or min(args.lr,args.alien_lr,args.sharing_lr)<=0
            or not .1<=args.initial_percent<=100
            or abs(args.initial_percent*10-round(args.initial_percent*10))>1e-9):
        p.error('Invalid record-learning configuration')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    out=Path(args.out or 'runs/record-'+datetime.now().strftime('%Y%m%d-%H%M%S')).resolve()
    out.mkdir(parents=True,exist_ok=True)
    args.out=str(out)
    if (out/'config.json').exists() and not args.resume:p.error('Choose a new run or resume')
    config=ModelConfig(args.width,args.layers,args.heads)
    template=OthelloGPT(config).to(device)
    description=template.description()
    mask=ProgressiveUnfreezer(template,args.seed+500,initial_percent=args.initial_percent,
                             min_hold_updates=0,granularity=10)
    del template
    saved=torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else None
    if saved:
        if saved['protocol']!=PROTOCOL:p.error('Checkpoint protocol differs')
        for k,v in vars(args).items():
            if k not in ('updates','device','resume','publish') and saved['args'].get(k,True if k=='alien_learning' else None)!=v:
                p.error(f'Resume must preserve {k}')
        mask.load_state_dict(saved['mask'])
    groups=[PopulationGroup(config,args.population,mask,device,args.lr,args.alien_lr,
                            saved['groups'][i]['weights'] if saved else None,
                            sharing_lr=args.sharing_lr) for i in (0,1)]
    controller=RecordController(args.patience)
    rngs={name:np.random.default_rng(args.seed+offset) for name,offset in
          (('training',0),('record',700000),('sharing',800000),('teacher',810000))}
    step=training_games=record_games=teacher_games=0
    elapsed_prior=0.
    pending=best_state=None
    learner_steps=[0,0]
    alien_totals=[0,0]
    sharing_totals=[0,0]
    sharing_examples=[np.zeros(args.population,dtype=np.int64) for _ in (0,1)]
    teacher_reports=[None,None]
    last_sharing=[None,None]
    evaluation=None
    latest_training={}
    if saved:
        for g,gs in zip(groups,saved['groups']):g.load_state_dict(gs)
        controller.load_state_dict(saved['controller'])
        for name,rng in rngs.items():rng.bit_generator.state=saved['rngs'][name]
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device=='cuda' and saved['cuda_rng']:
            torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
        step,training_games,record_games,teacher_games,elapsed_prior=(saved[k] for k in
            ('step','training_games','record_games','teacher_games','elapsed'))
        (pending,best_state,learner_steps,alien_totals,sharing_totals,sharing_examples,
         teacher_reports,last_sharing,tokens,masks,targets,diversity,evaluation,latest_training)=(saved[k] for k in
            ('pending','best_state','learner_steps','alien_totals','sharing_totals','sharing_examples',
             'teacher_reports','last_sharing','tokens','masks','targets','diversity','evaluation','latest_training'))
        initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)['groups']
        # Completed-step logs and pending-step alien events have different horizons.
        for name in ('metrics.jsonl','evaluations.jsonl','switches.jsonl','records.jsonl',
                     'record_evaluations.jsonl','sharing_sessions.jsonl','alien_events.jsonl','alien_sessions.jsonl'):
            path=out/name
            if not path.exists():continue
            horizon=pending['next_step'] if pending and name=='alien_events.jsonl' else step
            rows=[json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l]
            path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows if r['step']<=horizon),encoding='utf-8')
        del saved
    else:
        initial=[g.policy.snapshot() for g in groups]
        torch.save({'protocol':PROTOCOL,'groups':initial},out/'initial.pt')
        tokens,masks=common_positions(args.seed+9000,args.diversity_positions)
        diversity=[population_diversity(g.policy,tokens,masks,args.chunk) for g in groups]
        targets=[d['value'] for d in diversity]
    metadata=vars(args)|{'protocol':PROTOCOL,'matchmaking':'per_game_permutation',
        'updates_limit':args.updates or None,'device_used':device,
        'models':{'transformer':description|{'all_parameters_trainable':False,'population_per_group':args.population}},
        'reward':{'mode':'margin','formula':'own_stones - opponent_stones'},
        'selection':{'mode':'record_score','score':'(wins + 0.5 * draws) / games; strict record updates',
            'evaluation':'all cross-group pairs, balanced colors, fresh random rounds; separate RNG',
            'games_per_evaluation':args.population**2*args.record_games_per_pair,
            'rollback':'entire learner group including three Adam states and reward baselines',
            'switch':f'after {args.patience} completed steps without a record; no win-rate threshold',
            'comparison':'opponent weights fixed throughout each phase; record reset on switch'},
        'sharing':{'method':'within-group policy distillation on pooled competition boards',
            'rounds_per_step':args.sharing_rounds,'positions_per_individual_per_round':args.sharing_positions,
            'teachers':'independent draw-half ranking vs random and position; higher-scored top peers',
            'scope':'active group only; separate Adam; opened weights only'},
        'diversity':{'metric':'JS across policies on fixed common positions, nats','targets':targets,
            'reward':'log pi_i(a|s) - log mean_j pi_j(a|s)' if args.alien_learning else None,
            'scope':'active group only; repair until its initial diversity is met' if args.alien_learning else 'measurement only; no diversity constraint or updates',
            'enabled':args.alien_learning,'maximum_rounds':None if args.alien_learning else 0},
        'alien_learning_enabled':args.alien_learning,
        'entropy_reward_enabled':False,'p_values_enabled':False,'streaks_enabled':False,
        'unfreezing':'initial 0.1%; add 0.1 percentage points only on B-to-A switch'}
    atomic_json(out/'config.json',metadata)
    started=time.monotonic()
    state={'status':'running','run_dir':str(out),'models':metadata['models'],'reward':metadata['reward'],
           'device':torch.cuda.get_device_name(0) if device=='cuda' else device,
           'population':args.population,'updates':args.updates or None,'selection_mode':'record_score',
           'matchmaking':metadata['matchmaking'],
           'record_every':args.record_every,'record_evaluation_games':metadata['selection']['games_per_evaluation'],
           'sharing_rounds_per_step':args.sharing_rounds,'sharing_positions':args.sharing_positions,
           'batch_games':args.population*args.games_per_pair,'diversity_scope':'active_group_only',
           'alien_learning_enabled':args.alien_learning}
    def append(name,row):
        with (out/name).open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    def publish(phase):
        state.update(phase=phase,step=step,competition_step=pending['next_step'] if pending else step,
            elapsed_seconds=round(elapsed_prior+time.monotonic()-started,2),trainer_pid=os.getpid(),heartbeat_unix=time.time(),
            training_games={'transformer':training_games},test_games=record_games,teacher_games=teacher_games,
            alternating=controller.summary(),plateau=controller.plateau_summary(step),
            unfreezing={'transformer':mask.summary()},evaluation=evaluation,latest_training=latest_training,
            experience={name:{'competition_steps':learner_steps[i],
                'training_games_per_individual':learner_steps[i]*args.games_per_pair,
                'sharing_boards_per_individual':sharing_examples[i].tolist()} for i,name in enumerate('AB')},
            sharing={name:{'updates':sharing_totals[i],'last':last_sharing[i],
                'teacher_evaluation':teacher_reports[i],
                'mean_boards_per_individual':float(sharing_examples[i].mean())} for i,name in enumerate('AB')},
            diversity={name:dict(diversity[i],target=targets[i],met=diversity[i]['value']+1e-10>=targets[i],
                total_alien_updates=alien_totals[i]) for i,name in enumerate('AB')},
            pending_stage=pending['stage'] if pending else None)
        atomic_json(out/'progress.json',state)
    def save_torch(name,data):
        temporary=out/(name+'.tmp')
        torch.save(data,temporary)
        replace_with_retry(temporary,out/name)
    def checkpoint():
        save_torch('latest.pt',{'protocol':PROTOCOL,'args':vars(args),'step':step,
            'groups':[g.state_dict() for g in groups],'controller':controller.state_dict(),'mask':mask.state_dict(),
            'pending':pending,'best_state':best_state,'learner_steps':learner_steps,'alien_totals':alien_totals,
            'sharing_totals':sharing_totals,'sharing_examples':sharing_examples,'teacher_reports':teacher_reports,
            'last_sharing':last_sharing,'tokens':tokens,'masks':masks,'targets':targets,'diversity':diversity,
            'evaluation':evaluation,'latest_training':latest_training,'training_games':training_games,
            'record_games':record_games,'teacher_games':teacher_games,
            'rngs':{name:rng.bit_generator.state for name,rng in rngs.items()},'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else [],
            'elapsed':elapsed_prior+time.monotonic()-started})
    def assess(at_step):
        nonlocal record_games,best_state
        publish('evaluation_record')
        margins,pairings=record_evaluation(groups,args.record_games_per_pair,rngs['record'],
                                          lambda:publish('evaluation_record'))
        record_games+=margins.size
        improved=controller.observe(at_step,margins)
        row={'step':at_step,'phase':controller.phase,'learner':'AB'[controller.learner],
             'improved':improved,'evaluation':copy.deepcopy(controller.current),'pairings':pairings}
        append('record_evaluations.jsonl',row)
        if improved:
            best_state={'phase':controller.phase,'learner':controller.learner,'step':at_step,
                'evaluation':copy.deepcopy(controller.best),
                'group':copy.deepcopy(groups[controller.learner].state_dict()),
                'diversity':copy.deepcopy(diversity[controller.learner])}
            save_torch('best.pt',best_state)
            append('records.jsonl',row)
        return improved
    def diagnostic():
        nonlocal evaluation
        publish('evaluation_fixed')
        evaluation=evaluate_population(groups,initial,mask,args.eval_games_per_model,args.seed+900000,
                                       step,lambda:publish('evaluation_fixed'))
        append('evaluations.jsonl',evaluation)
    try:
        publish('initializing')
        if args.publish:Path(__file__).with_name('latest_run.txt').write_text(str(out),encoding='utf-8')
        if not args.resume:diagnostic()
        while args.updates==0 or step<args.updates or pending:
            if (out/'STOP').exists():state['status']='stopped';break
            if controller.best is None:
                assess(step)
                checkpoint()
                if (out/'STOP').exists():state['status']='stopped';break
            if pending is None:
                learner=controller.learner
                pairing=random_game_pairings(args.population,args.games_per_pair,rngs['training'])
                publish('competition_play')
                match=population_match([g.policy for g in groups],args.population,args.games_per_pair,
                    rngs['training'],collect=learner,record=True,opponent_indices=pairing)
                publish('competition_update')
                update=competition_update(groups[learner],match,learner,args.chunk)
                learner_steps[learner]+=1;training_games+=match['games']
                pending={'next_step':step+1,'learner':learner,'stage':'sharing','sharing_round':0,
                    'records':match['records'],'alien_rounds':[0,0], 'sharing_results':[],
                    'row':{'step':step+1,'learner':'AB'[learner],'pairing':pairing.tolist(),
                        'pairing_mode':'per_game_permutation',
                        'games':match['games'],'a_wins':match['a_wins'],'b_wins':match['b_wins'],'draws':match['draws'],
                        'a_mean_reward':float(match['margins'].mean()),'b_mean_reward':-float(match['margins'].mean()),
                        'agents':{'AB'[learner]:update}},
                    'game':{'step':step+1,'family':'transformer','pair':0,'b_index':int(pairing[0,0]),
                        'moves':match['trace'],'black_stones':match['black_stones'],'white_stones':match['white_stones']}}
            learner=pending['learner']
            if pending['stage']=='sharing':
                report=teacher_reports[learner]
                if report is None or learner_steps[learner]-report['learner_steps']>=args.teacher_eval_every:
                    publish('sharing_evaluation')
                    report=evaluate_teachers(groups[learner],args.teacher_games_per_opponent,rngs['teacher'],
                                             lambda:publish('sharing_evaluation'))
                    report.update(step=pending['next_step'],learner_steps=learner_steps[learner])
                    teacher_reports[learner]=report;teacher_games+=report['games']
                while pending['sharing_round']<args.sharing_rounds and not (out/'STOP').exists():
                    publish('sharing_'+'AB'[learner])
                    plan=sharing_plan(groups[learner],pending['records'],report['scores'],rngs['sharing'],
                        args.sharing_positions,args.teacher_fraction,args.teachers_per_student,args.chunk)
                    shared=sharing_update(groups[learner],plan,args.chunk)
                    pending['sharing_round']+=1
                    shared.update(step=pending['next_step'],group='AB'[learner],round=pending['sharing_round'],
                                  teacher_evaluation_step=report['step'])
                    if shared['students']:
                        sharing_totals[learner]+=1
                        sharing_examples[learner]+=plan['enabled']*args.sharing_positions
                    pending['sharing_results'].append(shared);last_sharing[learner]=shared
                if (out/'STOP').exists():state['status']='stopped';break
                del pending['records'];pending['stage']='alien'
            if pending['stage']=='alien':
                if not args.alien_learning:publish('diversity_measure')
                diversity[learner]=population_diversity(groups[learner].policy,tokens,masks,args.chunk)
                while args.alien_learning and diversity[learner]['value']+1e-10<targets[learner] and not (out/'STOP').exists():
                    publish('alien_'+'AB'[learner])
                    update=alien_update(groups[learner],tokens,masks,args.chunk)
                    alien_totals[learner]+=1;pending['alien_rounds'][learner]+=1
                    diversity[learner]=population_diversity(groups[learner].policy,tokens,masks,args.chunk)
                    append('alien_events.jsonl',{'step':pending['next_step'],'group':'AB'[learner],
                        'type':'alien_learning','value':diversity[learner]['value'],'target':targets[learner],
                        'round':alien_totals[learner],'update':update})
                if (out/'STOP').exists():state['status']='stopped';break
                pending['stage']='finish'
            if pending['stage']=='finish':
                step=pending['next_step']
                improved=assess(step) if step%args.record_every==0 or controller.expired(step) else False
                event=None
                if controller.expired(step):
                    publish('record_rollback')
                    if best_state['phase']!=controller.phase or best_state['learner']!=learner:
                        raise RuntimeError('Best snapshot belongs to a different phase')
                    groups[learner].load_state_dict(copy.deepcopy(best_state['group']))
                    diversity[learner]=copy.deepcopy(best_state['diversity'])
                    teacher_reports[learner]=None
                    event=controller.switch(step,mask)
                    append('switches.jsonl',event)
                    if event['released']:
                        for g in groups:g.expand(mask)
                        teacher_reports[:]=[None,None]
                    best_state=None
                row=pending['row']
                row.update(active_percent=mask.summary()['percent'],switch=event,
                    record_improved=improved,record=copy.deepcopy(controller.summary()),
                    diversity={name:diversity[i] for i,name in enumerate('AB')},
                    sharing=pending['sharing_results'],alien_rounds=pending['alien_rounds'],
                    training_games=training_games,record_games=record_games)
                append('metrics.jsonl',row)
                if any(x['students'] for x in pending['sharing_results']):
                    append('sharing_sessions.jsonl',{'step':step,'type':'shared_learning','group':'AB'[learner],
                        'students':max(x['students'] for x in pending['sharing_results']),
                        'rounds':len(pending['sharing_results']),'positions_per_student':args.sharing_positions})
                if any(pending['alien_rounds']):
                    append('alien_sessions.jsonl',{'step':step,'type':'alien_learning',
                        'rounds':{'AB'[i]:v for i,v in enumerate(pending['alien_rounds']) if v}})
                latest_training={'transformer':row}
                atomic_json(out/'last_game_transformer.json',pending['game'])
                pending=None
                if event or improved or step%args.save_every==0:publish('saving');checkpoint()
                publish('batch_complete')
                print(json.dumps({'step':step,'learner':row['learner'],'next_learner':'AB'[controller.learner],
                    'best':controller.best,'rollback':event is not None,'sharing_rounds':args.sharing_rounds,
                    'alien_rounds':row['alien_rounds']}),flush=True)
                if step%args.eval_every==0 and not (out/'STOP').exists():diagnostic()
        else:state['status']='completed'
        publish('saving');checkpoint();publish(state['status'])
    except Exception:
        state.update(status='failed',error=traceback.format_exc());publish('failed');raise


if __name__=='__main__':main()
