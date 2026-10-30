"""100 A vs 100 B: competition, within-group distillation, then novelty."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from curriculum import ProgressiveUnfreezer
from model import ModelConfig, OthelloGPT
from population import (PopulationGroup, PopulationPolicy, population_match,
                        competition_update, population_diversity, alien_update)
from population_control import PopulationController
from runtime_io import atomic_json, replace_with_retry
from sharing import evaluate_teachers, sharing_plan, sharing_update
from train import play_batch

PROTOCOL = 'population-sharing-distillation-v3'


def common_positions(seed, count):
    match = play_batch(['random', 'random'], max(8, 2*((count+119)//120)),
                       np.random.default_rng(seed), 'cpu', collect=True)
    candidates, seen = [], set()
    for side in match['records']:
        for tokens, masks, _, _ in side:
            for t, m in zip(tokens, masks):
                key = t.tobytes()
                if m.sum() > 1 and key not in seen:
                    seen.add(key)
                    candidates.append((t, m))
    if len(candidates) < count:
        raise ValueError('Not enough distinct common positions')
    ids = np.random.default_rng(seed+1).permutation(len(candidates))[:count]
    return np.stack([candidates[i][0] for i in ids]), np.stack([candidates[i][1] for i in ids])


def match_summary(match):
    margins = match['margins']
    wins = int((margins>0).sum())
    n = margins.size
    # Group diagnostics, not the uncertainty interval of the group-mean test.
    return {'wins': wins, 'draws': int((margins==0).sum()), 'losses': int((margins<0).sum()),
            'games': n, 'win_rate': wins/n,
            'mean_margin_diagnostic_only': float(margins.mean()),
            'individual_mean_margins': margins.mean(1).tolist()}


def evaluate_population(groups, initial_weights, mask, games, seed, step, heartbeat=None):
    result = {'step': step, 'families': {'transformer': {}}, 'cross': {}}
    for agent, group in enumerate(groups):
        scores = {}
        for index, opponent in enumerate(('random', 'position', None)):
            if heartbeat:
                heartbeat()
            if opponent is None:
                opponent = PopulationPolicy(group.policy.config, group.policy.size, mask,
                                            group.policy.device, initial_weights[1-agent])
            match = population_match([group.policy, opponent], group.policy.size, games,
                                     np.random.default_rng(seed+100*agent+index))
            scores[('random', 'position', 'initial')[index]] = match_summary(match)
            if not isinstance(opponent, str):
                del opponent
        result['families']['transformer']['AB'[agent]] = scores
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--population', type=int, default=100)
    parser.add_argument('--games-per-pair', type=int, default=2)
    parser.add_argument('--test-games-per-pair', type=int, default=2)
    parser.add_argument('--eval-games-per-model', type=int, default=2)
    parser.add_argument('--eval-every', type=int, default=25)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--diversity-positions', type=int, default=128)
    parser.add_argument('--diversity-multiplier', type=float, default=1.)
    parser.add_argument('--updates', type=int, default=0)
    parser.add_argument('--initial-percent', type=float, default=.1)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=.05)
    parser.add_argument('--width', type=int, default=96)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alien-lr', type=float, default=1e-4)
    parser.add_argument('--sharing-lr', type=float, default=1e-5)
    parser.add_argument('--sharing-positions', type=int, default=32)
    parser.add_argument('--teacher-eval-every', type=int, default=25)
    parser.add_argument('--teacher-games-per-opponent', type=int, default=16)
    parser.add_argument('--teacher-fraction', type=float, default=.2)
    parser.add_argument('--teachers-per-student', type=int, default=2)
    parser.add_argument('--chunk', type=int, default=16)
    parser.add_argument('--seed', type=int, default=20261023)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
    parser.add_argument('--out', default='')
    parser.add_argument('--resume', default='')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    if (args.population<2 or args.updates<0 or args.patience<1 or not 0<args.alpha<1
            or min(args.eval_every,args.save_every,args.diversity_positions,args.chunk,
                   args.sharing_positions,args.teacher_eval_every,args.teachers_per_student)<1
            or min(args.lr,args.alien_lr,args.sharing_lr,args.diversity_multiplier)<=0
            or not 0<args.teacher_fraction<=1
            or not .1<=args.initial_percent<=100 or abs(args.initial_percent*10-round(args.initial_percent*10))>1e-9
            or any(n<2 or n%2 for n in (args.games_per_pair,args.test_games_per_pair,
                                       args.eval_games_per_model,args.teacher_games_per_opponent))):
        parser.error('Invalid population configuration')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    out = Path(args.out or ('runs/population-'+datetime.now().strftime('%Y%m%d-%H%M%S'))).resolve()
    out.mkdir(parents=True, exist_ok=True)
    args.out = str(out)
    if (out/'config.json').exists() and not args.resume:
        parser.error('Choose a new output directory or explicitly resume')
    config = ModelConfig(args.width,args.layers,args.heads)
    mask_model = OthelloGPT(config).to(device)
    description = mask_model.description()
    mask = ProgressiveUnfreezer(mask_model,args.seed+500,initial_percent=args.initial_percent,
                               min_hold_updates=0,granularity=10)
    del mask_model
    saved = None
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved['protocol'] != PROTOCOL:
            parser.error('Checkpoint protocol differs')
        for key, value in vars(args).items():
            if key not in ('updates','device','resume','publish') and saved['args'][key] != value:
                parser.error(f'Resume must preserve {key}={saved["args"][key]}')
        mask.load_state_dict(saved['mask'])
    groups = [PopulationGroup(config,args.population,mask,device,args.lr,args.alien_lr,
                              saved['groups'][i]['weights'] if saved else None,
                              sharing_lr=args.sharing_lr) for i in (0,1)]
    controller = PopulationController(args.alpha,args.patience)
    train_rng = np.random.default_rng(args.seed)
    test_rng = np.random.default_rng(args.seed+700000)
    sharing_rng = np.random.default_rng(args.seed+800000)
    teacher_rng = np.random.default_rng(args.seed+810000)
    step = training_games = test_games = 0
    elapsed_prior = 0.
    pending = None
    alien_totals = [0,0]
    learner_steps = [0,0]
    teacher_reports = [None,None]
    teacher_games = 0
    sharing_totals = [0,0]
    sharing_examples = [np.zeros(args.population,dtype=np.int64) for _ in (0,1)]
    last_sharing = [None,None]
    if saved:
        for group, state in zip(groups, saved['groups']):
            group.load_state_dict(state)
        controller.load_state_dict(saved['controller'])
        train_rng.bit_generator.state = saved['train_rng']
        test_rng.bit_generator.state = saved['test_rng']
        sharing_rng.bit_generator.state = saved['sharing_rng']
        teacher_rng.bit_generator.state = saved['teacher_rng']
        teacher_reports,teacher_games,sharing_totals,sharing_examples,last_sharing = (saved[k] for k in
            ('teacher_reports','teacher_games','sharing_totals','sharing_examples','last_sharing'))
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device=='cuda' and saved['cuda_rng']:
            torch.cuda.set_rng_state_all([value.cpu() for value in saved['cuda_rng']])
        step,training_games,test_games,elapsed_prior = (saved[k] for k in ('step','training_games','test_games','elapsed'))
        learner_steps = saved['learner_steps']
        tokens,masks,targets,diversity,alien_totals,pending = (saved[k] for k in
            ('tokens','position_masks','targets','diversity','alien_totals','pending'))
        initial = torch.load(out/'initial.pt',map_location='cpu',weights_only=False)['groups']
        # Events may belong to a safely saved, still-pending competition step.
        horizon = pending['next_step'] if pending else step
        for name in ('metrics.jsonl','evaluations.jsonl','switches.jsonl','copies.jsonl',
                     'alien_events.jsonl','alien_sessions.jsonl','sharing_sessions.jsonl'):
            path = out/name
            if path.exists():
                rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
                path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows
                                      if row['step'] <= (step if name=='metrics.jsonl' else horizon)),encoding='utf-8')
        del saved
    else:
        initial = [group.policy.snapshot() for group in groups]
        torch.save({'protocol':PROTOCOL,'groups':initial},out/'initial.pt')
        tokens,masks = common_positions(args.seed+9000,args.diversity_positions)
        diversity = [population_diversity(group.policy,tokens,masks,args.chunk) for group in groups]
        targets = [value['value']*args.diversity_multiplier for value in diversity]
    metadata = vars(args) | {'protocol':PROTOCOL,'updates_limit':args.updates or None,
        'device_used':device,'reward':{'mode':'margin','formula':'own_stones - opponent_stones'},
        'models':{'transformer':description | {'all_parameters_trainable':False,'population_per_group':args.population}},
        'competition':'only active group; fresh uniform permutation of B each step; own raw margin per individual',
        'sharing':{'type':'within-group policy distillation; active group only; one update after competition',
            'boards':'pooled valid competition boards; independently sample boards for each student',
            'teachers':f'up to {args.teachers_per_student} distinct higher-scored peers from observed top {args.teacher_fraction*100:g}%; average their policies',
            'ranking':'separate color-balanced games vs random and position AIs; draw-half score; not a significance claim',
            'refresh':f'first use, every {args.teacher_eval_every} learner updates, after copy or mask expansion',
            'loss':'KL(detached teacher mixture || student); separate Adam, opened weights only',
            'weakness_control':f'sharing learning rate {args.sharing_lr} vs competition {args.lr}; initial diversity restored afterward',
            'isolation':'independent RNGs; teacher games and shared boards never enter the p-value test',
            'sources':['https://arxiv.org/abs/1511.06295','https://arxiv.org/abs/1803.03835']},
        'test':{'type':'one-sided Hoeffding exponential mixture on independent held-out random-opponent color pairs',
                'statistic':'group mean win signs (+1 win, 0 draw, -1 loss), one batch update per competition step',
                'null':'conditional group expected win sign <= 0 (draw-half score <= 0.5) at every evaluation batch',
                'assumptions':'both populations fixed; independent uniform B opponent per A (with replacement); independent pair RNG; averaged colors within each pair',
                'reset':'switch or index-matched group copy; alpha per direction per epoch, not experiment-wide',
                'source':'https://arxiv.org/abs/2210.01948'},
        'diversity':{'metric':'JS of policies on fixed common legal positions, nats','targets':targets,
                'positions':len(tokens),'reward':'log pi_i(a|s) - log mean_j pi_j(a|s)',
                'optimizer':'separate Adam for intrinsic objective; opened weights only',
                'groups':'repair both groups until each initial diversity target is met',
                'maximum_rounds':None},
        'unfreezing':'start 0.1%; add 0.1 percentage points only on B-to-A switch',
        'mutations_enabled':False}
    atomic_json(out/'config.json',metadata)
    started = time.monotonic()
    working_step = pending['next_step'] if pending else step
    state = {'status':'running','step':step,'updates':args.updates or None,'run_dir':str(out),
        'models':metadata['models'],'reward':metadata['reward'],
        'device':torch.cuda.get_device_name(0) if device=='cuda' else device,
        'evaluation':None,'latest_training':{},'population':args.population}
    if args.resume and (out/'evaluations.jsonl').exists():
        lines = (out/'evaluations.jsonl').read_text(encoding='utf-8').splitlines()
        if lines: state['evaluation'] = json.loads(lines[-1])

    def append(name,row):
        with (out/name).open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

    def publish(phase):
        state.update(phase=phase,step=step,elapsed_seconds=round(elapsed_prior+time.monotonic()-started,2),
            trainer_pid=os.getpid(),heartbeat_unix=time.time(),
            training_games={'transformer':training_games},test_games=test_games,
            alternating=controller.summary(),unfreezing={'transformer':mask.summary()},
            plateau=controller.plateau_summary(step),competition_step=working_step,
            experience={name:{'competition_steps':learner_steps[i],
                        'training_games_per_individual':learner_steps[i]*args.games_per_pair,
                        'sharing_boards_per_individual':sharing_examples[i].tolist()}
                        for i,name in enumerate('AB')},
            sharing={name:{'updates':sharing_totals[i], 'last':last_sharing[i],
                           'teacher_evaluation':teacher_reports[i],
                           'mean_boards_per_individual':float(sharing_examples[i].mean())}
                     for i,name in enumerate('AB')}, teacher_games=teacher_games,
            diversity={name:dict(diversity[i],target=targets[i],met=diversity[i]['value']+1e-10>=targets[i],
                                total_alien_updates=alien_totals[i]) for i,name in enumerate('AB')},
            pending_stage=pending['stage'] if pending else None)
        atomic_json(out/'progress.json',state)

    def checkpoint():
        data={'protocol':PROTOCOL,'args':vars(args),'step':step,'groups':[g.state_dict() for g in groups],
            'controller':controller.state_dict(),'mask':mask.state_dict(),'tokens':tokens,'position_masks':masks,
            'targets':targets,'diversity':diversity,'alien_totals':alien_totals,'pending':pending,
            'learner_steps':learner_steps,
            'teacher_reports':teacher_reports,'teacher_games':teacher_games,'sharing_totals':sharing_totals,
            'sharing_examples':sharing_examples,'last_sharing':last_sharing,
            'training_games':training_games,'test_games':test_games,'train_rng':train_rng.bit_generator.state,
            'test_rng':test_rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
            'sharing_rng':sharing_rng.bit_generator.state,'teacher_rng':teacher_rng.bit_generator.state,
            'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else [],
            'elapsed':elapsed_prior+time.monotonic()-started}
        torch.save(data,out/'latest.tmp')
        replace_with_retry(out/'latest.tmp',out/'latest.pt')

    def expand_if_released(event):
        if event and event['released']:
            for group in groups: group.expand(mask)
            teacher_reports[:] = [None,None]

    def repair_diversity():
        nonlocal diversity
        for i,group in enumerate(groups):
            diversity[i] = population_diversity(group.policy,tokens,masks,args.chunk)
        while any(value['value']+1e-10 < target for value,target in zip(diversity,targets)):
            if (out/'STOP').exists():
                return False
            for i,group in enumerate(groups):
                if diversity[i]['value']+1e-10 >= targets[i]: continue
                publish('alien_'+ 'AB'[i])
                update=alien_update(group,tokens,masks,args.chunk)
                alien_totals[i] += 1
                pending['alien_rounds'][i] += 1
                diversity[i] = population_diversity(group.policy,tokens,masks,args.chunk)
                append('alien_events.jsonl',{'step':pending['next_step'],'type':'alien_learning',
                    'group':'AB'[i],'round':alien_totals[i],'value':diversity[i]['value'],
                    'target':targets[i],'update':update})
                publish('alien_'+ 'AB'[i])
        return True

    def diagnostic():
        publish('evaluation_fixed')
        state['evaluation']=evaluate_population(groups,initial,mask,args.eval_games_per_model,
                        args.seed+900000,step,lambda:publish('evaluation_fixed'))
        append('evaluations.jsonl',state['evaluation'])
        publish('evaluation_complete')

    try:
        publish('initializing')
        if args.publish:
            Path(__file__).with_name('latest_run.txt').write_text(str(out),encoding='utf-8')
        if not args.resume:
            checkpoint()
            diagnostic()
        while args.updates==0 or step<args.updates or pending:
            if (out/'STOP').exists():
                state['status']='stopped'
                break
            if pending is None:
                n=step+1
                working_step=n
                before=controller.before_step(n,mask)
                if before:
                    expand_if_released(before)
                    append('switches.jsonl',before)
                learner=controller.learner
                publish('competition_play')
                pairing=train_rng.permutation(args.population)
                match=population_match([g.policy for g in groups],args.population,args.games_per_pair,
                                       train_rng,collect=learner,record=True,opponent_indices=pairing)
                publish('competition_update')
                update=competition_update(groups[learner],match,learner,args.chunk)
                learner_steps[learner]+=1
                training_games += match['games']
                pending={'next_step':n,'stage':'sharing','sharing_records':match['records'],
                    'learner':learner,'before':before,'alien_rounds':[0,0],
                    'row':{'step':n,'learner':'AB'[learner],'pairing':pairing.tolist(),
                        'a_wins':match['a_wins'],'b_wins':match['b_wins'],
                        'draws':match['draws'],'games':match['games'],
                        'a_mean_reward':float(match['margins'].mean()),'b_mean_reward':-float(match['margins'].mean()),
                        'agents':{'AB'[learner]:update}},
                    'game':{'step':n,'family':'transformer','pair':0,'b_index':int(pairing[0]),'moves':match['trace'],
                        'black_stones':match['black_stones'],'white_stones':match['white_stones']}}
            if pending['stage']=='sharing':
                learner=pending['learner']
                report=teacher_reports[learner]
                if report is None or learner_steps[learner]-report['learner_steps']>=args.teacher_eval_every:
                    publish('sharing_evaluation')
                    report=evaluate_teachers(groups[learner],args.teacher_games_per_opponent,teacher_rng,
                                             lambda:publish('sharing_evaluation'))
                    report.update(step=pending['next_step'],learner_steps=learner_steps[learner])
                    teacher_reports[learner]=report
                    teacher_games+=report['games']
                if (out/'STOP').exists():
                    state['status']='stopped'
                    break
                publish('sharing_'+ 'AB'[learner])
                plan=sharing_plan(groups[learner],pending['sharing_records'],report['scores'],sharing_rng,
                                  args.sharing_positions,args.teacher_fraction,args.teachers_per_student,args.chunk)
                shared=sharing_update(groups[learner],plan,args.chunk)
                shared.update(step=pending['next_step'],group='AB'[learner],teacher_evaluation_step=report['step'])
                if shared['students']:
                    sharing_totals[learner]+=1
                    sharing_examples[learner]+=plan['enabled']*args.sharing_positions
                last_sharing[learner]=shared
                pending['row']['sharing']=shared
                del pending['sharing_records']
                pending['stage']='alien'
                publish('sharing_complete')
            if pending['stage'] in ('alien','after_copy_alien'):
                if not repair_diversity():
                    state['status']='stopped'
                    break
                pending['stage']='test' if pending['stage']=='alien' else 'finish'
            if pending['stage']=='test':
                publish('evaluation_switch')
                test_pairing=test_rng.integers(args.population,size=args.population)
                match_test=population_match([g.policy for g in groups],args.population,args.test_games_per_pair,
                                            test_rng,opponent_indices=test_pairing)
                test_games += match_test['games']
                event=controller.observe(pending['next_step'],match_test['margins'],mask)
                pending['row']['held_out_mean_margin_a']=float(match_test['margins'].mean())
                pending['row']['test_pairing']=test_pairing.tolist()
                pending['event']=event
                pending['stage']='finish'
                if event:
                    if event['type']=='opponent_copy':
                        groups[pending['learner']].copy_from(groups[1-pending['learner']])
                        teacher_reports[pending['learner']]=None
                        append('copies.jsonl',event)
                        pending['stage']='after_copy_alien'
                    else:
                        expand_if_released(event)
                        append('switches.jsonl',event)
                if pending['stage']=='after_copy_alien':
                    if not repair_diversity():
                        state['status']='stopped'
                        break
                    pending['stage']='finish'
            if pending['stage']=='finish':
                step=pending['next_step']
                working_step=step
                row=pending['row']
                event=pending.get('event')
                row.update(active_percent=mask.summary()['percent'],test=controller.test.summary(),
                    loss_test=controller.loss_test.summary(),plateau=controller.plateau_summary(step),
                    switch=event if event and event['type']=='learner_switch' else pending['before'],
                    copy=event if event and event['type']=='opponent_copy' else None,
                    diversity={name:diversity[i] for i,name in enumerate('AB')},alien_rounds=pending['alien_rounds'],
                    training_games=training_games,test_games=test_games)
                append('metrics.jsonl',row)
                if row['sharing']['students']:
                    append('sharing_sessions.jsonl',dict(row['sharing'],type='shared_learning'))
                if any(pending['alien_rounds']):
                    append('alien_sessions.jsonl',{'step':step,'type':'alien_learning',
                        'rounds':{'AB'[i]:rounds for i,rounds in enumerate(pending['alien_rounds']) if rounds},
                        'diversity':{name:diversity[i]['value'] for i,name in enumerate('AB')}})
                state['latest_training']={'transformer':row}
                atomic_json(out/'last_game_transformer.json',pending['game'])
                pending=None
                if event or row['switch'] or step%args.save_every==0:
                    publish('saving')
                    checkpoint()
                publish('batch_complete')
                print(json.dumps({'step':step,'learner':row['learner'],'next_learner':'AB'[controller.learner],
                    'percent':row['active_percent'],'p':controller.test.p_value,'loss_p':controller.loss_test.p_value,
                    'diversity':[d['value'] for d in diversity],'alien_rounds':row['alien_rounds']}),flush=True)
                if step%args.eval_every==0 and not (out/'STOP').exists(): diagnostic()
        else:
            state['status']='completed'
        publish('saving')
        checkpoint()
        publish(state['status'])
    except KeyboardInterrupt:
        state.update(status='interrupted',error='Use STOP for a resumable partial alien-learning checkpoint.')
        publish('interrupted')
    except Exception:
        state.update(status='failed',error=traceback.format_exc())
        publish('failed')
        raise


if __name__=='__main__':
    main()
