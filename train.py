"""Two independent agents; on-policy terminal-reward REINFORCE.

Rollouts freeze both policies. Recompute log probabilities, accumulate gradients
over all decisions, then take exactly one optimizer step per model per batch.
Reward can be terminal stone margin or win-only; gamma=1. The optional
normalized legal-policy entropy reward defaults to zero for legacy runs.
No weight decay, search, or genetic operations.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import json
import itertools
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch

from model import ModelConfig, build_model, masked_log_probs
from curriculum import ProgressiveUnfreezer
from runtime_io import atomic_json, replace_with_retry
from entropy import normalized_entropy, rollout_entropy_returns
from othello import (BITS, POSITION, U64, apply_moves, bits_to_mask,
                     initial_boards, legal_moves, row_tokens, stone_counts,
                     terminal_rewards)


def policy_logits(policy, tokens, device):
    if isinstance(policy, str):
        if policy == 'random':
            return np.zeros((len(tokens), 64), dtype=np.float32)
        if policy == 'position':
            return np.broadcast_to(POSITION, (len(tokens), 64))
        raise ValueError(policy)
    with torch.no_grad():
        scores = policy(torch.as_tensor(tokens, device=device, dtype=torch.long))
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError('Non-finite policy logits')
    return scores.cpu().numpy()


def play_batch(policies, n, rng, device, collect=False, record=False, reward_mode='win'):
    """Half the games: policy 0 is black; half: policy 1 is black.

    Neural policies sample at temperature 1. Position heuristic chooses maximum
    positional weight, breaking ties randomly. Games start at the standard board.
    """
    if n < 2 or n % 2:
        raise ValueError('Games per batch must be an even number >=2')
    black, white = initial_boards(n)
    black_agents = np.arange(n) % 2
    done = np.zeros(n, dtype=bool)
    records = [[], []]
    trace = []
    for turn in range(122):
        color = turn % 2
        own, opp = (black, white) if color == 0 else (white, black)
        legal = legal_moves(own, opp)
        done |= (legal == 0) & (legal_moves(opp, own) == 0)
        if done.all():
            break
        players = black_agents if color == 0 else 1 - black_agents
        playable = (~done) & (legal != 0)
        moves = np.zeros(n, dtype=np.uint64)
        if record and not done[0] and not playable[0]:
            trace.append({'color': color, 'agent': int(players[0]), 'action': None})
        for agent, policy in enumerate(policies):
            ids = np.flatnonzero(playable & (players == agent))
            if not len(ids):
                continue
            tokens = row_tokens(own[ids], opp[ids])
            masks = bits_to_mask(legal[ids])
            scores = policy_logits(policy, tokens, device)
            noise = rng.gumbel(size=scores.shape)
            if isinstance(policy, str) and policy == 'position':
                # Integer positional weights: noise may only break equal scores.
                noise = rng.random(size=scores.shape) * .01
            actions = np.argmax(np.where(masks, scores + noise, -np.inf), axis=-1)
            selected = U64(1) << actions.astype(np.uint64)
            if not np.all((selected & legal[ids]) == selected):
                raise AssertionError('Illegal policy action')
            moves[ids] = selected
            if collect:
                records[agent].append((tokens.copy(), masks.copy(), actions.copy(), ids.copy()))
            if record and 0 in ids:
                local = int(np.flatnonzero(ids == 0)[0])
                trace.append({'color': color, 'agent': agent, 'action': int(actions[local])})
        ids = np.flatnonzero(playable)
        own[ids], opp[ids] = apply_moves(own[ids], opp[ids], moves[ids])
        if np.any(black & white):
            raise AssertionError('Overlapping stones')
    else:
        raise AssertionError('Othello exceeded the maximum turn count')
    rewards = [terminal_rewards(black, white, black_agents, agent, reward_mode) for agent in (0, 1)]
    margin = stone_counts(black) - stone_counts(white)
    a_margin = np.where(black_agents == 0, margin, -margin)
    return {'records': records, 'rewards': rewards, 'black_agents': black_agents,
            'black': black, 'white': white, 'trace': trace,
            'a_wins': int((a_margin > 0).sum()), 'b_wins': int((a_margin < 0).sum()),
            'draws': int((margin == 0).sum()), 'games': n,
            'a_mean_margin': float(np.where(black_agents == 0, margin, -margin).mean())}


def update_policy(model, optimizer, records, rewards, black_agents, agent,
                  baseline, chunk_size=256, max_grad_norm=1.0, unfreezer=None, entropy_coef=0.):
    """Previous-batch color-specific EMA baseline does not alter reward.

    Divide the SUM of action losses by games, not by moves or batch chunks;
    this is the undiscounted episodic policy gradient estimator. With an
    entropy reward, include future entropy returns and its direct derivative.
    No current-batch centering or per-trajectory normalization is performed.
    """
    device = next(model.parameters()).device
    tokens, masks, actions, ids = [np.concatenate([r[i] for r in records]) for i in range(4)]
    colors = (black_agents != agent).astype(np.int64)
    advantage = rewards[ids] - np.asarray(baseline)[colors[ids]]
    if not np.isfinite(entropy_coef) or entropy_coef<0:
        raise ValueError('Entropy reward coefficient must be finite and nonnegative')
    entropy_totals=np.zeros(len(rewards))
    if entropy_coef:
        future,entropy_totals=rollout_entropy_returns(model,tokens,masks,ids,len(rewards),chunk_size)
        advantage=advantage+entropy_coef*future
    optimizer.zero_grad(set_to_none=True)
    loss_sum, entropy_sum, normalized_sum, choice_count = 0., 0., 0., 0
    for start in range(0, len(tokens), chunk_size):
        end = start + chunk_size
        logits = model(torch.as_tensor(tokens[start:end], device=device, dtype=torch.long))
        legal = torch.as_tensor(masks[start:end], device=device, dtype=torch.bool)
        log_probs = masked_log_probs(logits, legal)
        chosen = log_probs.gather(1, torch.as_tensor(actions[start:end, None], device=device)).squeeze(1)
        adv = torch.as_tensor(advantage[start:end], device=device, dtype=chosen.dtype)
        loss = -(chosen * adv).sum() / len(rewards)
        ent,h_norm=normalized_entropy(log_probs,legal)
        if entropy_coef:
            # Full entropy-reward gradient: future rewards above plus the
            # explicit derivative of H(pi(.|s)) at the current state.
            loss=loss-entropy_coef*h_norm.sum()/len(rewards)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Non-finite training loss')
        loss.backward()
        loss_sum += float(loss.detach())
        with torch.no_grad():
            entropy_sum += float(ent.sum())
            choices=legal.sum(-1)>1
            normalized_sum+=float(h_norm[choices].sum())
            choice_count+=int(choices.sum())
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    if missing:
        raise AssertionError(f'Parameters disconnected from autograd: {missing}')
    if unfreezer is not None:
        unfreezer.mask_gradients(model, optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm,
                                              error_if_nonfinite=True)
    before = [p.detach().clone() for p in model.parameters()]
    optimizer.step()
    if unfreezer is not None:
        unfreezer.restore_frozen(model, before)
    with torch.no_grad():
        delta = sum((p - old).square().sum() for p, old in zip(model.parameters(), before)).sqrt()
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise FloatingPointError('Non-finite updated parameter')
    for color in (0, 1):
        augmented_rewards=rewards+entropy_coef*entropy_totals
        baseline[color] = .9 * baseline[color] + .1 * float(augmented_rewards[colors == color].mean())
    return {'loss': loss_sum, 'entropy': entropy_sum / len(tokens),
            'gradient_norm': float(grad_norm), 'parameter_delta_l2': float(delta),
            'decisions': len(tokens), 'baseline_black': baseline[0], 'baseline_white': baseline[1],
            'mean_reward': float(rewards.mean()), 'normalized_entropy':normalized_sum/max(1,choice_count),
            'entropy_coefficient':entropy_coef,'mean_entropy_bonus':float(entropy_coef*entropy_totals.mean()),
            'mean_augmented_reward':float((rewards+entropy_coef*entropy_totals).mean())}


def summarize(result):
    n = result['games']
    wins = result['a_wins']
    p = wins / n
    z = 1.95996398454
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    by_color = {}
    for color, label in ((0, 'black'), (1, 'white')):
        mask = result['black_agents'] == color
        own_margin = np.where(result['black_agents'] == 0,
                              stone_counts(result['black']) - stone_counts(result['white']),
                              stone_counts(result['white']) - stone_counts(result['black']))
        by_color[label] = {'wins': int((own_margin[mask] > 0).sum()),
                           'draws': int((own_margin[mask] == 0).sum()), 'games': int(mask.sum())}
    return {'wins': wins, 'draws': result['draws'], 'losses': result['b_wins'],
            'games': n, 'win_rate': p, 'win_rate_ci95': [float(center-half), float(center+half)],
            'mean_margin_diagnostic_only': result['a_mean_margin'], 'by_color': by_color}


def evaluate(groups, initial, games, seed, device, step):
    result = {'step': step, 'families': {}, 'cross': {}}
    # Evaluation has its own RNG; it cannot change training trajectories.
    for family, group in groups.items():
        result['families'][family] = {}
        for agent in (0, 1):
            scores = {}
            for opponent_id, opponent in enumerate(('random', 'position', initial[family][1-agent])):
                name = ('random', 'position', 'initial')[opponent_id]
                match = play_batch([group['models'][agent], opponent], games,
                                   np.random.default_rng(seed + 100*agent + opponent_id), device)
                scores[name] = summarize(match)
            result['families'][family]['AB'[agent]] = scores
    if 'transformer' in groups and 'taylor' in groups:
        for a in (0, 1):
            for b in (0, 1):
                match = play_batch([groups['transformer']['models'][a], groups['taylor']['models'][b]],
                                   games, np.random.default_rng(seed + 1000 + 100*a + b), device)
                result['cross'][f'Transformer {"AB"[a]} vs Taylor {"AB"[b]}'] = summarize(match)
    return result


def save_checkpoint(path, groups, step, args, elapsed, training_games):
    data = {'version': 1, 'step': step, 'args': vars(args), 'elapsed': elapsed,
            'training_games': training_games, 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'groups': {family: {
                'models': [m.state_dict() for m in g['models']],
                'optimizers': [o.state_dict() for o in g['optimizers']],
                'unfreezer': g['unfreezer'].state_dict() if g.get('unfreezer') else None,
                'baselines': g['baselines'], 'rng': g['rng'].bit_generator.state,
            } for family, g in groups.items()}}
    temp = path.with_suffix('.tmp')
    torch.save(data, temp)
    replace_with_retry(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--families', nargs='+', choices=['transformer', 'taylor'], default=['transformer'])
    parser.add_argument('--reward', choices=['win', 'margin'], default='margin')
    parser.add_argument('--initial-percent', type=int, default=1)
    parser.add_argument('--release-window', type=int, default=15)
    parser.add_argument('--release-reversals', type=int, default=3)
    parser.add_argument('--min-hold-updates', type=int, default=50)
    parser.add_argument('--updates', type=int, default=0, help='0 = unlimited')
    parser.add_argument('--games-per-batch', type=int, default=128)
    parser.add_argument('--eval-games', type=int, default=64)
    parser.add_argument('--eval-every', type=int, default=25)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--width', type=int, default=96)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dimension', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad-chunk', type=int, default=256)
    parser.add_argument('--seed', type=int, default=20261022)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--out', default='')
    parser.add_argument('--resume', default='')
    parser.add_argument('--publish', action='store_true', help='Select this run in the live monitor (omit for tests)')
    args = parser.parse_args()
    if (args.updates < 0 or args.eval_every < 1 or args.save_every < 1 or args.grad_chunk < 1
            or args.dimension < 1 or args.lr <= 0 or len(set(args.families)) != len(args.families)
            or not 1 <= args.initial_percent <= 100 or args.release_window < 3
            or not 1 <= args.release_reversals < args.release_window
            or args.min_hold_updates < 0
            or any(n < 2 or n % 2 for n in (args.games_per_batch, args.eval_games))):
        parser.error('Invalid run configuration (games must be positive even numbers)')
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    if device == 'auto':
        device = 'cpu'
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = ModelConfig(args.width, args.layers, args.heads)
    out = Path(args.out or ('runs/' + datetime.now().strftime('%Y%m%d-%H%M%S'))).resolve()
    out.mkdir(parents=True, exist_ok=True)
    args.out = str(out)
    if (out / 'config.json').exists() and not args.resume:
        parser.error('Output already contains an experiment; choose a new directory or --resume')
    groups, initial = {}, {}
    for family in args.families:
        torch.manual_seed(args.seed + (0 if family == 'transformer' else 100))
        models = [build_model(family, config, args.dimension).to(device) for _ in range(2)]
        groups[family] = {'models': models,
                          'optimizers': [torch.optim.Adam(m.parameters(), lr=args.lr) for m in models],
                          'baselines': [[0., 0.], [0., 0.]] if args.reward == 'margin' else [[.5, .5], [.5, .5]],
                          'unfreezer': ProgressiveUnfreezer(models[0], args.seed + 500, args.initial_percent,
                                                           args.release_window, args.release_reversals,
                                                           args.min_hold_updates),
                          'rng': np.random.default_rng(args.seed)}
        initial[family] = [copy.deepcopy(m).eval().requires_grad_(False) for m in models]
    step, elapsed_prior, training_games = 0, 0., {family: 0 for family in args.families}
    if args.resume:
        data = torch.load(args.resume, map_location=device, weights_only=False)
        old = data['args']
        for key in ('families', 'width', 'layers', 'heads', 'dimension', 'seed', 'lr', 'games_per_batch'):
            if old[key] != getattr(args, key):
                parser.error(f'Resume requires original --{key.replace("_", "-")}={old[key]}')
        for key, legacy in (('reward', 'win'), ('initial_percent', 100), ('release_window', 10), ('release_reversals', 2)):
            if old.get(key, legacy) != getattr(args, key):
                parser.error(f'Resume requires original --{key.replace("_", "-")}={old.get(key, legacy)}')
        if Path(old['out']).resolve() != out:
            parser.error('Resume must use the original --out directory (logs and initial opponents)')
        for family, group in groups.items():
            saved = data['groups'][family]
            for agent in (0, 1):
                group['models'][agent].load_state_dict(saved['models'][agent])
                group['optimizers'][agent].load_state_dict(saved['optimizers'][agent])
            group['baselines'] = saved['baselines']
            group['rng'].bit_generator.state = saved['rng']
            if saved.get('unfreezer'):
                group['unfreezer'].load_state_dict(saved['unfreezer'], override_min_hold=True)
        step, elapsed_prior = data['step'], data['elapsed']
        training_games = data['training_games']
        torch.set_rng_state(data['torch_rng'].cpu())
        if device == 'cuda' and data['cuda_rng']:
            torch.cuda.set_rng_state_all([s.cpu() for s in data['cuda_rng']])
        # Discard logs newer than the last atomic checkpoint before continuing.
        for name in ('metrics.jsonl', 'evaluations.jsonl'):
            path = out / name
            if path.exists():
                rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
                path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows if r['step'] <= step), encoding='utf-8')
        if (out / 'metrics.csv').exists():
            with (out / 'metrics.csv').open(encoding='utf-8', newline='') as file:
                rows = list(csv.DictReader(file))
            with (out / 'metrics.csv').open('w', encoding='utf-8', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(r for r in rows if int(r['step']) <= step)
    else:
        torch.save({f: [m.state_dict() for m in initial[f]] for f in initial}, out / 'initial.pt')
    descriptions = {f: {**g['models'][0].description(),
                        'all_parameters_trainable': g['unfreezer'].percent == 100} for f, g in groups.items()}
    metadata = {**vars(args), 'device_used': device,
                'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
                'torch': torch.__version__, 'numpy': np.__version__, 'models': descriptions,
                'reward': ({'mode': 'margin', 'formula': 'own_stones - opponent_stones', 'range': [-64, 64]}
                           if args.reward == 'margin' else {'mode': 'win', 'win': 1, 'draw': 0, 'loss': 0}),
                'unfreezing': {'initial_percent': args.initial_percent, 'increment_percent': 1,
                              'min_hold_updates': args.min_hold_updates,
                              'window': args.release_window, 'leader_reversals': args.release_reversals,
                              'full_window_required': True, 'ties': 'ignored', 'reset_after_release': True,
                              'mask': 'same nested scalar mask for A and B; distributed across all tensors'},
                'algorithm': 'on-policy REINFORCE, previous-batch EMA baseline by color, gamma=1',
                'entropy_bonus': 0, 'weight_decay': 0, 'training_temperature': 1,
                'evaluation_temperature': 1, 'cross_matches': 'evaluation only; no gradients',
                'evaluation_seed': args.seed + 900000,
                'initial_opponent': 'other agent from the same family before any training',
                'resume_exactness': 'RNG and optimizers restored; CUDA kernels may be nondeterministic'}
    if args.resume:
        metadata['resumed_from_step'] = step
        metadata['minimum_hold_before_resume'] = old.get('min_hold_updates', 0)
        metadata['minimum_hold_changed_on_resume'] = old.get('min_hold_updates', 0) != args.min_hold_updates
    atomic_json(out / 'config.json', metadata)
    if args.publish:
        Path(__file__).with_name('latest_run.txt').write_text(str(out), encoding='utf-8')
    started = time.monotonic()
    status = {'status': 'running', 'phase': 'initializing', 'step': step, 'updates': args.updates,
              'run_dir': str(out), 'models': descriptions, 'training_games': training_games.copy(),
              'reward': metadata['reward'], 'device': metadata['gpu'] or device,
              'elapsed_seconds': elapsed_prior, 'evaluation': None, 'latest_training': {}}
    # Preserve already-recorded evaluation on resume.
    eval_log = out / 'evaluations.jsonl'
    if args.resume and eval_log.exists() and eval_log.read_text(encoding='utf-8').strip():
        status['evaluation'] = json.loads(eval_log.read_text(encoding='utf-8').splitlines()[-1])

    def publish(phase):
        status['unfreezing'] = {f: g['unfreezer'].summary() for f, g in groups.items()}
        status.update(phase=phase, step=step, training_games=training_games.copy(),
                      trainer_pid=os.getpid(), heartbeat_unix=time.time(),
                      elapsed_seconds=round(time.monotonic() - started + elapsed_prior, 2))
        atomic_json(out / 'progress.json', status)

    def checkpoint():
        save_checkpoint(out / 'latest.pt', groups, step, args,
                        time.monotonic() - started + elapsed_prior, training_games)

    def run_evaluation():
        publish('evaluation')
        scores = evaluate(groups, initial, args.eval_games, args.seed + 900000, device, step)
        with eval_log.open('a', encoding='utf-8') as file:
            file.write(json.dumps(scores, ensure_ascii=False) + '\n')
        status['evaluation'] = scores
        publish('evaluation_complete')
        print(json.dumps({'event': 'evaluation', 'step': step, 'scores': scores}, ensure_ascii=False), flush=True)

    try:
        publish('initializing')
        if step == 0 and not args.resume:
            checkpoint()
            run_evaluation()
        for next_step in itertools.count(step + 1):
            if args.updates and next_step > args.updates:
                status['status'] = 'completed'
                break
            if (out / 'STOP').exists():
                status['status'] = 'stopped'
                break
            row = {'step': next_step, 'families': {}}
            for family, group in groups.items():
                publish(f'selfplay_{family}')
                match = play_batch(group['models'], args.games_per_batch, group['rng'], device,
                                   collect=True, record=True, reward_mode=args.reward)
                training = {'a_wins': match['a_wins'], 'b_wins': match['b_wins'],
                            'draws': match['draws'], 'games': match['games'], 'agents': {},
                            'a_mean_reward': float(match['rewards'][0].mean()),
                            'b_mean_reward': float(match['rewards'][1].mean()),
                            'active_percent_used': group['unfreezer'].percent}
                publish(f'update_{family}')
                # All trajectories were generated before either policy changes.
                for agent in (0, 1):
                    update = update_policy(group['models'][agent], group['optimizers'][agent],
                                           match['records'][agent], match['rewards'][agent],
                                           match['black_agents'], agent, group['baselines'][agent],
                                           args.grad_chunk, unfreezer=group['unfreezer'])
                    training['agents']['AB'[agent]] = update
                released = group['unfreezer'].observe(next_step, match['a_wins'], match['b_wins'], match['draws'])
                training['unfreezing'] = group['unfreezer'].summary()
                training['released'] = released
                training_games[family] += args.games_per_batch
                row['families'][family] = training
                atomic_json(out / f'last_game_{family}.json', {
                    'step': next_step, 'family': family, 'black_agent': 'A', 'white_agent': 'B',
                    'moves': match['trace'], 'black_stones': int(stone_counts(match['black'][:1])[0]),
                    'white_stones': int(stone_counts(match['white'][:1])[0]),
                    'final_cells': (bits_to_mask(match['black'][:1]).astype(int)
                                    - bits_to_mask(match['white'][:1]).astype(int))[0].tolist()})
            step = next_step
            row['elapsed_seconds'] = round(time.monotonic() - started + elapsed_prior, 2)
            status['latest_training'] = row['families']
            with (out / 'metrics.jsonl').open('a', encoding='utf-8') as file:
                file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            write_csv(out / 'metrics.csv', row, training_games)
            if step % args.save_every == 0 or any(t['released'] for t in row['families'].values()):
                checkpoint()
            publish('batch_complete')
            print(json.dumps({'event': 'training', **row}, ensure_ascii=False), flush=True)
            if (step % args.eval_every == 0 or step == args.updates) and not (out / 'STOP').exists():
                run_evaluation()
        else:
            status['status'] = 'completed'
        if status['status'] == 'running':
            status['status'] = 'stopped'
        checkpoint()
        publish(status['status'])
    except KeyboardInterrupt:
        # A partially updated batch cannot be saved with the previous step index.
        # Keep the most recent complete-batch checkpoint for consistent resume.
        status['status'] = 'interrupted'
        status['error'] = 'Use STOP for graceful save; resume latest.pt after an interrupt.'
        publish('interrupted')
    except Exception:
        status['status'] = 'failed'
        status['error'] = traceback.format_exc()
        try:
            publish('failed')
        except OSError:
            # Reporting an I/O failure must never hide the original exception.
            print('Could not publish failure status:\n' + status['error'], file=sys.stderr, flush=True)
        raise


CSV_FIELDS = ['step', 'family', 'agent', 'training_games', 'wins', 'draws', 'losses',
              'win_rate', 'mean_reward', 'active_percent_used', 'active_percent_next',
              'loss', 'entropy', 'gradient_norm', 'parameter_delta_l2', 'elapsed_seconds']


def write_csv(path, row, games):
    exists = path.exists()
    with path.open('a', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        for family, training in row['families'].items():
            for agent in ('A', 'B'):
                wins = training['a_wins' if agent == 'A' else 'b_wins']
                metrics = training['agents'][agent]
                writer.writerow({'step': row['step'], 'family': family, 'agent': agent,
                                 'training_games': games[family], 'wins': wins, 'draws': training['draws'],
                                 'losses': training['games'] - wins - training['draws'],
                                 'win_rate': wins / training['games'],
                                 'mean_reward': metrics['mean_reward'],
                                 'active_percent_used': training['active_percent_used'],
                                 'active_percent_next': training['unfreezing']['percent'],
                                 **{k: metrics[k] for k in ('loss', 'entropy', 'gradient_norm', 'parameter_delta_l2')},
                                 'elapsed_seconds': row['elapsed_seconds']})


if __name__ == '__main__':
    main()
