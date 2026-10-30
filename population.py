"""Independent Transformer populations with sparse trainable scalar values.

Each individual owns a full set of weights. Only opened scalar values are
optimizer parameters; frozen tensors remain buffers. torch.func.vmap batches
the independent models without sharing their learned weights.
"""
import copy
import numpy as np
import torch
from torch.func import functional_call, vmap
from torch.nn.attention import sdpa_kernel, SDPBackend

from model import OthelloGPT, masked_log_probs
from othello import (U64, POSITION, initial_boards, legal_moves, bits_to_mask,
                     row_tokens, apply_moves, stone_counts)


class PopulationPolicy:
    def __init__(self, config, size, mask, device='cpu', weights=None):
        self.config, self.size, self.device = config, size, torch.device(device)
        self.template = OthelloGPT(config).to('meta')
        if weights is None:
            models = [OthelloGPT(config) for _ in range(size)]
            self.weights = {name: torch.stack([m.state_dict()[name] for m in models]).to(device)
                            for name in models[0].state_dict()}
            del models
        else:
            self.weights = {name: value.to(device).clone() for name, value in weights.items()}
        self.active, self.indices = {}, {}
        self.configure(mask)
        self._forward = vmap(lambda pars, rows: functional_call(self.template, pars, (rows,)),
                             in_dims=(0, 0), randomness='error')

    def configure(self, mask):
        # Commit current opened values before extending the nested mask.
        with torch.no_grad():
            for name, parameter in self.active.items():
                self.weights[name].flatten(1).scatter_(1, self.indices[name], parameter)
        self.active, self.indices = {}, {}
        for name, weight in self.weights.items():
            count = int(mask.masks[name].sum())
            if count:
                index = mask.orders[name][:count].to(self.device)[None].expand(self.size, -1)
                self.indices[name] = index
                self.active[name] = torch.nn.Parameter(weight.flatten(1).gather(1, index).clone())

    def materialize(self):
        return {name: weight.flatten(1).scatter(1, self.indices[name], self.active[name]).view_as(weight)
                if name in self.active else weight for name, weight in self.weights.items()}

    def __call__(self, rows, individuals=None):
        # Short 9-token contexts: math attention supports efficient vmap,
        # avoiding a per-model fallback in flash-attention batching rules.
        with sdpa_kernel(SDPBackend.MATH):
            parameters = self.materialize()
            if individuals is not None:
                ids = torch.as_tensor(individuals, device=self.device, dtype=torch.long)
                parameters = {name:value.index_select(0,ids) for name,value in parameters.items()}
            return self._forward(parameters, rows)

    @torch.no_grad()
    def logits(self, rows, individuals=None):
        return self(torch.as_tensor(rows, device=self.device, dtype=torch.long),individuals).cpu().numpy()

    @torch.no_grad()
    def snapshot(self):
        return {name: value.detach().cpu().clone() for name, value in self.materialize().items()}

    @torch.no_grad()
    def load_weights(self, weights):
        for name, value in weights.items():
            self.weights[name].copy_(value)
            if name in self.active:
                self.active[name].copy_(self.weights[name].flatten(1).gather(1, self.indices[name]))


class PopulationOptimizer:
    def __init__(self, policy, lr):
        self.policy, self.lr = policy, lr
        self.optimizer = torch.optim.Adam(list(policy.active.values()), lr=lr)

    def rebind(self, old_active):
        old = self.optimizer
        self.optimizer = torch.optim.Adam(list(self.policy.active.values()), lr=self.lr)
        for name, parameter in self.policy.active.items():
            previous = old_active.get(name)
            if previous is None or previous not in old.state:
                continue
            migrated = {}
            for key, value in old.state[previous].items():
                if isinstance(value, torch.Tensor) and value.shape == previous.shape:
                    expanded = torch.zeros_like(parameter)
                    expanded[:, :previous.shape[1]] = value
                    migrated[key] = expanded
                else:
                    migrated[key] = value.clone() if isinstance(value, torch.Tensor) else value
            self.optimizer.state[parameter] = migrated

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self):
        # Clip each individual's gradient independently, not the whole group.
        norms = torch.zeros(self.policy.size, device=self.policy.device)
        before = {}
        for name, p in self.policy.active.items():
            if p.grad is None or not bool(torch.isfinite(p.grad).all()):
                raise FloatingPointError('Missing or non-finite population gradient')
            norms += p.grad.square().sum(1)
            before[name] = p.detach().clone()
        norms = norms.sqrt()
        factor = (1 / (norms + 1e-6)).clamp(max=1)
        for p in self.policy.active.values():
            p.grad.mul_(factor[:, None])
        self.optimizer.step()
        delta = torch.zeros_like(norms)
        for name, p in self.policy.active.items():
            if not bool(torch.isfinite(p).all()):
                raise FloatingPointError('Non-finite population weights')
            delta += (p-before[name]).square().sum(1)
        return {'gradient_norm': float(norms.mean()), 'parameter_delta_l2': float(delta.sqrt().mean())}

    def reset(self):
        self.optimizer.state.clear()
        self.zero_grad()


class PopulationGroup:
    def __init__(self, config, size, mask, device, lr=1e-4, alien_lr=1e-4, weights=None, sharing_lr=1e-5):
        self.policy = PopulationPolicy(config, size, mask, device, weights)
        self.competition = PopulationOptimizer(self.policy, lr)
        self.alien = PopulationOptimizer(self.policy, alien_lr)
        self.sharing = PopulationOptimizer(self.policy, sharing_lr)
        self.baseline = np.zeros((size, 2))

    def expand(self, mask):
        old = self.policy.active
        self.policy.configure(mask)
        self.competition.rebind(old)
        self.alien.rebind(old)
        self.sharing.rebind(old)

    def copy_from(self, other):
        # A_i <- B_i (or vice versa), including frozen scalar values.
        with torch.no_grad():
            self.policy.load_weights(other.policy.materialize())
        self.competition.reset()
        self.alien.reset()
        self.sharing.reset()
        self.baseline[:] = 0

    def state_dict(self):
        return {'weights': self.policy.snapshot(), 'baseline': self.baseline,
                'competition_optimizer': self.competition.optimizer.state_dict(),
                'alien_optimizer': self.alien.optimizer.state_dict(),
                'sharing_optimizer': self.sharing.optimizer.state_dict()}

    def load_state_dict(self, state):
        self.policy.load_weights(state['weights'])
        self.baseline[:] = state['baseline']
        self.competition.optimizer.load_state_dict(state['competition_optimizer'])
        self.alien.optimizer.load_state_dict(state['alien_optimizer'])
        self.sharing.optimizer.load_state_dict(state['sharing_optimizer'])


def random_game_pairings(size, games, rng):
    """One independent permutation per game; every model plays once per column."""
    return np.stack([rng.permutation(size) for _ in range(games)], axis=1)


def population_match(policies, size, games_per_pair, rng, collect=None, record=False, opponent_indices=None):
    """a_i vs b_opponent_indices[i], with colors reversed inside each pair.

    Competition uses a fresh permutation (all individuals play). Held-out
    group tests draw each opponent independently with replacement; those
    independent pairs, not the games sharing an opponent, are test units.
    A size-by-games matrix instead selects B independently for every game.
    Each column must be a permutation, preserving equal games and colors per B.
    Default identity matching preserves fixed diagnostic comparisons.
    """
    if games_per_pair < 2 or games_per_pair % 2:
        raise ValueError('Use an even number of games per individual pair')
    n = size*games_per_pair
    pairing = np.arange(size) if opponent_indices is None else np.asarray(opponent_indices,dtype=np.int64)
    per_game = pairing.ndim == 2
    if pairing.shape not in ((size,), (size,games_per_pair)) or (pairing<0).any() or (pairing>=size).any():
        raise ValueError('Opponent indices must select one B individual per A individual')
    if per_game and not np.array_equal(np.sort(pairing,axis=0),
                                      np.broadcast_to(np.arange(size)[:,None],pairing.shape)):
        raise ValueError('Each game column must be a permutation of B individuals')
    if not per_game and collect==1 and len(np.unique(pairing))!=size:
        raise ValueError('Training B requires a permutation so every individual plays equally')
    inverse = np.argsort(pairing,axis=0)
    black, white = initial_boards(n)
    black_agents = np.tile(np.arange(games_per_pair) % 2, size)
    done = np.zeros(n, dtype=bool)
    records = []
    trace = []
    for turn in range(122):
        color = turn % 2
        own, opp = (black, white) if color == 0 else (white, black)
        legal = legal_moves(own, opp)
        done |= (legal == 0) & (legal_moves(opp, own) == 0)
        if done.all():
            break
        players = black_agents if color == 0 else 1-black_agents
        playable = (~done) & (legal != 0)
        moves = np.zeros(n, dtype=np.uint64)
        if record and not done[0] and not playable[0]:
            trace.append({'color': color, 'agent': int(players[0]), 'action': None})
        for agent, policy in enumerate(policies):
            ids = np.flatnonzero(players == agent)
            if not playable[ids].any():
                continue
            tokens = row_tokens(own[ids], opp[ids]).reshape(size, -1, 8)
            masks = bits_to_mask(legal[ids]).reshape(size, -1, 64)
            valid = playable[ids].reshape(size, -1)
            game_ids = (ids % games_per_pair).reshape(size,-1)
            if agent==1 and per_game:
                # Batch inference by actual B model, then return to A-board order.
                columns = np.arange(tokens.shape[1])[None,:]
                selected_pairing = pairing[:,game_ids[0]]
                order = inverse[:,game_ids[0]]
            # Padded inactive slots receive a legal dummy action and zero loss.
            masks[:, :, 0] |= ~valid
            if isinstance(policy, str):
                scores = np.zeros_like(masks, dtype=np.float32)
                if policy == 'position':
                    scores[:] = POSITION
                elif policy != 'random':
                    raise ValueError(policy)
            else:
                if agent==1 and per_game:
                    scores = policy.logits(tokens[order,columns])[selected_pairing,columns]
                else:
                    scores = policy.logits(tokens,individuals=pairing if agent==1 else None)
            if not np.isfinite(scores).all():
                raise FloatingPointError('Non-finite population logits')
            noise = rng.random(scores.shape)*.01 if policy == 'position' else rng.gumbel(size=scores.shape)
            actions = np.argmax(np.where(masks, scores+noise, -np.inf), axis=-1)
            selected = U64(1) << actions.flatten().astype(np.uint64)
            active_ids = ids[valid.flatten()]
            active_moves = selected[valid.flatten()]
            if np.any((active_moves & legal[active_ids]) != active_moves):
                raise AssertionError('Illegal population action')
            moves[active_ids] = active_moves
            if collect == agent:
                # Records always follow the learner's actual model index,
                # including B after shuffled matchmaking.
                values = (tokens,masks,actions,valid,game_ids)
                if agent==1 and per_game:
                    records.append(tuple(value[order,columns].copy() for value in values))
                else:
                    order = inverse if agent==1 else np.arange(size)
                    records.append(tuple(value[order].copy() for value in values))
            if record and ids[0] == 0 and playable[0]:
                trace.append({'color': color, 'agent': agent, 'action': int(actions[0, 0])})
        ids = np.flatnonzero(playable)
        own[ids], opp[ids] = apply_moves(own[ids], opp[ids], moves[ids])
        if np.any(black & white):
            raise AssertionError('Overlapping population stones')
    else:
        raise AssertionError('Population game exceeded turn limit')
    margin = stone_counts(black)-stone_counts(white)
    a_margin = np.where(black_agents == 0, margin, -margin).reshape(size, games_per_pair)
    return {'records': records, 'margins': a_margin, 'games': n,'opponent_indices':pairing.copy(),
            'a_wins': int((a_margin > 0).sum()), 'b_wins': int((a_margin < 0).sum()),
            'draws': int((a_margin == 0).sum()), 'trace': trace,
            'black_stones': int(int(black[0]).bit_count()), 'white_stones': int(int(white[0]).bit_count())}


def learner_rewards(match, learner):
    if learner==0:
        return match['margins']
    pairing = match['opponent_indices']
    inverse = np.argsort(pairing,axis=0)
    return (-np.take_along_axis(match['margins'],inverse,axis=0)
            if pairing.ndim==2 else -match['margins'][inverse])


def competition_update(group, match, learner, chunk=16):
    policy, optimizer = group.policy, group.competition
    tokens, masks, actions, valid, ids = [np.concatenate([r[i] for r in match['records']], axis=1) for i in range(5)]
    rewards = learner_rewards(match,learner)
    colors = ((np.arange(rewards.shape[1]) % 2) != learner).astype(int)
    advantage = rewards - group.baseline[:, colors]
    adv = np.take_along_axis(advantage, ids, axis=1)
    optimizer.zero_grad()
    loss_total = entropy = decisions = 0.
    for start in range(0, tokens.shape[1], chunk):
        sl = slice(start, start+chunk)
        logits = policy(torch.as_tensor(tokens[:, sl], device=policy.device, dtype=torch.long))
        legal = torch.as_tensor(masks[:, sl], device=policy.device, dtype=torch.bool)
        logp = masked_log_probs(logits, legal)
        chosen = logp.gather(2, torch.as_tensor(actions[:, sl, None], device=policy.device)).squeeze(2)
        weight = torch.as_tensor(adv[:, sl]*valid[:, sl], device=policy.device, dtype=logits.dtype)
        loss = -(chosen*weight).sum()/rewards.shape[1]
        loss.backward()
        loss_total += float(loss.detach())/policy.size
        with torch.no_grad():
            per_state = -(logp.exp()*logp.masked_fill(~legal, 0)).sum(2)
            active = torch.as_tensor(valid[:, sl], device=policy.device)
            entropy += float((per_state*active).sum())
            decisions += int(active.sum())
    result = optimizer.step()
    for color in (0, 1):
        group.baseline[:, color] = .9*group.baseline[:, color]+.1*rewards[:, colors == color].mean(1)
    optimizer.zero_grad()
    return result | {'loss': loss_total, 'entropy': entropy/max(1, decisions),
                     'decisions': int(decisions), 'mean_reward': float(rewards.mean())}


def population_diversity(policy, tokens, masks, chunk=16):
    """Mean JS across a fixed common set of positions; units are nats."""
    with torch.no_grad():
        scores = np.zeros(policy.size)
        for start in range(0, len(tokens), chunk):
            t = torch.as_tensor(tokens[start:start+chunk], device=policy.device, dtype=torch.long)
            legal = torch.as_tensor(masks[start:start+chunk], device=policy.device, dtype=torch.bool)
            logp = masked_log_probs(policy(t[None].expand(policy.size, -1, -1)), legal)
            probs = logp.exp()
            mean = probs.mean(0).clamp_min(1e-30)
            kl = (probs*(logp.masked_fill(~legal, 0)-mean.log().masked_fill(~legal, 0))).sum(2)
            scores += kl.sum(1).cpu().numpy()
        scores /= len(tokens)
    return {'value': float(scores.mean()), 'individual_novelty': scores.tolist(), 'unit': 'nats'}


def alien_update(group, tokens, masks, chunk=16):
    """Expected intrinsic reward log pi_i(a|s)-log mean_j pi_j(a|s).

The group mean is fixed per chunk while all individuals take one update.
All legal actions are integrated exactly instead of sampled; this is the
gradient of population JS and cannot count identical random policies diverse.
"""
    policy, optimizer = group.policy, group.alien
    optimizer.zero_grad()
    reward_total = 0.
    for start in range(0, len(tokens), chunk):
        t = torch.as_tensor(tokens[start:start+chunk], device=policy.device, dtype=torch.long)
        legal = torch.as_tensor(masks[start:start+chunk], device=policy.device, dtype=torch.bool)
        logp = masked_log_probs(policy(t[None].expand(policy.size, -1, -1)), legal)
        probs = logp.exp()
        mean = probs.detach().mean(0).clamp_min(1e-30)
        novelty = (probs*(logp.masked_fill(~legal, 0)-mean.log().masked_fill(~legal, 0))).sum(2)
        objective = novelty.sum()/len(tokens)
        (-objective).backward()
        reward_total += float(objective.detach())/policy.size
    result = optimizer.step()
    optimizer.zero_grad()
    return result | {'mean_intrinsic_reward': reward_total}
