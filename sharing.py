"""Within-group policy distillation on boards from the competition games.

Teacher ranking uses separate games and never touches the sequential-test RNG.
Targets are detached snapshots made before any student is updated. A student's
own match result is not reused as another student's policy-gradient reward.
"""
import math
import numpy as np
import torch

from model import masked_log_probs
from population import population_match


def evaluate_teachers(group, games_per_opponent, rng, heartbeat=None):
    """Observed draw-half score vs random and position AIs, colors balanced."""
    scores, results = [], {}
    for opponent in ('random', 'position'):
        if heartbeat:
            heartbeat()
        match = population_match([group.policy, opponent], group.policy.size,
                                 games_per_opponent, rng)
        margins = match['margins']
        scores.append(((np.sign(margins)+1)/2).sum(1))
        results[opponent] = {'wins': int((margins>0).sum()),
                             'draws': int((margins==0).sum()),
                             'games': int(margins.size)}
    return {'scores': (sum(scores)/(2*games_per_opponent)).tolist(),
            'games_per_individual': 2*games_per_opponent,
            'games': 2*games_per_opponent*group.policy.size,
            'opponents': results}


@torch.no_grad()
def sharing_plan(group, records, scores, rng, positions=32, fraction=.2, teachers=2, chunk=16):
    """Different boards and teachers per student; only higher scored top peers.

records have already been reordered to the actual learner's individual IDs.
All valid boards are pooled, including the other individuals' experience.
"""
    policy, size = group.policy, group.policy.size
    score = np.asarray(scores, dtype=float)
    if score.shape != (size,) or not np.isfinite(score).all():
        raise ValueError('Teacher scores must match the population')
    bank_t = np.concatenate([t[v] for t,m,a,v,ids in records])
    bank_m = np.concatenate([m[v] for t,m,a,v,ids in records])
    if not len(bank_t):
        raise ValueError('No valid competition boards to share')
    # Break tied ranks independently; a strictly higher score is still required.
    order = rng.permutation(size)
    pool = order[np.argsort(-score[order], kind='stable')][:max(1, math.ceil(size*fraction))]
    chosen = np.empty((size, teachers), dtype=np.int64)
    enabled = np.zeros(size, dtype=bool)
    for i in range(size):
        candidates = pool[(pool != i) & (score[pool] > score[i])]
        enabled[i] = len(candidates)>0
        chosen[i] = rng.choice(candidates, teachers, replace=len(candidates)<teachers) if enabled[i] else i
    sample = rng.integers(len(bank_t), size=(size, positions))
    tokens, masks = bank_t[sample], bank_m[sample]
    targets = np.zeros((size, positions, 64), dtype=np.float32)
    # Snapshot all teacher outputs before training any student.
    for start in range(0, positions, chunk):
        t = torch.as_tensor(tokens[:,start:start+chunk], device=policy.device, dtype=torch.long)
        legal = torch.as_tensor(masks[:,start:start+chunk], device=policy.device, dtype=torch.bool)
        target = torch.zeros((*t.shape[:2],64), device=policy.device)
        for k in range(teachers):
            target += masked_log_probs(policy(t, individuals=chosen[:,k]), legal).exp()/teachers
        targets[:,start:start+chunk] = target.cpu().numpy()
    return {'tokens':tokens, 'masks':masks, 'targets':targets, 'enabled':enabled,
            'teacher_ids':chosen, 'teacher_pool':pool.tolist(), 'scores':score.tolist(),
            'available_boards':len(bank_t)}


def sharing_update(group, plan, chunk=16):
    """One small Adam step minimizing KL(frozen teacher mixture || student)."""
    policy, optimizer = group.policy, group.sharing
    enabled = torch.as_tensor(plan['enabled'], device=policy.device, dtype=torch.bool)
    positions = plan['tokens'].shape[1]
    summary = {'students':int(enabled.sum()), 'positions_per_student':positions,
               'teacher_pool':plan['teacher_pool'], 'teacher_ids':plan['teacher_ids'].tolist(),
               'available_boards':plan['available_boards'], 'learning_rate':optimizer.lr}
    if not bool(enabled.any()):
        return summary | {'kl_before':0., 'kl_after':0., 'parameter_delta_l2':0., 'gradient_norm':0.}
    optimizer.zero_grad()
    def loss_chunks(backward):
        total = 0.
        for start in range(0, positions, chunk):
            t = torch.as_tensor(plan['tokens'][:,start:start+chunk], device=policy.device, dtype=torch.long)
            legal = torch.as_tensor(plan['masks'][:,start:start+chunk], device=policy.device, dtype=torch.bool)
            target = torch.as_tensor(plan['targets'][:,start:start+chunk], device=policy.device)
            logp = masked_log_probs(policy(t),legal).masked_fill(~legal,0)
            kl = (target*(target.clamp_min(1e-30).log()-logp)).sum(2)
            loss = (kl*enabled[:,None]).sum()/positions
            if backward:
                loss.backward()
            total += float(loss.detach())/int(enabled.sum())
        return total
    before = loss_chunks(True)
    # Zero gradients alone do not prevent Adam momentum moving excluded peers.
    # Preserve their exact values and row-shaped optimizer history.
    excluded = ~enabled
    frozen_rows = {name:p.detach()[excluded].clone() for name,p in policy.active.items()}
    history = {name:{k:v[excluded].clone() for k,v in optimizer.optimizer.state.get(p,{}).items()
                     if isinstance(v,torch.Tensor) and v.shape==p.shape}
               for name,p in policy.active.items()}
    metrics = optimizer.step()
    with torch.no_grad():
        for name,p in policy.active.items():
            p[excluded] = frozen_rows[name]
            for key,value in optimizer.optimizer.state[p].items():
                if isinstance(value,torch.Tensor) and value.shape==p.shape:
                    value[excluded] = history[name].get(key,torch.zeros_like(value[excluded]))
        after = loss_chunks(False)
    optimizer.zero_grad()
    return summary | metrics | {'kl_before':before,'kl_after':after}
