"""Legal-action entropy rewards and a fixed diagnostic position bank."""
import numpy as np
import torch
from model import masked_log_probs


def normalized_entropy(log_probs, legal):
    """0..1; uniform legal policy is 1, forced moves receive no bonus."""
    entropy = -(log_probs.exp()*log_probs.masked_fill(~legal,0)).sum(-1)
    counts = legal.sum(-1)
    normalized = torch.where(counts>1,entropy/counts.float().log().clamp_min(1e-12),0.)
    return entropy, normalized


def future_entropy_returns(values, ids, games):
    """Exclude the current state's entropy (independent of its sampled action).

The score-function term accounts for how earlier actions reach later states;
the current state's entropy also has an explicit differentiable term.
"""
    future = np.zeros(len(values),dtype=np.float64)
    totals = np.zeros(games,dtype=np.float64)
    for index in range(len(values)-1,-1,-1):
        game = ids[index]
        future[index] = totals[game]
        totals[game] += values[index]
    return future,totals


@torch.no_grad()
def rollout_entropy_returns(model,tokens,masks,ids,games,chunk=256):
    device = next(model.parameters()).device
    values = np.empty(len(tokens))
    for start in range(0,len(tokens),chunk):
        t=torch.as_tensor(tokens[start:start+chunk],device=device,dtype=torch.long)
        legal=torch.as_tensor(masks[start:start+chunk],device=device,dtype=torch.bool)
        _,h=normalized_entropy(masked_log_probs(model(t),legal),legal)
        values[start:start+chunk]=h.cpu().numpy()
    return future_entropy_returns(values,ids,games)


def probe_positions(seed,count):
    from train import play_batch
    match=play_batch(['random','random'],max(8,2*((count+119)//120)),
                     np.random.default_rng(seed),'cpu',collect=True)
    candidates,seen=[],set()
    for side in match['records']:
        for tokens,masks,_,_ in side:
            for t,m in zip(tokens,masks):
                key=t.tobytes()
                if m.sum()>1 and key not in seen:
                    seen.add(key);candidates.append((t,m))
    if len(candidates)<count:
        raise ValueError('Not enough distinct entropy probe positions')
    ids=np.random.default_rng(seed+1).permutation(len(candidates))[:count]
    return np.stack([candidates[i][0] for i in ids]),np.stack([candidates[i][1] for i in ids])


@torch.no_grad()
def measure_entropy(model,tokens,masks):
    device=next(model.parameters()).device
    t=torch.as_tensor(tokens,device=device,dtype=torch.long)
    legal=torch.as_tensor(masks,device=device,dtype=torch.bool)
    logp=masked_log_probs(model(t),legal)
    h,h_norm=normalized_entropy(logp,legal)
    return {'normalized_entropy':float(h_norm.mean()),'entropy_nats':float(h.mean()),
            'mean_max_probability':float(logp.exp().max(-1).values.mean()),
            'mean_uniform_probability':float((1/legal.sum(-1).float()).mean()),'positions':len(tokens)}
