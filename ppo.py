"""Clipped PPO and a state-value head; only the selected learner is updated."""
import hashlib
import time
import numpy as np
import torch
from torch import nn
from train import masked_log_probs
from entropy import normalized_entropy


def value_head(width, device):
    # fork_rng keeps adding the critic from changing actor initialization/RNG.
    with torch.random.fork_rng(devices=[]):
        head=nn.Linear(width,1)
    nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
    return head.to(device)


def trajectory_arrays(records):
    return [np.concatenate([r[i] for r in records]) for i in range(4)]


def rollout_digest(records):
    h=hashlib.sha256()
    for arrays in records:
        for a in arrays:h.update(np.asarray(a).tobytes())
    return h.hexdigest()


def gae(values, game_ids, terminal_rewards, gamma=1., lam=.95):
    """One transition per learner decision (includes any intervening opponent turns).

    A pass has no policy action. Each game's last learner decision receives the
    terminal reward, including games that end after an opponent move/pass.
    """
    values=np.asarray(values,dtype=np.float64);ids=np.asarray(game_ids)
    if len(values)!=len(ids) or not 0<=gamma<=1 or not 0<=lam<=1:raise ValueError('Invalid GAE input')
    adv=np.zeros_like(values)
    for game,reward in enumerate(terminal_rewards):
        positions=np.flatnonzero(ids==game);next_value=0.;running=0.
        for j,pos in enumerate(positions[::-1]):
            delta=(float(reward) if j==0 else 0.)+gamma*next_value-values[pos]
            running=delta+gamma*lam*running;adv[pos]=running;next_value=values[pos]
    return adv.astype(np.float32),(adv+values).astype(np.float32)


def clipped_policy_loss(logp, old_logp, advantage, clip=.2):
    ratio=(logp-old_logp).exp()
    return -torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage)


def update_ppo(model, critic, optimizer, records, rewards, rng, unfreezer=None,
               epochs=4, minibatch=256, chunk_size=16, clip=.2, lam=.95,
               value_coef=.5, target_kl=.02, heartbeat=None):
    tokens,masks,actions,ids=trajectory_arrays(records)
    device=next(model.parameters()).device;n=len(tokens)
    old_logp=np.empty(n,np.float32);old_values=np.empty(n,np.float32)
    entropy_total=0.;choices_total=0
    with torch.no_grad():
        for start in range(0,n,chunk_size):
            end=min(n,start+chunk_size)
            x=model.features(torch.as_tensor(tokens[start:end],device=device,dtype=torch.long))
            legal=torch.as_tensor(masks[start:end],device=device,dtype=torch.bool)
            lp=masked_log_probs(model.head(x),legal)
            a=torch.as_tensor(actions[start:end,None],device=device)
            old_logp[start:end]=lp.gather(1,a).squeeze(1).cpu().numpy()
            old_values[start:end]=critic(x).squeeze(-1).cpu().numpy()
            _,ent=normalized_entropy(lp,legal);choices=legal.sum(-1)>1
            entropy_total+=float(ent[choices].sum());choices_total+=int(choices.sum())
    advantages,targets=gae(old_values,ids,rewards,lam=lam)
    advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
    before=[p.detach().clone() for p in model.parameters()]
    sums={'policy_loss':0.,'value_loss':0.,'approx_kl':0.,'clip_fraction':0.}
    samples=updates=0;stopped=False;last_norm=0.;epoch_done=0
    heartbeat_at=time.monotonic()
    for epoch in range(epochs):
        if heartbeat:heartbeat()
        permutation=rng.permutation(n)
        for start in range(0,n,minibatch):
            if heartbeat and time.monotonic()-heartbeat_at>=5:
                heartbeat();heartbeat_at=time.monotonic()
            batch=permutation[start:start+minibatch];optimizer.zero_grad(set_to_none=True)
            batch_kl=0.
            # Microbatches add up to one minibatch update; they do not create
            # extra optimizer steps or change the minibatch normalization.
            for offset in range(0,len(batch),chunk_size):
                ix=batch[offset:offset+chunk_size]
                x=model.features(torch.as_tensor(tokens[ix],device=device,dtype=torch.long))
                lp=masked_log_probs(model.head(x),torch.as_tensor(masks[ix],device=device,dtype=torch.bool))
                chosen=lp.gather(1,torch.as_tensor(actions[ix,None],device=device)).squeeze(1)
                old=torch.as_tensor(old_logp[ix],device=device)
                adv=torch.as_tensor(advantages[ix],device=device)
                policy=clipped_policy_loss(chosen,old,adv,clip)
                value=(critic(x).squeeze(-1)-torch.as_tensor(targets[ix],device=device)).square()
                loss=(policy.sum()+value_coef*value.sum())/len(batch)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite PPO loss')
                loss.backward()
                with torch.no_grad():
                    ratio=(chosen-old).exp();kl=((ratio-1)-(chosen-old)).sum()
                    batch_kl+=float(kl)
                    sums['policy_loss']+=float(policy.sum());sums['value_loss']+=float(value.sum())
                    sums['approx_kl']+=float(kl);sums['clip_fraction']+=float(((ratio-1).abs()>clip).sum())
                    samples+=len(ix)
            if batch_kl/len(batch)>1.5*target_kl:
                optimizer.zero_grad(set_to_none=True);stopped=True;break
            if unfreezer is not None:unfreezer.mask_gradients(model,optimizer)
            last_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
            torch.nn.utils.clip_grad_norm_(critic.parameters(),1.,error_if_nonfinite=True)
            previous=[p.detach().clone() for p in model.parameters()] if unfreezer is not None else None
            optimizer.step()
            if unfreezer is not None:unfreezer.restore_frozen(model,previous)
            updates+=1
        epoch_done=epoch+1
        # Measure KL on the whole rollout AFTER the epoch. Masked actions and
        # zero-probability illegal moves never enter the ratio.
        kl_total=0.
        with torch.no_grad():
            for start in range(0,n,chunk_size):
                end=min(n,start+chunk_size)
                lp=masked_log_probs(model(torch.as_tensor(tokens[start:end],device=device,dtype=torch.long)),
                    torch.as_tensor(masks[start:end],device=device,dtype=torch.bool))
                chosen=lp.gather(1,torch.as_tensor(actions[start:end,None],device=device)).squeeze(1)
                d=chosen-torch.as_tensor(old_logp[start:end],device=device)
                kl_total+=float((d.exp()-1-d).sum())
        if stopped or kl_total/n>target_kl:stopped=True;break
    with torch.no_grad():
        delta=float(sum((p-old).square().sum() for p,old in zip(model.parameters(),before)).sqrt())
        if not all(bool(torch.isfinite(p).all()) for p in list(model.parameters())+list(critic.parameters())):
            raise FloatingPointError('Nonfinite PPO weights')
    return {**{k:v/max(1,samples) for k,v in sums.items()},'algorithm':'ppo','optimizer_steps':updates,
        'epochs_completed':epoch_done,'kl_early_stop':stopped,'final_approx_kl':kl_total/n,
        'gradient_norm':last_norm,'parameter_delta_l2':delta,'decisions':n,
        'mean_reward':float(np.mean(rewards)),'value_prediction_mean':float(old_values.mean()),
        'value_target_mean':float(targets.mean()),'normalized_entropy':entropy_total/max(1,choices_total),
        'entropy_coefficient':0.,'mean_entropy_bonus':0.}
