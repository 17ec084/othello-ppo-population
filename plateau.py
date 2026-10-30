"""Mutate a fixed plateau-origin snapshot, then restore/switch at t+499.

Mutation happens BEFORE training t+50, t+100, ..., t+450. The deadline
is checked AFTER evaluating t+499, so improvement on that step still wins.
Only the learner and already-open scalar entries may change.
"""
import torch


class PlateauRecovery:
    def __init__(self, seed, sigma=.005, interval=50, timeout_offset=499):
        if sigma <= 0 or interval < 1 or timeout_offset <= interval:
            raise ValueError('Invalid plateau recovery settings')
        self.seed, self.sigma = seed, sigma
        self.interval, self.timeout_offset = interval, timeout_offset
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.t = None
        self.anchor_p = None
        self.phase = self.learner = None
        self.anchor_weights = {}
        self.anchor_baseline = []
        self.mutation_steps = []
        self.total_mutations = 0
        self.events = []

    def capture(self, step, phase, learner, p_value, model, baseline, reason):
        self.t, self.phase, self.learner, self.anchor_p = step, phase, learner, float(p_value)
        self.anchor_weights = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        self.anchor_baseline = list(baseline)
        self.mutation_steps = []
        event = {'step': step, 'type': reason, 't': step, 'phase': phase,
                 'learner': 'AB'[learner], 'p_value': self.anchor_p}
        self.events.append(event)
        return event

    def _restore(self, model, optimizer, baseline):
        model.load_state_dict(self.anchor_weights)
        optimizer.state.clear()
        optimizer.zero_grad(set_to_none=True)
        baseline[:] = self.anchor_baseline

    def before_step(self, step, phase, learner, model, optimizer, baseline, unfreezer):
        if self.t is None or (phase, learner) != (self.phase, self.learner):
            raise ValueError('Capture the current learner before advancing the plateau clock')
        age = step - self.t
        if age <= 0 or age % self.interval or age >= self.timeout_offset or step in self.mutation_steps:
            return None
        # Every restart comes from EXACTLY the same t snapshot, never the last
        # mutant or the weights just before this call. Do not rewind game RNGs
        # or the anytime test: future evaluations must remain fresh.
        self._restore(model, optimizer, baseline)
        changed = 0
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                mask = unfreezer.masks[name]
                n = int(mask.sum())
                if n:
                    noise = torch.randn(n, generator=self.generator, dtype=parameter.dtype) * self.sigma
                    parameter[mask] += noise.to(parameter.device)
                    changed += n
        self.mutation_steps.append(step)
        self.total_mutations += 1
        event = {'step': step, 'type': 'mutation', 't': self.t, 'phase': phase,
                 'learner': 'AB'[learner], 'p_value': self.anchor_p,
                 'attempt': len(self.mutation_steps), 'sigma': self.sigma,
                 'mutated_parameters': changed, 'source': 'original_t_weights'}
        self.events.append(event)
        return event

    def after_step(self, step, phase, learner, p_value, model, optimizer, baseline):
        if (phase, learner) != (self.phase, self.learner):
            raise ValueError('Handle a learner switch by capturing the new phase')
        if p_value < self.anchor_p:
            return self.capture(step, phase, learner, p_value, model, baseline, 'p_improved')
        if step - self.t >= self.timeout_offset:
            self._restore(model, optimizer, baseline)
            event = {'step': step, 'type': 'restore_and_switch', 't': self.t,
                     'phase': phase, 'learner': 'AB'[learner], 'p_value': self.anchor_p,
                     'attempts': len(self.mutation_steps), 'restored': 'original_t_weights'}
            self.events.append(event)
            return event
        return None

    def summary(self, step):
        if self.t is None:
            return {'enabled': True, 'initialized': False}
        next_mutation = self.t + ((max(0, step-self.t)//self.interval)+1)*self.interval
        while next_mutation in self.mutation_steps:
            next_mutation += self.interval
        if next_mutation >= self.t + self.timeout_offset:
            next_mutation = None
        return {'enabled': True, 'initialized': True, 't': self.t,
                'phase': self.phase, 'learner': 'AB'[self.learner], 'anchor_p': self.anchor_p,
                'unchanged_updates': max(0, step-self.t), 'interval': self.interval,
                'timeout_offset': self.timeout_offset, 'deadline_step': self.t+self.timeout_offset,
                'next_mutation_step': next_mutation, 'mutation_steps': list(self.mutation_steps),
                'mutations_this_plateau': len(self.mutation_steps), 'total_mutations': self.total_mutations,
                'sigma': self.sigma, 'scope': 'currently_open_parameters_only',
                'recent_events': self.events[-10:]}

    def state_dict(self):
        return {name: getattr(self, name) for name in (
            'seed','sigma','interval','timeout_offset','t','anchor_p','phase','learner',
            'anchor_weights','anchor_baseline','mutation_steps','total_mutations','events')
        } | {'generator_state': self.generator.get_state()}

    def load_state_dict(self, data):
        for name in ('seed','sigma','interval','timeout_offset'):
            if data[name] != getattr(self, name):
                raise ValueError(f'Saved plateau configuration differs: {name}')
        for name in ('t','anchor_p','phase','learner','anchor_baseline','mutation_steps','total_mutations','events'):
            setattr(self, name, data[name])
        self.anchor_weights = {name: tensor.cpu().clone() for name, tensor in data['anchor_weights'].items()}
        self.generator.set_state(data['generator_state'].cpu())
