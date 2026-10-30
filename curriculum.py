"""Nested scalar-parameter masks and rolling leader-change release schedule."""
from __future__ import annotations

import torch


class ProgressiveUnfreezer:
    def __init__(self, model, seed, initial_percent=1, window=15, reversals=3, min_hold_updates=50, granularity=1):
        if (granularity < 1 or not 1 / granularity <= initial_percent <= 100 or window < 3 or not 1 <= reversals < window
                or min_hold_updates < 0):
            raise ValueError('Invalid unfreezing schedule')
        self.seed = seed
        self.granularity = granularity
        self.percent = int(round(initial_percent * granularity))
        self.window = window
        self.required_reversals = reversals
        self.min_hold_updates = min_hold_updates
        self.stage_start_step = 0
        self.latest_step = 0
        self.shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}
        self.sizes = {name: p.numel() for name, p in model.named_parameters()}
        self.total = sum(self.sizes.values())
        self.orders = {}
        for index, (name, size) in enumerate(self.sizes.items()):
            generator = torch.Generator().manual_seed(seed + 104729 * index)
            self.orders[name] = torch.randperm(size, generator=generator)
        # Allocate each additional percent across all tensors, proportionally
        # to remaining capacity. Exact global counts; nested sets at every stage.
        self.counts_by_percent = [{name: 0 for name in self.sizes}]
        for percent in range(1, 100 * granularity + 1):
            previous = self.counts_by_percent[-1]
            target = self.total * percent // (100 * granularity)
            budget = target - sum(previous.values())
            remaining = {name: size - previous[name] for name, size in self.sizes.items()}
            capacity = sum(remaining.values())
            adds = {name: budget * n // capacity for name, n in remaining.items()}
            remainder_order = sorted(self.sizes, key=lambda name: budget * remaining[name] % capacity,
                                     reverse=True)
            for name in remainder_order[:budget - sum(adds.values())]:
                adds[name] += 1
            self.counts_by_percent.append({name: previous[name] + adds[name] for name in self.sizes})
        self.history = []
        self.events = []
        self.latest_reversals = 0
        self.device = next(model.parameters()).device
        self._build_masks()

    def _build_masks(self):
        self.masks = {}
        counts = self.counts_by_percent[self.percent]
        for name, size in self.sizes.items():
            mask = torch.zeros(size, dtype=torch.bool)
            mask[self.orders[name][:counts[name]]] = True
            self.masks[name] = mask.reshape(self.shapes[name]).to(self.device)

    def observe(self, step, a_wins, b_wins, draws=0):
        if step <= self.latest_step:
            raise ValueError('Updates must be observed in increasing order')
        self.latest_step = step
        leader = 'A' if a_wins > b_wins else 'B' if b_wins > a_wins else None
        self.history.append({'step': step, 'leader': leader, 'a_wins': a_wins,
                             'b_wins': b_wins, 'draws': draws})
        self.history = self.history[-self.window:]
        leaders = [row['leader'] for row in self.history if row['leader'] is not None]
        self.latest_reversals = sum(a != b for a, b in zip(leaders, leaders[1:]))
        if (self.percent < 100 * self.granularity and len(self.history) == self.window
                and step - self.stage_start_step >= self.min_hold_updates
                and self.latest_reversals >= self.required_reversals):
            old = self.percent
            self.percent += 1
            self.events.append({'step': step, 'from_percent': old, 'to_percent': self.percent,
                                'reversals': self.latest_reversals, 'window': list(self.history),
                                'stage_updates': step - self.stage_start_step,
                                'min_hold_updates': self.min_hold_updates})
            self.stage_start_step = step
            self.history = []
            self.latest_reversals = 0
            self._build_masks()
            return True
        return False

    def summary(self):
        active = sum(self.counts_by_percent[self.percent].values())
        return {'percent': self.percent / self.granularity, 'active_parameters': active, 'total_parameters': self.total,
                'frozen_parameters': self.total - active, 'window': self.window,
                'required_reversals': self.required_reversals, 'observations': len(self.history),
                'reversals': self.latest_reversals, 'history': self.history,
                'events': self.events, 'per_tensor_active': self.counts_by_percent[self.percent],
                'last_release_step': self.events[-1]['step'] if self.events else None,
                'min_hold_updates': self.min_hold_updates, 'stage_start_step': self.stage_start_step,
                'latest_step': self.latest_step, 'stage_updates': self.latest_step - self.stage_start_step,
                'hold_remaining': max(0, self.min_hold_updates - (self.latest_step - self.stage_start_step)),
                'earliest_release_step': self.stage_start_step + max(self.min_hold_updates, self.window)}

    def state_dict(self):
        return {'seed': self.seed, 'percent': self.percent, 'window': self.window,
                'required_reversals': self.required_reversals, 'history': self.history,
                'events': self.events, 'orders': self.orders, 'shapes': self.shapes,
                'min_hold_updates': self.min_hold_updates, 'stage_start_step': self.stage_start_step,
                'latest_step': self.latest_step, 'granularity': self.granularity}

    def load_state_dict(self, state, *, override_min_hold=False):
        if (state['seed'] != self.seed or state['shapes'] != self.shapes
                or state.get('granularity', 1) != self.granularity
                or state['window'] != self.window or state['required_reversals'] != self.required_reversals):
            raise ValueError('Unfreezing schedule differs from saved checkpoint')
        if (state.get('min_hold_updates', self.min_hold_updates) != self.min_hold_updates
                and not override_min_hold):
            raise ValueError('Minimum hold differs from saved checkpoint')
        self.percent, self.history, self.events = state['percent'], state['history'], state['events']
        self.stage_start_step = state.get('stage_start_step', self.events[-1]['step'] if self.events else 0)
        self.latest_step = state.get('latest_step', self.history[-1]['step'] if self.history else self.stage_start_step)
        self.orders = {name: order.cpu() for name, order in state['orders'].items()}
        leaders = [row['leader'] for row in self.history if row['leader'] is not None]
        self.latest_reversals = sum(a != b for a, b in zip(leaders, leaders[1:]))
        self._build_masks()

    def release_increment(self, step, reason='B_to_A'):
        """Release one grid unit (0.1 percentage points when granularity=10)."""
        if self.percent >= 100 * self.granularity:
            return False
        old = self.percent / self.granularity
        self.percent += 1
        self.events.append({'step': step, 'from_percent': old,
                            'to_percent': self.percent / self.granularity, 'reason': reason})
        self._build_masks()
        return True

    def mask_gradients(self, model, optimizer):
        for name, parameter in model.named_parameters():
            mask = self.masks[name]
            if parameter.grad is not None:
                parameter.grad.mul_(mask)
            # Frozen entries must not move due to old optimizer momentum.
            for value in optimizer.state.get(parameter, {}).values():
                if isinstance(value, torch.Tensor) and value.shape == parameter.shape:
                    value.mul_(mask)

    def restore_frozen(self, model, previous_parameters):
        with torch.no_grad():
            for (name, parameter), previous in zip(model.named_parameters(), previous_parameters):
                parameter.copy_(torch.where(self.masks[name], parameter, previous))
