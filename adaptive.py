"""Alternate after 50 unchanged updates; clone the opponent after inferiority.

Both one-sided tests use held-out color-balanced pairs, with phase/epoch
conditional nulls. Copying resets both tests; it is not a learner switch.
"""
import numpy as np
from sequential import AlternatingController, AnytimeWinTest


class AdaptiveController(AlternatingController):
    def __init__(self, alpha=.05, patience=50):
        super().__init__(alpha)
        if patience < 1:
            raise ValueError('Patience must be positive')
        self.patience = patience
        self.loss_test = AnytimeWinTest(alpha)
        self.t = 0
        self.anchor_p = 1.
        self.epoch = 1
        self.copies = []

    def reset_evidence(self, step):
        self.test = AnytimeWinTest(self.alpha)
        self.loss_test = AnytimeWinTest(self.alpha)
        self.t, self.anchor_p = step, 1.
        self.epoch += 1

    def switch(self, step, unfreezer, reason, details=None):
        loss = self.loss_test.summary()
        event = super().switch(step, unfreezer, reason, details)
        event.update(type='learner_switch', loss_test=loss,
                     timing='before_update' if reason == 'plateau_50' else 'after_update')
        self.reset_evidence(step)
        return event

    def before_step(self, step, unfreezer):
        if step - self.t >= self.patience:
            return self.switch(step, unfreezer, 'plateau_50',
                               {'t': self.t, 'patience': self.patience,
                                'weights': 'keep_current_weights_no_rollback'})
        return None

    def observe(self, step, margins, unfreezer):
        margins = np.asarray(margins)
        if len(margins) % 2:
            raise ValueError('Color-balanced pairs required')
        for first, second in margins.reshape(-1, 2):
            direction = 1 if self.learner == 0 else -1
            win = self.test.add_pair(direction*first, direction*second)
            loss = self.loss_test.add_pair(-direction*first, -direction*second)
            if win:
                return self.switch(step, unfreezer, 'p_threshold')
            if loss:
                event = {'step': step, 'type': 'opponent_copy', 'timing': 'after_update',
                         'phase': self.phase, 'epoch': self.epoch, 'learner': 'AB'[self.learner],
                         'source': 'AB'[1-self.learner], 'test': self.test.summary(),
                         'loss_test': self.loss_test.summary(), 'scope': 'all_model_weights',
                         'p_value': self.loss_test.p_value, 't': self.t}
                self.copies.append(event)
                self.reset_evidence(step)
                return event
        if self.test.p_value < self.anchor_p:
            self.t, self.anchor_p = step, self.test.p_value
        return None

    def summary(self):
        return super().summary() | {'loss_test': self.loss_test.summary(),
                                    'epoch': self.epoch, 'copies': len(self.copies)}

    def plateau_summary(self, step):
        return {'enabled': True, 'initialized': True, 'mode': 'switch_without_mutation',
                't': self.t, 'anchor_p': self.anchor_p, 'learner': 'AB'[self.learner],
                'unchanged_updates': max(0, step-self.t), 'patience': self.patience,
                'deadline_step': self.t+self.patience, 'mutations_enabled': False}

    def state_dict(self):
        return super().state_dict() | {'loss_test': self.loss_test.state_dict(),
                'patience': self.patience, 't': self.t, 'anchor_p': self.anchor_p,
                'epoch': self.epoch, 'copies': self.copies}

    def load_state_dict(self, state):
        if state['patience'] != self.patience:
            raise ValueError('Saved patience differs')
        super().load_state_dict(state)
        self.loss_test.load_state_dict(state['loss_test'])
        for name in ('t', 'anchor_p', 'epoch', 'copies'):
            setattr(self, name, state[name])


def copy_opponent(models, optimizers, baselines, learner):
    """Exact full copy, including frozen entries, without changing the source."""
    models[learner].load_state_dict(models[1-learner].state_dict())
    optimizers[learner].state.clear()
    optimizers[learner].zero_grad(set_to_none=True)
    baselines[learner][:] = [0., 0.]
