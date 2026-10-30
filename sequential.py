"""One-sided bounded betting mixture and alternating learner protocol.

For each color-balanced two-game unit, X=(sign(m1)+sign(m2))/2 in [-1,1]
from the learner's perspective. Under the conditional null E[X|past]<=0,
each product of (1+lambda*X), 0<=lambda<1, is a test supermartingale.
A prespecified mixture is also one; p=1/max(1,max E) is anytime valid
by Ville's inequality. The learner may change predictably BETWEEN batches.
This is a phase-level conditional null, not a fixed-snapshot win probability.

Reference: https://arxiv.org/abs/2210.01948
"""
import math
import numpy as np


class AnytimeWinTest:
    BETS = np.array([.01, .02, .04, .08, .16, .32, .5, .75, .95])

    def __init__(self, alpha=.05):
        if not 0 < alpha < 1:
            raise ValueError('alpha must be between 0 and 1')
        self.alpha = alpha
        self.log_wealth = np.zeros(len(self.BETS))
        self.max_log_e = 0.
        self.pairs = 0
        self.total_payoff = 0.
        self.wins = self.draws = self.losses = 0

    def add_pair(self, first_margin, second_margin):
        signs = np.sign([first_margin, second_margin])
        payoff = float(signs.mean())
        if not np.isfinite(payoff):
            raise ValueError('Finite margins are required')
        self.wins += int((signs > 0).sum())
        self.draws += int((signs == 0).sum())
        self.losses += int((signs < 0).sum())
        return self.add_payoff(payoff)

    def add_payoff(self, payoff):
        """One bounded [-1,1] observation; no sign conversion."""
        if not np.isfinite(payoff) or not -1 <= payoff <= 1:
            raise ValueError('Payoff must be finite and in [-1,1]')
        self.log_wealth += np.log1p(self.BETS * payoff)
        maximum = float(self.log_wealth.max())
        log_e = maximum + math.log(float(np.exp(self.log_wealth - maximum).mean()))
        self.max_log_e = max(self.max_log_e, log_e)
        self.pairs += 1
        self.total_payoff += payoff
        return self.p_value <= self.alpha

    @property
    def p_value(self):
        return math.exp(-self.max_log_e)

    def summary(self):
        return {'p_value': self.p_value, 'alpha': self.alpha, 'pairs': self.pairs,
                'games': self.pairs * 2, 'wins': self.wins, 'draws': self.draws,
                'losses': self.losses, 'score': (self.wins + .5*self.draws)/(2*self.pairs) if self.pairs else None,
                'log_max_e': self.max_log_e, 'passed': self.p_value <= self.alpha}

    def state_dict(self):
        return {**self.__dict__, 'log_wealth': self.log_wealth.tolist()}

    def load_state_dict(self, data):
        if data['alpha'] != self.alpha:
            raise ValueError('Saved significance threshold differs')
        self.__dict__.update(data)
        self.log_wealth = np.asarray(data['log_wealth'], dtype=np.float64)


class AlternatingController:
    def __init__(self, alpha=.05):
        self.alpha = alpha
        self.learner = 0
        self.phase = 1
        self.phase_start_step = 0
        self.events = []
        self.test = AnytimeWinTest(alpha)

    def observe(self, step, margins, unfreezer):
        """Margins in policy-0/A perspective, alternating A-black/A-white."""
        margins = np.asarray(margins)
        if len(margins) % 2:
            raise ValueError('Color-balanced pairs required')
        for first, second in margins.reshape(-1, 2):
            direction = 1 if self.learner == 0 else -1
            if self.test.add_pair(direction*first, direction*second):
                return self.switch(step, unfreezer, reason='p_threshold')
        return None

    def switch(self, step, unfreezer, reason, details=None):
        old = self.learner
        event = {'step': step, 'phase': self.phase, 'from': 'AB'[old],
                 'to': 'AB'[1-old], 'test': self.test.summary(),
                 'percent_before': unfreezer.summary()['percent'], 'reason': reason,
                 'significance_confirmed': reason == 'p_threshold'}
        if details is not None:
            event['recovery'] = details
        event['released'] = unfreezer.release_increment(step) if old == 1 else False
        event['percent_after'] = unfreezer.summary()['percent']
        self.events.append(event)
        self.learner = 1-old
        self.phase += 1
        self.phase_start_step = step
        self.test = AnytimeWinTest(self.alpha)
        return event

    def summary(self):
        return {'learner': 'AB'[self.learner], 'frozen_opponent': 'AB'[1-self.learner],
                'phase': self.phase, 'phase_start_step': self.phase_start_step,
                'test': self.test.summary(), 'switches': len(self.events), 'events': self.events}

    def state_dict(self):
        return {'alpha': self.alpha, 'learner': self.learner, 'phase': self.phase,
                'phase_start_step': self.phase_start_step, 'events': self.events,
                'test': self.test.state_dict()}

    def load_state_dict(self, state):
        if state['alpha'] != self.alpha:
            raise ValueError('Saved significance threshold differs')
        self.learner, self.phase = state['learner'], state['phase']
        self.phase_start_step, self.events = state['phase_start_step'], state['events']
        self.test.load_state_dict(state['test'])
