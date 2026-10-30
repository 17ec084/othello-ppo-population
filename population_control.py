"""Group-average win signs, accumulated once per held-out population batch."""
import numpy as np
import math
from sequential import AnytimeWinTest
from adaptive import AdaptiveController


class GroupWinTest:
    def __init__(self, alpha=.05):
        self.evidence = AnytimeWinTest(alpha)
        self.alpha = alpha
        self.games = self.wins = self.draws = self.losses = 0
        self.margin_sum = 0.
        self.last_mean = None

    @property
    def p_value(self):
        return self.evidence.p_value

    def add_group(self, margins):
        margins = np.asarray(margins, dtype=float)
        if margins.ndim != 2 or margins.shape[1] % 2 or not np.isfinite(margins).all() or (abs(margins)>64).any():
            raise ValueError('Finite, color-balanced population stone margins required')
        self.games += margins.size
        self.wins += int((margins>0).sum())
        self.draws += int((margins==0).sum())
        self.losses += int((margins<0).sum())
        self.margin_sum += float(margins.sum())
        self.last_mean = float(margins.mean())
        # Independent opponent draws and action RNG for each A individual.
        # Colors inside one pair share the sampled opponent, so average
        # them FIRST; do not treat them as independent test observations.
        # Hoeffding's
        # lemma: E exp(lambda sum X - n lambda^2/2) <= 1 whenever
        # sum E[X|past] <= 0, X=mean_color sign(margin) in [-1,1]. Individual pairs
        # may have different nonzero means; only the group mean null is used.
        bets = self.evidence.BETS
        pair_payoffs = np.sign(margins).mean(1)
        self.evidence.log_wealth += bets*float(pair_payoffs.sum()) - .5*len(pair_payoffs)*bets**2
        maximum = float(self.evidence.log_wealth.max())
        log_e = maximum+math.log(float(np.exp(self.evidence.log_wealth-maximum).mean()))
        self.evidence.max_log_e = max(self.evidence.max_log_e, log_e)
        self.evidence.pairs += 1
        self.evidence.total_payoff += float(np.sign(margins).mean())
        return self.p_value <= self.alpha

    def summary(self):
        return {'p_value': self.p_value, 'alpha': self.alpha, 'games': self.games,
                'wins': self.wins, 'draws': self.draws, 'losses': self.losses,
                'pairs': self.games//2, 'rounds': self.evidence.pairs,
                'score': (self.wins+.5*self.draws)/self.games if self.games else None,
                'mean_margin': self.margin_sum/self.games if self.games else None,
                'last_mean_margin': self.last_mean, 'unit': 'group_mean_win_sign', 'mean_win_sign': (self.wins-self.losses)/self.games if self.games else None,
                'method': 'independent-random-opponent-pairs-Hoeffding-mixture',
                'log_max_e': self.evidence.max_log_e, 'passed': self.p_value<=self.alpha}

    def state_dict(self):
        return self.__dict__ | {'evidence': self.evidence.state_dict()}

    def load_state_dict(self, state):
        if state['alpha'] != self.alpha:
            raise ValueError('Saved group alpha differs')
        for key, value in state.items():
            if key != 'evidence':
                setattr(self, key, value)
        self.evidence.load_state_dict(state['evidence'])


class PopulationController(AdaptiveController):
    def __init__(self, alpha=.05, patience=50):
        super().__init__(alpha, patience)
        self.test, self.loss_test = GroupWinTest(alpha), GroupWinTest(alpha)

    def reset_evidence(self, step):
        self.test, self.loss_test = GroupWinTest(self.alpha), GroupWinTest(self.alpha)
        self.t, self.anchor_p = step, 1.
        self.epoch += 1

    def observe(self, step, margins, unfreezer):
        direction = 1 if self.learner == 0 else -1
        win = self.test.add_group(direction*np.asarray(margins))
        loss = self.loss_test.add_group(-direction*np.asarray(margins))
        if win:
            return self.switch(step, unfreezer, 'p_threshold')
        if loss:
            event = {'step': step, 'type': 'opponent_copy', 'timing': 'after_update',
                     'phase': self.phase, 'learner': 'AB'[self.learner], 'source': 'AB'[1-self.learner],
                     'test': self.test.summary(), 'loss_test': self.loss_test.summary(),
                     'scope': 'all_group_weights_index_matched', 'p_value': self.loss_test.p_value,
                     't': self.t}
            self.copies.append(event)
            self.reset_evidence(step)
            return event
        if self.test.p_value < self.anchor_p:
            self.t, self.anchor_p = step, self.test.p_value
        return None

    def summary(self):
        return super().summary() | {'mode': 'population_win_sign'}
