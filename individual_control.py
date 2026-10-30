"""One learner versus all other individuals; keep weights on reselection."""
import copy
from fractions import Fraction
import numpy as np


class IndividualController:
    def __init__(self,size,patience,rng):
        if size<2 or patience<1:raise ValueError('At least two models and positive patience required')
        self.size,self.patience=size,patience
        self.learner=int(rng.integers(size))
        self.initial_learner=self.learner
        self.phase=1
        self.phase_start_step=0
        self.best=self.current=None
        self.events=[]
        self.records=0

    def observe(self,step,margins):
        own=np.asarray(margins)
        if not own.size or not np.isfinite(own).all():raise ValueError('Finite results required')
        w,d,l=(int((own>0).sum()),int((own==0).sum()),int((own<0).sum()))
        self.current={'step':step,'wins':w,'draws':d,'losses':l,'games':int(own.size),
                      'score_units':2*w+d,'score':(w+.5*d)/own.size,'win_rate':w/own.size}
        improved=self.best is None or self.current['score_units']*self.best['games']>self.best['score_units']*own.size
        if improved:self.best=copy.deepcopy(self.current);self.records+=1
        return improved

    def expired(self,step):return self.best is not None and step-self.best['step']>=self.patience

    def reselect(self,step,mask,rng):
        if not self.expired(step):raise ValueError('Record has not stalled')
        old=self.learner
        self.learner=int(rng.choice(np.flatnonzero(np.arange(self.size)!=old)))
        before=mask.summary()['percent']
        released=mask.release_increment(step,reason='learner_reselection')
        event={'type':'learner_switch','reason':'pool_record_plateau','timing':'after_update',
               'step':step,'from':old,'to':self.learner,'phase':self.phase,'best':copy.deepcopy(self.best),
               'patience':self.patience,'rollback':False,'released':released,
               'percent_before':before,'percent_after':mask.summary()['percent']}
        self.events.append(event)
        self.phase+=1;self.phase_start_step=step;self.best=self.current=None
        return event

    def summary(self):
        return copy.deepcopy(self.__dict__)|{'switches':len(self.events),'rollback':False}

    def state_dict(self):return copy.deepcopy(self.__dict__)

    def load_state_dict(self,state):
        if state['size']!=self.size or state['patience']!=self.patience:raise ValueError('Controller configuration differs')
        self.__dict__.update(copy.deepcopy(state))


class WinRateController:
    """Strict raw win-rate threshold on one balanced round; no timeout/rollback."""
    def __init__(self,size,threshold,games_per_opponent,rng):
        if size<2 or not 0<threshold<1 or games_per_opponent<2 or games_per_opponent%2:
            raise ValueError('Invalid threshold controller configuration')
        self.size=size;self.threshold=threshold;self.games_per_opponent=games_per_opponent
        self.learner=int(rng.integers(size));self.initial_learner=self.learner
        self.phase=1;self.phase_start_step=0;self.best=self.current=None
        self.events=[];self.records=0;self.passed=False

    def observe(self,step,margins,opponents):
        own=np.asarray(margins);opponents=np.asarray(opponents,dtype=int)
        if own.shape!=opponents.shape or own.ndim!=1 or not np.isfinite(own).all():
            raise ValueError('One finite margin per opponent game is required')
        if own.size!=(self.size-1)*self.games_per_opponent or np.any((opponents<0)|(opponents>=self.size)|(opponents==self.learner)):
            raise ValueError('Every other individual must play the prescribed number of games')
        counts=np.bincount(opponents,minlength=self.size)
        if any(counts[i]!=self.games_per_opponent for i in range(self.size) if i!=self.learner):
            raise ValueError('Unbalanced opponent coverage')
        w,d,l=(int((own>0).sum()),int((own==0).sum()),int((own<0).sum()))
        self.current={'step':step,'learner':self.learner,'wins':w,'draws':d,'losses':l,
            'games':int(own.size),'score':w/own.size,'win_rate':w/own.size,
            'per_opponent':[{'id':i,'games':int(counts[i]),'wins':int((own[opponents==i]>0).sum()),
                'draws':int((own[opponents==i]==0).sum()),'losses':int((own[opponents==i]<0).sum()),
                'win_rate':float((own[opponents==i]>0).mean())} for i in range(self.size) if i!=self.learner]}
        if self.best is None or w*self.best['games']>self.best['wins']*own.size:
            self.best=copy.deepcopy(self.current);self.records+=1
        boundary=Fraction(str(self.threshold))
        self.passed=w*boundary.denominator>boundary.numerator*own.size
        return self.passed

    def reselect(self,step,rng):
        if not self.passed:raise ValueError('The strict win-rate threshold was not exceeded')
        previous=self.learner
        self.learner=int(rng.choice(np.flatnonzero(np.arange(self.size)!=previous)))
        event={'type':'learner_switch','reason':'win_rate_threshold','timing':'after_update',
            'step':step,'from':previous,'to':self.learner,'phase':self.phase,'threshold':self.threshold,
            'result':copy.deepcopy(self.current),'rollback':False,'released':False,
            'percent_before':100.,'percent_after':100.}
        self.events.append(event);self.phase+=1;self.phase_start_step=step
        self.best=self.current=None;self.passed=False
        return event

    def summary(self):
        return copy.deepcopy(self.__dict__)|{'switches':len(self.events),'rollback':False,'patience':None,
                                           'selection_mode':'win_rate_threshold'}

    def state_dict(self):return copy.deepcopy(self.__dict__)

    def load_state_dict(self,state):
        if any(state[k]!=getattr(self,k) for k in ('size','threshold','games_per_opponent')):
            raise ValueError('Controller configuration differs')
        self.__dict__.update(copy.deepcopy(state))
