"""Alternating populations selected by observed record score, without p-values."""
import copy
import numpy as np


class RecordController:
    def __init__(self, patience=200):
        if patience<1:raise ValueError('Patience must be positive')
        self.patience=patience
        self.learner=0
        self.phase=1
        self.phase_start_step=0
        self.best=None
        self.current=None
        self.events=[]
        self.records=0

    def observe(self,step,margins):
        margins=np.asarray(margins)
        if not margins.size or not np.isfinite(margins).all():
            raise ValueError('Finite evaluation results required')
        own=margins if self.learner==0 else -margins
        w,d,l=(int((own>0).sum()),int((own==0).sum()),int((own<0).sum()))
        self.current={'step':step,'wins':w,'draws':d,'losses':l,'games':own.size,
                      'score_units':2*w+d,'score':(w+.5*d)/own.size,'win_rate':w/own.size}
        # Integer cross-products preserve strict improvements and exact ties.
        improved=(self.best is None or
                  self.current['score_units']*self.best['games']>
                  self.best['score_units']*self.current['games'])
        if improved:
            self.best=copy.deepcopy(self.current)
            self.records+=1
        return improved

    def expired(self,step):
        return self.best is not None and step-self.best['step']>=self.patience

    def switch(self,step,mask):
        if not self.expired(step):raise ValueError('Record has not stalled')
        old=self.learner
        before=mask.summary()['percent']
        released=mask.release_increment(step) if old==1 else False
        event={'type':'learner_switch','reason':'record_plateau','timing':'after_update',
               'step':step,'from':'AB'[old],'to':'AB'[1-old],'phase':self.phase,
               'best':copy.deepcopy(self.best),'restored_step':self.best['step'],
               'patience':self.patience,'released':released,'percent_before':before,
               'percent_after':mask.summary()['percent']}
        self.events.append(event)
        self.learner=1-old
        self.phase+=1
        self.phase_start_step=step
        self.best=self.current=None
        return event

    def summary(self):
        return {'mode':'record_score','learner':'AB'[self.learner],
                'frozen_opponent':'AB'[1-self.learner],'phase':self.phase,
                'phase_start_step':self.phase_start_step,'best':self.best,'current':self.current,
                'patience':self.patience,'switches':len(self.events),'events':self.events,
                'records':self.records}

    def plateau_summary(self,step):
        t=self.best['step'] if self.best else None
        return {'mode':'record_score','initialized':t is not None,'t':t,'patience':self.patience,
                'unchanged_updates':step-t if t is not None else 0,
                'deadline_step':t+self.patience if t is not None else None}

    def state_dict(self):return copy.deepcopy(self.__dict__)

    def load_state_dict(self,state):
        if state['patience']!=self.patience:raise ValueError('Saved patience differs')
        self.__dict__.update(copy.deepcopy(state))
