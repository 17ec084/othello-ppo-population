"""Exact t+50/t+499 behavior, original-anchor mutations, and resume."""
import copy
import io
import unittest

import torch

from curriculum import ProgressiveUnfreezer
from plateau import PlateauRecovery
from sequential import AlternatingController


class PlateauTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(78)
        self.model=torch.nn.Linear(4,2)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=.001)
        self.mask=ProgressiveUnfreezer(self.model,31,initial_percent=50,min_hold_updates=0)
        self.baseline=[2.,-3.]
        self.recovery=PlateauRecovery(15)
        self.recovery.capture(100,1,0,.07,self.model,self.baseline,'phase_start')
        self.original={n:p.detach().clone() for n,p in self.model.named_parameters()}

    def before(self,step):
        return self.recovery.before_step(step,1,0,self.model,self.optimizer,self.baseline,self.mask)

    def test_no_mutation_until_t_plus_50_and_frozen_values_stay_original(self):
        for step in range(101,150):self.assertIsNone(self.before(step))
        # Simulate stale optimizer momentum and post-anchor drift.
        for p in self.model.parameters():
            self.optimizer.state[p]={'step':torch.tensor(5.),'exp_avg':torch.ones_like(p),'exp_avg_sq':torch.ones_like(p)}
            with torch.no_grad():p.add_(20)
        self.baseline[:]=[100.,100.]
        event=self.before(150)
        self.assertEqual(event['t'],100)
        self.assertEqual(event['attempt'],1)
        self.assertEqual(event['mutated_parameters'],self.mask.summary()['active_parameters'])
        self.assertEqual(len(self.optimizer.state),0)
        self.assertEqual(self.baseline,[2.,-3.])
        changed=0
        for name,p in self.model.named_parameters():
            mask=self.mask.masks[name]
            self.assertTrue(torch.equal(p.detach()[~mask],self.original[name][~mask]))
            self.assertTrue(torch.equal(self.recovery.anchor_weights[name],self.original[name]))
            changed+=int((p.detach()[mask]!=self.original[name][mask]).sum())
        self.assertEqual(changed,event['mutated_parameters'])
        self.assertIsNone(self.before(150))

    def test_every_mutation_restarts_from_original_t_not_previous_model(self):
        self.before(150)
        previous_mutant={n:p.detach().clone() for n,p in self.model.named_parameters()}
        clone=PlateauRecovery(15)
        clone.load_state_dict(copy.deepcopy(self.recovery.state_dict()))
        model2=copy.deepcopy(self.model)
        opt2=torch.optim.Adam(model2.parameters(),lr=.001)
        with torch.no_grad():
            for p in self.model.parameters():p.fill_(1000)
            for p in model2.parameters():p.fill_(-1000)
        self.before(200)
        clone.before_step(200,1,0,model2,opt2,[9.,9.],self.mask)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(self.model.parameters(),model2.parameters())))
        self.assertTrue(any(not torch.equal(p,previous_mutant[n]) for n,p in self.model.named_parameters()))
        self.assertTrue(all(torch.equal(self.recovery.anchor_weights[n],p) for n,p in self.original.items()))

    def test_nine_mutations_then_restore_at_t_plus_499(self):
        for step in range(101,600):
            self.before(step)
            event=self.recovery.after_step(step,1,0,.07,self.model,self.optimizer,self.baseline)
            if step<599:self.assertIsNone(event)
            else:self.assertEqual(event['type'],'restore_and_switch')
        self.assertEqual(self.recovery.mutation_steps,list(range(150,551,50)))
        self.assertEqual(len(self.recovery.mutation_steps),9)
        for name,p in self.model.named_parameters():self.assertTrue(torch.equal(p,self.original[name]))

    def test_improvement_even_on_timeout_step_cancels_old_clock(self):
        self.before(150)
        self.recovery.after_step(151,1,0,.069,self.model,self.optimizer,self.baseline)
        self.assertEqual(self.recovery.t,151)
        self.assertEqual(self.recovery.mutation_steps,[])
        self.assertIsNone(self.before(200))
        event=self.recovery.after_step(650,1,0,.068,self.model,self.optimizer,self.baseline)
        self.assertEqual(event['type'],'p_improved')
        self.assertEqual(self.recovery.t,650)
        self.assertEqual(self.recovery.summary(650)['deadline_step'],1149)

    def test_snapshot_and_mutation_rng_survive_checkpoint(self):
        self.before(150)
        buffer=io.BytesIO()
        torch.save(self.recovery.state_dict(),buffer)
        buffer.seek(0)
        restored=PlateauRecovery(15)
        restored.load_state_dict(torch.load(buffer,weights_only=True))
        model2=copy.deepcopy(self.model)
        opt2=torch.optim.Adam(model2.parameters(),lr=.001)
        self.assertEqual(self.recovery.summary(199),restored.summary(199))
        self.before(200)
        restored.before_step(200,1,0,model2,opt2,[0.,0.],self.mask)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(self.model.parameters(),model2.parameters())))

    def test_timeout_switch_is_not_labeled_statistical_significance(self):
        controller=AlternatingController()
        event=controller.switch(499,self.mask,'plateau_timeout',{'t':0})
        self.assertFalse(event['significance_confirmed'])
        self.assertFalse(event['test']['passed'])
        self.assertEqual(controller.learner,1)
        self.assertFalse(event['released'])
        event=controller.switch(998,self.mask,'plateau_timeout',{'t':499})
        self.assertTrue(event['released'])
        self.assertEqual(controller.learner,0)



if __name__=='__main__':unittest.main(verbosity=2)
