import copy
import unittest
import numpy as np
import torch

from alternating_train import set_roles
from curriculum import ProgressiveUnfreezer
from model import ModelConfig, OthelloGPT
from sequential import AnytimeWinTest, AlternatingController
from train import play_batch, update_policy


class AnytimeTest(unittest.TestCase):
    def test_all_wins_cross_and_draws_never_create_evidence(self):
        test=AnytimeWinTest()
        for _ in range(100):
            if test.add_pair(20,20): break
        self.assertLessEqual(test.p_value,.05)
        draws=AnytimeWinTest()
        for _ in range(100): draws.add_pair(0,0)
        self.assertEqual(draws.p_value,1.)
        losses=AnytimeWinTest()
        for _ in range(100): losses.add_pair(-20,-20)
        self.assertEqual(losses.p_value,1.)

    def test_exact_fair_null_optional_stopping_crossing_probability(self):
        # Enumerate all 2^16 paths of paired payoffs ±1 under a fair null.
        # This checks ever crossing, not merely the terminal rejection rate.
        horizon=16
        paths=np.arange(2**horizon,dtype=np.uint32)
        wealth=np.ones((len(paths),len(AnytimeWinTest.BETS)))
        crossed=np.zeros(len(paths),dtype=bool)
        for time in range(horizon):
            x=2*((paths>>time)&1).astype(float)-1
            wealth*=1+x[:,None]*AnytimeWinTest.BETS
            crossed|=wealth.mean(-1)>=20
        self.assertGreater(int(crossed.sum()),0)
        self.assertLessEqual(float(crossed.mean()),.05)

    def test_switch_A_B_then_B_A_releases_exactly_point_one(self):
        torch.set_num_threads(1)
        model=OthelloGPT(ModelConfig(16,1,2))
        mask=ProgressiveUnfreezer(model,15,initial_percent=.1,min_hold_updates=0,granularity=10)
        controller=AlternatingController()
        old={n:m.clone() for n,m in mask.masks.items()}
        event=controller.observe(1,np.full(200,16),mask)
        self.assertEqual((event['from'],event['to']),('A','B'))
        self.assertFalse(event['released'])
        self.assertEqual(mask.summary()['percent'],.1)
        self.assertEqual(controller.test.p_value,1.)
        self.assertEqual(controller.test.pairs,0)
        event=controller.observe(2,np.full(200,-16),mask)
        self.assertEqual((event['from'],event['to']),('B','A'))
        self.assertTrue(event['released'])
        self.assertEqual(mask.summary()['percent'],.2)
        self.assertEqual(mask.summary()['active_parameters'],mask.total*2//1000)
        for name in old:self.assertFalse((old[name]&~mask.masks[name]).any())

    def test_no_switch_without_evidence_and_resume_preserves_evidence(self):
        controller=AlternatingController()
        for _ in range(5):controller.test.add_pair(1,1)
        data=copy.deepcopy(controller.state_dict())
        restored=AlternatingController()
        restored.load_state_dict(data)
        self.assertEqual(restored.summary(),controller.summary())
        for a,b in [(1,-1),(0,1),(-1,-1),(1,1)]:
            self.assertEqual(controller.test.add_pair(a,b),restored.test.add_pair(a,b))
        self.assertEqual(controller.test.p_value,restored.test.p_value)

    def test_only_active_model_changes_and_inactive_adam_does_not_run(self):
        torch.set_num_threads(1)
        torch.manual_seed(48)
        models=[OthelloGPT(ModelConfig(16,1,2)) for _ in range(2)]
        optimizers=[torch.optim.Adam(m.parameters(),lr=1e-4) for m in models]
        mask=ProgressiveUnfreezer(models[0],15,initial_percent=.1,min_hold_updates=0,granularity=10)
        for learner in (0,1):
            set_roles(models,learner)
            opponent_before={n:p.detach().clone() for n,p in models[1-learner].named_parameters()}
            active_before={n:p.detach().clone() for n,p in models[learner].named_parameters()}
            match=play_batch(models,8,np.random.default_rng(11+learner),'cpu',collect=True,reward_mode='margin')
            result=update_policy(models[learner],optimizers[learner],match['records'][learner],
                                 match['rewards'][learner],match['black_agents'],learner,[0.,0.],unfreezer=mask)
            self.assertGreater(result['parameter_delta_l2'],0)
            for name,p in models[1-learner].named_parameters():
                self.assertTrue(torch.equal(p,opponent_before[name]),name)
            for name,p in models[learner].named_parameters():
                self.assertTrue(torch.equal(p.detach()[~mask.masks[name]],active_before[name][~mask.masks[name]]),name)


if __name__=='__main__':unittest.main(verbosity=2)
