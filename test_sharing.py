import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from curriculum import ProgressiveUnfreezer
from model import ModelConfig, OthelloGPT, masked_log_probs
from population import PopulationGroup, population_match
from sharing import sharing_plan, sharing_update, evaluate_teachers


class SharingTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(52)
        config=ModelConfig(16,1,2)
        self.mask=ProgressiveUnfreezer(OthelloGPT(config),17,initial_percent=10,granularity=10)
        self.groups=[PopulationGroup(config,3,self.mask,'cpu',sharing_lr=.001) for _ in (0,1)]
        self.match=population_match([g.policy for g in self.groups],3,2,np.random.default_rng(4),collect=0)
        self.plan=sharing_plan(self.groups[0],self.match['records'],[.3,.6,.9],
                               np.random.default_rng(9),positions=8,fraction=1,chunk=4)

    def test_teacher_selection_and_detached_targets_match_the_selected_policies(self):
        plan,policy=self.plan,self.groups[0].policy
        self.assertEqual(plan['enabled'].tolist(),[True,True,False])
        self.assertEqual(set(plan['teacher_ids'][0]),{1,2})
        self.assertTrue((plan['teacher_ids'][1]==2).all())
        self.assertFalse(np.array_equal(plan['tokens'][0],plan['tokens'][1]))
        self.assertTrue(np.all(plan['targets'][~plan['masks']]==0))
        np.testing.assert_allclose(plan['targets'].sum(2),1,atol=2e-7)
        with torch.no_grad():
            rows=torch.as_tensor(plan['tokens'],dtype=torch.long)
            legal=torch.as_tensor(plan['masks'])
            expected=sum(masked_log_probs(policy(rows,individuals=plan['teacher_ids'][:,k]),legal).exp()
                         for k in range(2))/2
        np.testing.assert_allclose(plan['targets'],expected.numpy(),atol=2e-7)

    def test_sharing_moves_students_toward_teachers_preserves_frozen_and_other_group(self):
        before=[g.policy.snapshot() for g in self.groups]
        targets=self.plan['targets'].copy()
        result=sharing_update(self.groups[0],self.plan,chunk=4)
        self.assertGreater(result['parameter_delta_l2'],0)
        self.assertLess(result['kl_after'],result['kl_before'])
        np.testing.assert_array_equal(self.plan['targets'],targets)
        for name,value in self.groups[0].policy.snapshot().items():
            self.assertTrue(torch.equal(value[:,~self.mask.masks[name]],before[0][name][:,~self.mask.masks[name]]))
            self.assertTrue(torch.equal(value[2],before[0][name][2]))
        for name,value in self.groups[1].policy.snapshot().items():
            self.assertTrue(torch.equal(value,before[1][name]))

    def test_excluded_teacher_does_not_move_from_existing_adam_momentum(self):
        warm=copy.deepcopy(self.plan)
        warm['enabled'][:]=True
        warm['targets'][2]=warm['targets'][0]
        sharing_update(self.groups[0],warm,4)
        before=self.groups[0].policy.snapshot()
        old_state={n:copy.deepcopy(self.groups[0].sharing.optimizer.state[p])
                   for n,p in self.groups[0].policy.active.items()}
        sharing_update(self.groups[0],self.plan,4)
        for name,value in self.groups[0].policy.snapshot().items():
            self.assertTrue(torch.equal(value[2],before[name][2]))
        for name,p in self.groups[0].policy.active.items():
            for k in ('exp_avg','exp_avg_sq'):
                torch.testing.assert_close(self.groups[0].sharing.optimizer.state[p][k][2],
                                           old_state[name][k][2],atol=0,rtol=0)

    def test_tied_scores_skip_sharing_without_moving_any_weights(self):
        plan=sharing_plan(self.groups[0],self.match['records'],[.5,.5,.5],np.random.default_rng(2),positions=4)
        before=self.groups[0].policy.snapshot()
        result=sharing_update(self.groups[0],plan)
        self.assertEqual(result['students'],0)
        self.assertEqual(len(self.groups[0].sharing.optimizer.state),0)
        for name,value in self.groups[0].policy.snapshot().items():
            self.assertTrue(torch.equal(value,before[name]))

    def test_teacher_evaluation_uses_draw_half_scores_on_separate_games(self):
        with patch('sharing.population_match',return_value={'margins':np.array([[-4,0],[4,8],[0,0]])}) as play:
            result=evaluate_teachers(self.groups[0],2,np.random.default_rng(3))
        self.assertEqual(result['scores'],[.25,1.,.5])
        self.assertEqual(result['games'],12)
        self.assertEqual([c.args[0][1] for c in play.call_args_list],['random','position'])


if __name__=='__main__':unittest.main(verbosity=2)
