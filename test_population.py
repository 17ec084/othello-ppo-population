import copy
import io
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from curriculum import ProgressiveUnfreezer
from model import ModelConfig, OthelloGPT
from population import (PopulationGroup,population_match,competition_update,
                        population_diversity,alien_update,random_game_pairings,learner_rewards)
from population_control import GroupWinTest,PopulationController


class PopulationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(14)
        self.config=ModelConfig(16,1,2)
        self.mask=ProgressiveUnfreezer(OthelloGPT(self.config),17,initial_percent=10,
                                      granularity=10,min_hold_updates=0)
        self.groups=[PopulationGroup(self.config,3,self.mask,'cpu',lr=.001,alien_lr=.001) for _ in range(2)]

    def test_batched_logits_equal_independent_full_models(self):
        policy=self.groups[0].policy
        tokens=torch.randint(0,6561,(3,4,8))
        logits=policy(tokens)
        weights=policy.snapshot()
        self.assertFalse(torch.equal(weights['embedding.weight'][0],weights['embedding.weight'][1]))
        for i in range(3):
            model=OthelloGPT(self.config)
            model.load_state_dict({name:value[i] for name,value in weights.items()})
            torch.testing.assert_close(logits[i],model(tokens[i]),atol=2e-6,rtol=1e-5)

    def test_selected_opponents_allow_repetition_for_independent_evaluation(self):
        policy=self.groups[1].policy
        tokens=torch.randint(0,6561,(3,4,8))
        ids=np.array([2,2,0])
        logits=policy(tokens,individuals=ids)
        weights=policy.snapshot()
        for slot,index in enumerate(ids):
            model=OthelloGPT(self.config)
            model.load_state_dict({name:value[index] for name,value in weights.items()})
            torch.testing.assert_close(logits[slot],model(tokens[slot]),atol=2e-6,rtol=1e-5)
        match=population_match([g.policy for g in self.groups],3,2,np.random.default_rng(8),opponent_indices=ids)
        np.testing.assert_array_equal(match['opponent_indices'],ids)
        with self.assertRaises(ValueError):
            population_match([g.policy for g in self.groups],3,2,np.random.default_rng(8),collect=1,opponent_indices=ids)

    def test_shuffled_B_results_records_rewards_and_gradients_follow_actual_individual(self):
        pairing=np.array([2,0,1]);inverse=np.argsort(pairing)
        before=self.groups[1].policy.snapshot()
        reference=PopulationGroup(self.config,3,self.mask,'cpu',lr=.001,alien_lr=.001,
                                  weights={name:value[pairing] for name,value in before.items()})
        actual=population_match([g.policy for g in self.groups],3,2,np.random.default_rng(9),
                                collect=1,record=True,opponent_indices=pairing)
        expected=population_match([self.groups[0].policy,reference.policy],3,2,np.random.default_rng(9),collect=1,record=True)
        np.testing.assert_array_equal(actual['margins'],expected['margins'])
        self.assertEqual(actual['trace'],expected['trace'])
        for left,right in zip(actual['records'],expected['records']):
            for a,b in zip(left,right):np.testing.assert_array_equal(a,b[inverse])
        competition_update(self.groups[1],actual,1,8)
        competition_update(reference,expected,1,8)
        expected_weights=reference.policy.snapshot()
        for name,value in self.groups[1].policy.snapshot().items():
            torch.testing.assert_close(value,expected_weights[name][inverse],atol=2e-6,rtol=1e-5)
        np.testing.assert_allclose(self.groups[1].baseline,reference.baseline[inverse])

    def test_competition_updates_only_active_group_and_only_open_entries(self):
        before=[g.policy.snapshot() for g in self.groups]
        match=population_match([g.policy for g in self.groups],3,2,np.random.default_rng(4),collect=0,record=True)
        self.assertEqual(match['margins'].shape,(3,2))
        self.assertEqual(match['games'],6)
        self.assertGreater(len(match['trace']),50)
        result=competition_update(self.groups[0],match,0,8)
        self.assertGreater(result['parameter_delta_l2'],0)
        for i,g in enumerate(self.groups):
            for name,value in g.policy.snapshot().items():
                if i==1:
                    self.assertTrue(torch.equal(value,before[i][name]))
                else:
                    self.assertTrue(torch.equal(value[:,~self.mask.masks[name]],before[i][name][:,~self.mask.masks[name]]))

    def test_per_game_draws_preserve_128_games_and_balanced_colors_for_every_model(self):
        pairing=random_game_pairings(10,128,np.random.default_rng(42))
        self.assertEqual(pairing.shape,(10,128))
        np.testing.assert_array_equal(np.sort(pairing,axis=0),np.broadcast_to(np.arange(10)[:,None],(10,128)))
        for row in pairing:self.assertEqual(set(row),set(range(10)))
        self.assertGreater(len({tuple(p) for p in pairing.T}),100)
        for b in range(10):
            self.assertEqual(int((pairing[:,::2]==b).sum()),64)
            self.assertEqual(int((pairing[:,1::2]==b).sum()),64)

    def test_per_game_B_models_records_and_rewards_match_independent_greedy_games(self):
        # A scalar reference loads the actual A/B models for each board. Greedy
        # actions remove RNG ordering as a difference between batched and solo play.
        class Greedy:
            def gumbel(self,size):return np.zeros(size)
        pairing=np.array([[2,0,1,2],[0,2,2,1],[1,1,0,0]])
        policies=[g.policy for g in self.groups]
        before=[g.policy.snapshot() for g in self.groups]
        match=population_match(policies,3,4,Greedy(),collect=1,opponent_indices=pairing)
        references={}
        def trajectory(records,individual,game):
            return [(t[individual,k],m[individual,k],int(a[individual,k]))
                    for t,m,a,v,ids in records for k in range(v.shape[1])
                    if v[individual,k] and ids[individual,k]==game]
        manual_rewards=np.empty((3,4))
        for a in range(3):
            for game in range(4):
                b=int(pairing[a,game])
                if (a,b) not in references:
                    solo=[PopulationGroup(self.config,1,self.mask,'cpu',
                           weights={n:w[index:index+1] for n,w in before[g].items()})
                          for g,index in enumerate((a,b))]
                    references[a,b]=population_match([g.policy for g in solo],1,2,Greedy(),collect=1)
                reference=references[a,b]
                self.assertEqual(match['margins'][a,game],reference['margins'][0,game%2])
                manual_rewards[b,game]=-reference['margins'][0,game%2]
                actual=trajectory(match['records'],b,game)
                wanted=trajectory(reference['records'],0,game%2)
                self.assertEqual(len(actual),len(wanted))
                for left,right in zip(actual,wanted):
                    np.testing.assert_array_equal(left[0],right[0])
                    np.testing.assert_array_equal(left[1],right[1])
                    self.assertEqual(left[2],right[2])
        np.testing.assert_array_equal(learner_rewards(match,1),manual_rewards)
        result=competition_update(self.groups[1],match,1,8)
        self.assertGreater(result['parameter_delta_l2'],0)
        np.testing.assert_allclose(self.groups[1].baseline[:,0],.1*manual_rewards[:,1::2].mean(1))
        np.testing.assert_allclose(self.groups[1].baseline[:,1],.1*manual_rewards[:,::2].mean(1))
        for n,w in self.groups[0].policy.snapshot().items():
            torch.testing.assert_close(w,before[0][n],atol=0,rtol=0)
        for n,w in self.groups[1].policy.snapshot().items():
            torch.testing.assert_close(w[:,~self.mask.masks[n]],before[1][n][:,~self.mask.masks[n]],atol=0,rtol=0)

    def test_constant_per_game_matrix_matches_previous_fixed_pairing_exactly(self):
        pairing=np.array([2,0,1])
        for learner in (0,1):
            fixed=population_match([g.policy for g in self.groups],3,4,np.random.default_rng(73),
                                   collect=learner,record=True,opponent_indices=pairing)
            matrix=population_match([g.policy for g in self.groups],3,4,np.random.default_rng(73),
                                    collect=learner,record=True,opponent_indices=np.repeat(pairing[:,None],4,axis=1))
            np.testing.assert_array_equal(fixed['margins'],matrix['margins'])
            np.testing.assert_array_equal(learner_rewards(fixed,learner),learner_rewards(matrix,learner))
            self.assertEqual(fixed['trace'],matrix['trace'])
            self.assertEqual(len(fixed['records']),len(matrix['records']))
            for left,right in zip(fixed['records'],matrix['records']):
                for x,y in zip(left,right):np.testing.assert_array_equal(x,y)

    def test_per_game_matching_rejects_duplicate_models_in_one_round(self):
        pairing=np.array([[0,1],[0,2],[2,0]])
        with self.assertRaisesRegex(ValueError,'permutation'):
            population_match([g.policy for g in self.groups],3,2,np.random.default_rng(1),opponent_indices=pairing)

    def test_release_preserves_weights_and_all_optimizer_momenta(self):
        g=self.groups[0]
        tokens=torch.randint(0,6561,(3,2,8))
        for optimizer in (g.competition,g.alien,g.sharing):
            optimizer.zero_grad()
            g.policy(tokens).sum().backward()
            optimizer.step()
        old_active=g.policy.active
        states=[{name:copy.deepcopy(o.optimizer.state[p]) for name,p in old_active.items()} for o in (g.competition,g.alien,g.sharing)]
        before=g.policy.snapshot()
        self.mask.release_increment(1)
        g.expand(self.mask)
        for name,value in g.policy.snapshot().items():self.assertTrue(torch.equal(value,before[name]))
        for optimizer,old in zip((g.competition,g.alien,g.sharing),states):
            for name,p in g.policy.active.items():
                if name not in old:continue
                previous=old[name]['exp_avg']
                now=optimizer.optimizer.state[p]['exp_avg']
                torch.testing.assert_close(now[:,:previous.shape[1]],previous,atol=0,rtol=0)
                self.assertTrue((now[:,previous.shape[1]:]==0).all())

    def test_group_copy_is_index_matched_including_frozen_values(self):
        g=self.groups[0]
        for optimizer in (g.competition,g.alien,g.sharing):
            optimizer.zero_grad()
            g.policy(torch.randint(0,6561,(3,2,8))).sum().backward()
            optimizer.step()
        source=self.groups[1].policy.snapshot()
        g.baseline[:]=9
        g.copy_from(self.groups[1])
        for name,value in g.policy.snapshot().items():self.assertTrue(torch.equal(value,source[name]))
        self.assertEqual(len(g.competition.optimizer.state),0)
        self.assertEqual(len(g.alien.optimizer.state),0)
        self.assertEqual(len(g.sharing.optimizer.state),0)
        self.assertTrue((g.baseline==0).all())

    def test_js_does_not_reward_identical_random_policies(self):
        class Fixed:
            size=3
            device='cpu'
            values=torch.zeros(3,1,64)
            def __call__(self,tokens):return self.values.expand(-1,tokens.shape[1],-1)
        policy=Fixed()
        tokens=np.zeros((2,8),dtype=np.int64)
        masks=np.zeros((2,64),dtype=bool);masks[:,:2]=True
        self.assertAlmostEqual(population_diversity(policy,tokens,masks)['value'],0,places=7)
        policy.values[0,0,0]=30;policy.values[1:,0,1]=30
        d=population_diversity(policy,tokens,masks)
        self.assertGreater(d['value'],.6)
        self.assertGreater(d['individual_novelty'][0],d['individual_novelty'][1])

    def test_alien_reward_increases_js_without_moving_frozen_weights(self):
        g=self.groups[0]
        tokens=np.random.default_rng(4).integers(0,6561,(8,8))
        masks=np.ones((8,64),dtype=bool)
        before=g.policy.snapshot()
        d0=population_diversity(g.policy,tokens,masks)['value']
        for _ in range(3):alien_update(g,tokens,masks,4)
        self.assertGreater(population_diversity(g.policy,tokens,masks)['value'],d0)
        for name,value in g.policy.snapshot().items():
            self.assertTrue(torch.equal(value[:,~self.mask.masks[name]],before[name][:,~self.mask.masks[name]]))


class GroupTest(unittest.TestCase):
    def test_group_test_uses_win_signs_independently_of_margin_magnitude(self):
        a,b=GroupWinTest(),GroupWinTest()
        a.add_group(np.ones((100,2)))
        b.add_group(np.full((100,2),64))
        self.assertEqual(a.summary()['rounds'],1)
        self.assertEqual(a.summary()['mean_margin'],1)
        self.assertEqual(a.p_value,b.p_value)
        self.assertLess(b.p_value,.05)
        majority=GroupWinTest()
        margins=np.ones((100,2));margins[-10:]=-64
        majority.add_group(margins)
        self.assertEqual(majority.summary()['wins'],180)
        self.assertLess(majority.summary()['mean_margin'],0)
        self.assertLess(majority.p_value,.05)
        draws=GroupWinTest();draws.add_group(np.zeros((100,2)))
        self.assertEqual(draws.p_value,1)

    def test_independent_heterogeneous_null_and_optional_stopping(self):
        # Two independent +/-1 observations with means +.6 and -.6.
        # No assumption that each pair's expected margin is nonpositive.
        bets=GroupWinTest().evidence.BETS
        expected=np.zeros(len(bets))
        for first,p1 in ((1,.8),(-1,.2)):
            for second,p2 in ((1,.2),(-1,.8)):
                expected+=p1*p2*np.exp(bets*(first+second)/2-.5*bets**2)
        self.assertTrue((expected<=1+1e-12).all())
        # A random opponent can make both color outcomes perfectly correlated.
        # Treating these two games as independent would give an invalid bound.
        correct=np.cosh(bets)*np.exp(-.5*bets**2)
        incorrect=np.cosh(2*bets)*np.exp(-bets**2)
        self.assertTrue((correct<=1+1e-12).all())
        self.assertTrue((incorrect>1).all())
        horizon=6
        crossed=0
        for path in itertools.product((-1,1),repeat=2*horizon):
            test=GroupWinTest()
            ever=False
            for i in range(horizon):
                test.add_group(np.array(path[2*i:2*i+2]).reshape(1,2)*64)
                ever|=test.p_value<=.05
            crossed+=ever
        self.assertLessEqual(crossed/(2**(2*horizon)),.05)

    def test_group_controller_directions_and_resume(self):
        mask=ProgressiveUnfreezer(OthelloGPT(ModelConfig(16,1,2)),4,min_hold_updates=0,granularity=10)
        c=PopulationController()
        event=c.observe(1,np.full((100,2),64),mask)
        self.assertEqual(event['to'],'B')
        event=c.observe(2,np.full((100,2),64),mask)
        self.assertEqual(event['type'],'opponent_copy')
        self.assertEqual(event['learner'],'B')
        self.assertEqual(c.learner,1)
        c.observe(3,np.full((100,2),1),mask)
        restored=PopulationController();restored.load_state_dict(copy.deepcopy(c.state_dict()))
        self.assertEqual(c.summary(),restored.summary())


class PopulationIntegrationTest(unittest.TestCase):
    def run_trainer(self,directory,updates,resume=False,extra_patches=()):
        import population_train
        from contextlib import ExitStack
        real_match=population_train.population_match
        def held_out_draws(*args,**kwargs):
            result=real_match(*args,**kwargs)
            if kwargs.get('collect') is None:result['margins'][:]=0
            return result
        argv=['population_train.py','--out',directory,'--population','3','--width','16','--layers','1',
            '--heads','2','--diversity-positions','8','--patience','2','--initial-percent','10',
            '--lr','.001','--alien-lr','.001','--teacher-games-per-opponent','2','--sharing-positions','4','--eval-every','100','--device','cpu','--updates',str(updates)]
        if resume:argv+=['--resume',str(Path(directory)/'latest.pt')]
        with ExitStack() as stack:
            stack.enter_context(patch.object(population_train,'population_match',side_effect=held_out_draws))
            stack.enter_context(patch('sys.argv',argv));stack.enter_context(patch('sys.stdout',io.StringIO()))
            for extra in extra_patches:stack.enter_context(extra)
            population_train.main()
        return torch.load(Path(directory)/'latest.pt',map_location='cpu',weights_only=False)

    def test_full_round_switch_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_trainer(d,3)
            saved=self.run_trainer(d,5,True)
            rows=[json.loads(s) for s in (Path(d)/'metrics.jsonl').read_text().splitlines()]
            switches=[json.loads(s) for s in (Path(d)/'switches.jsonl').read_text().splitlines()]
            self.assertEqual(saved['training_games'],30)
            self.assertEqual(saved['test_games'],30)
            self.assertGreater(saved['teacher_games'],0)
            self.assertGreater(sum(saved['sharing_totals']),0)
            self.assertEqual(saved['learner_steps'],[3,2])
            self.assertEqual([e['step'] for e in switches],[2,4])
            self.assertEqual(rows[1]['learner'],'B')
            self.assertEqual(saved['mask']['percent'],101)
            self.assertIsNone(saved['pending'])
            pairings=[tuple(row['pairing']) for row in rows]
            self.assertGreater(len(set(pairings)),1)
            self.assertTrue(all(sorted(pairing)==[0,1,2] for pairing in pairings))
            self.assertEqual(saved['protocol'],'population-sharing-distillation-v3')
            for current,target in zip(saved['diversity'],saved['targets']):
                self.assertGreaterEqual(current['value']+1e-10,target)

    def test_stop_after_teacher_evaluation_resumes_same_weights_without_repeating_games(self):
        import population_train
        real_evaluation=population_train.evaluate_teachers
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as reference:
            def stop_after_evaluation(*args,**kwargs):
                result=real_evaluation(*args,**kwargs)
                (Path(d)/'STOP').touch()
                return result
            saved=self.run_trainer(d,1,extra_patches=(
                patch.object(population_train,'evaluate_teachers',side_effect=stop_after_evaluation),))
            self.assertEqual(saved['step'],0)
            self.assertEqual(saved['pending']['stage'],'sharing')
            self.assertEqual(saved['training_games'],6)
            self.assertEqual(saved['test_games'],0)
            self.assertEqual(saved['teacher_games'],12)
            (Path(d)/'STOP').unlink()
            resumed=self.run_trainer(d,1,True)
            expected=self.run_trainer(reference,1)
            self.assertEqual(resumed['training_games'],6)
            self.assertEqual(resumed['test_games'],6)
            self.assertEqual(resumed['teacher_games'],12)
            self.assertEqual(resumed['sharing_totals'],expected['sharing_totals'])
            for actual,wanted in zip(resumed['groups'],expected['groups']):
                for name,value in actual['weights'].items():
                    torch.testing.assert_close(value,wanted['weights'][name],atol=0,rtol=0)

    def test_stop_during_alien_and_resume_without_repeating_competition(self):
        import population_train
        real_diversity=population_train.population_diversity
        real_alien=population_train.alien_update
        count=0
        with tempfile.TemporaryDirectory() as d:
            def force_repair(*args,**kwargs):
                nonlocal count
                count+=1
                value=real_diversity(*args,**kwargs)
                if count>=3:value['value']=0
                return value
            def stop_after_one_round(*args,**kwargs):
                value=real_alien(*args,**kwargs)
                (Path(d)/'STOP').touch()
                return value
            saved=self.run_trainer(d,1,extra_patches=(
                patch.object(population_train,'population_diversity',side_effect=force_repair),
                patch.object(population_train,'alien_update',side_effect=stop_after_one_round)))
            self.assertEqual(saved['step'],0)
            self.assertEqual(saved['pending']['stage'],'alien')
            self.assertEqual(saved['training_games'],6)
            (Path(d)/'STOP').unlink()
            saved=self.run_trainer(d,1,True)
            self.assertEqual(saved['step'],1)
            self.assertEqual(saved['training_games'],6)
            self.assertEqual(saved['test_games'],6)
            self.assertEqual(saved['learner_steps'],[1,0])
            self.assertIsNone(saved['pending'])


if __name__=='__main__':unittest.main(verbosity=2)
