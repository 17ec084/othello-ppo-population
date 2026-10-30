import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from curriculum import ProgressiveUnfreezer
from individual_control import IndividualController
from individual_train import play_individual,training_rewards
from model import ModelConfig,OthelloGPT
from train import play_batch,update_policy


class IndividualControlTest(unittest.TestCase):
    def test_strict_record_200_boundary_and_no_rollback_reselection_every_time(self):
        rng=np.random.default_rng(19)
        mask=ProgressiveUnfreezer(OthelloGPT(ModelConfig(16,1,2)),5,initial_percent=.1,granularity=10,min_hold_updates=0)
        c=IndividualController(20,200,rng);first=c.learner
        self.assertTrue(c.observe(0,np.array([1,0,-1,1])))
        self.assertFalse(c.observe(10,np.array([1,0,-1,1])))
        self.assertEqual(c.best['step'],0)
        self.assertFalse(c.expired(199));self.assertTrue(c.expired(200))
        self.assertTrue(c.observe(200,np.ones(4)))
        self.assertFalse(c.expired(200));self.assertTrue(c.expired(400))
        e=c.reselect(400,mask,rng)
        self.assertNotEqual(first,c.learner);self.assertFalse(e['rollback'])
        self.assertEqual(mask.summary()['percent'],.2)
        self.assertEqual(mask.events[-1]['reason'],'learner_reselection')
        self.assertIsNone(c.best)
        c.observe(400,np.zeros(4));c.reselect(600,mask,rng)
        self.assertEqual(mask.summary()['percent'],.3)
        state=c.state_dict();loaded=IndividualController(20,200,rng);loaded.load_state_dict(state)
        self.assertEqual(c.summary(),loaded.summary())

    def test_uniform_selection_excludes_current_and_random_opponents_exclude_self(self):
        rng=np.random.default_rng(64)
        selected=[IndividualController(20,200,rng).learner for _ in range(1000)]
        self.assertEqual(set(selected),set(range(20)))
        self.assertLess(max(np.bincount(selected,minlength=20)),90)


class IndividualPlayTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(7)
        self.models=[OthelloGPT(ModelConfig(16,1,2)) for _ in range(3)]

    def test_mixed_opponent_games_match_solo_reference_and_only_learner_changes(self):
        class Greedy:
            def gumbel(self,size):return np.zeros(size)
        opponents=np.array([1,2,2,1,1,2])
        match=play_individual(self.models,0,6,Greedy(),collect=True,record=True,opponents=opponents)
        refs={j:play_batch([self.models[0],self.models[j]],2,Greedy(),'cpu',collect=True,reward_mode='margin') for j in (1,2)}
        def trajectory(records,game):
            return [(t[k],m[k],a[k]) for t,m,a,ids in records for k in range(len(ids)) if ids[k]==game]
        for game,j in enumerate(opponents):
            self.assertEqual(match['margins'][game],refs[j]['rewards'][0][game%2])
            actual=trajectory(match['records'],game);wanted=trajectory(refs[j]['records'][0],game%2)
            self.assertEqual(len(actual),len(wanted))
            for a,b in zip(actual,wanted):
                for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        before=[copy.deepcopy(m.state_dict()) for m in self.models]
        mask=ProgressiveUnfreezer(self.models[0],11,initial_percent=10,granularity=10,min_hold_updates=0)
        optimizer=torch.optim.Adam(self.models[0].parameters(),lr=1e-3)
        update_policy(self.models[0],optimizer,match['records'],match['margins'],match['black_agents'],0,[0.,0.],16,unfreezer=mask)
        self.assertTrue(any(not torch.equal(w,before[0][n]) for n,w in self.models[0].state_dict().items()))
        for n,w in self.models[0].state_dict().items():torch.testing.assert_close(w[~mask.masks[n]],before[0][n][~mask.masks[n]],atol=0,rtol=0)
        for i in (1,2):
            for n,w in self.models[i].state_dict().items():torch.testing.assert_close(w,before[i][n],atol=0,rtol=0)

    def test_opponent_sampling_never_selects_learner_and_balances_colors(self):
        match=play_individual(self.models,1,128,np.random.default_rng(88))
        self.assertEqual(set(match['opponents']),{0,2})
        self.assertEqual(int((match['black_agents']==0).sum()),64)
        self.assertEqual(int((match['black_agents']==1).sum()),64)
        with self.assertRaises(ValueError):play_individual(self.models,1,2,np.random.default_rng(0),opponents=[0,1])

    def test_reward_mapping_preserves_negative_losses_zero_draws_and_original_margins(self):
        margins=np.array([-64,-2,0,1,64])
        original=margins.copy()
        np.testing.assert_array_equal(training_rewards(margins,'margin'),original)
        np.testing.assert_array_equal(training_rewards(margins,'signed_win'),[-1,-1,0,1,1])
        np.testing.assert_array_equal(margins,original)
        with self.assertRaises(ValueError):training_rewards(margins,'win_only')


class IndividualIntegrationTest(unittest.TestCase):
    def run_trainer(self,directory,updates,resume=False,extra=(),extra_args=()):
        import individual_train
        real=individual_train.play_individual
        def controlled(*args,**kwargs):
            match=real(*args,**kwargs)
            if not kwargs.get('collect',False):match['margins'][:]=0
            return match
        argv=['individual_train.py','--out',directory,'--models','3','--games','4',
              '--record-games','4','--record-every','1','--patience','2','--eval-games','2',
              '--eval-every','100','--save-every','1','--width','16','--layers','1','--heads','2',
              '--grad-chunk','16','--initial-percent','10','--device','cpu','--updates',str(updates)]
        if resume:argv+=['--resume',str(Path(directory)/'latest.pt')]
        argv+=list(extra_args)
        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(patch('sys.argv',argv));stack.enter_context(patch('sys.stdout',io.StringIO()))
            stack.enter_context(patch.object(individual_train,'play_individual',side_effect=controlled))
            for item in extra:stack.enter_context(item)
            individual_train.main()
        return torch.load(Path(directory)/'latest.pt',map_location='cpu',weights_only=False)

    def test_switch_keeps_learned_weights_optimizer_and_baselines_and_opponents_frozen(self):
        with tempfile.TemporaryDirectory() as d:
            s=self.run_trainer(d,2)
            initial=torch.load(Path(d)/'initial.pt',weights_only=False)['models']
            event=s['controller']['events'][0];old=event['from']
            self.assertFalse(event['rollback']);self.assertEqual(s['mask']['percent'],101)
            self.assertNotEqual(old,s['controller']['learner'])
            self.assertTrue(any(not torch.equal(w,initial[old][n]) for n,w in s['models'][old].items()))
            self.assertTrue(any(v['step'].item()==2 for v in s['optimizers'][old]['state'].values()))
            self.assertEqual(s['learner_steps'][old],2)
            for i in set(range(3))-{old}:
                for n,w in s['models'][i].items():torch.testing.assert_close(w,initial[i][n],atol=0,rtol=0)
                self.assertEqual(s['optimizers'][i]['state'],{})
                self.assertEqual(s['baselines'][i],[0.,0.])
            self.assertFalse((Path(d)/'best.pt').exists())
            config=json.loads((Path(d)/'config.json').read_text(encoding='utf-8'))
            self.assertFalse(config['sharing_enabled']);self.assertFalse(config['alien_learning_enabled'])
            self.assertFalse(config['selection']['rollback'])

    def test_reselection_and_all_learning_states_resume_exactly(self):
        with tempfile.TemporaryDirectory() as d,tempfile.TemporaryDirectory() as ref:
            self.run_trainer(d,2);actual=self.run_trainer(d,4,True);expected=self.run_trainer(ref,4)
            self.assertEqual(actual['controller'],expected['controller'])
            self.assertEqual([e['step'] for e in actual['controller']['events']],[2,4])
            self.assertEqual(actual['mask']['percent'],102)
            self.assertEqual(actual['training_games'],16)
            self.assertEqual(actual['baselines'],expected['baselines'])
            for a,b in zip(actual['models'],expected['models']):
                for n,w in a.items():torch.testing.assert_close(w,b[n],atol=0,rtol=0)
            for a,b in zip(actual['optimizers'],expected['optimizers']):
                self.assertEqual(a['param_groups'],b['param_groups'])
                for i,entry in a['state'].items():
                    for n,w in entry.items():torch.testing.assert_close(w,b['state'][i][n],atol=0,rtol=0)

    def test_stop_after_update_saves_complete_step_without_repeating_on_resume(self):
        import individual_train
        real=individual_train.update_policy
        with tempfile.TemporaryDirectory() as d,tempfile.TemporaryDirectory() as ref:
            def stop(*args,**kwargs):
                result=real(*args,**kwargs);(Path(d)/'STOP').touch();return result
            s=self.run_trainer(d,2,extra=(patch.object(individual_train,'update_policy',side_effect=stop),))
            self.assertEqual(s['step'],1);self.assertEqual(s['training_games'],4)
            (Path(d)/'STOP').unlink()
            actual=self.run_trainer(d,2,True);expected=self.run_trainer(ref,2)
            self.assertEqual(actual['training_games'],8)
            self.assertEqual(actual['controller'],expected['controller'])
            for a,b in zip(actual['models'],expected['models']):
                for n,w in a.items():torch.testing.assert_close(w,b[n],atol=0,rtol=0)

    def test_reward_comparison_has_identical_initial_models_and_games_but_distinct_rewards(self):
        import individual_train
        real=individual_train.update_policy
        captured=[]
        def capture(*args,**kwargs):
            captured.append(np.asarray(args[3]).copy())
            return real(*args,**kwargs)
        with tempfile.TemporaryDirectory() as margin,tempfile.TemporaryDirectory() as binary:
            a=self.run_trainer(margin,1,extra=(patch.object(individual_train,'update_policy',side_effect=capture),))
            b=self.run_trainer(binary,1,extra=(patch.object(individual_train,'update_policy',side_effect=capture),),extra_args=('--reward','signed_win'))
            x=torch.load(Path(margin)/'initial.pt',weights_only=False)['models']
            y=torch.load(Path(binary)/'initial.pt',weights_only=False)['models']
            for left,right in zip(x,y):
                for n,w in left.items():torch.testing.assert_close(w,right[n],atol=0,rtol=0)
            self.assertEqual(a['controller']['initial_learner'],b['controller']['initial_learner'])
            self.assertEqual(a['latest_training']['opponents'],b['latest_training']['opponents'])
            self.assertEqual(a['latest_training']['wins'],b['latest_training']['wins'])
            np.testing.assert_array_equal(captured[1],np.sign(captured[0]))
            row=b['latest_training']
            self.assertEqual(row['mean_reward'],(row['wins']-row['losses'])/row['games'])
            self.assertEqual(row['reward_mode'],'signed_win')
            self.assertTrue(set(captured[1]).issubset({-1,0,1}))

    def test_binary_resume_matches_continuous_and_more_diagnostic_games_do_not_change_training(self):
        with tempfile.TemporaryDirectory() as d,tempfile.TemporaryDirectory() as ref:
            self.run_trainer(d,1,extra_args=('--reward','signed_win'))
            actual=self.run_trainer(d,2,True,extra_args=('--reward','signed_win','--eval-games','4'))
            expected=self.run_trainer(ref,2,extra_args=('--reward','signed_win'))
            self.assertEqual(actual['evaluation']['individuals'][0]['random']['games'],4)
            self.assertEqual(actual['controller'],expected['controller'])
            self.assertEqual(actual['baselines'],expected['baselines'])
            for a,b in zip(actual['models'],expected['models']):
                for n,w in a.items():torch.testing.assert_close(w,b[n],atol=0,rtol=0)
            from contextlib import redirect_stderr
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                self.run_trainer(d,3,True,extra_args=('--eval-games','4'))


if __name__=='__main__':unittest.main(verbosity=2)
