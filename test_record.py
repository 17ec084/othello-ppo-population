import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from curriculum import ProgressiveUnfreezer
from model import ModelConfig,OthelloGPT
from record_control import RecordController


class RecordControlTest(unittest.TestCase):
    def test_strict_records_draw_half_and_full_200_steps(self):
        c=RecordController(200)
        self.assertTrue(c.observe(0,np.array([[1,0],[-1,1]])))
        self.assertEqual(c.best['score'],.625)
        self.assertFalse(c.observe(10,np.array([[1,0],[-1,1]])))
        self.assertEqual(c.best['step'],0)
        self.assertFalse(c.expired(199));self.assertTrue(c.expired(200))
        self.assertTrue(c.observe(200,np.ones((2,2))))
        self.assertFalse(c.expired(200))
        self.assertEqual(c.best['step'],200)

    def test_switch_resets_opponent_comparison_and_only_B_to_A_opens(self):
        mask=ProgressiveUnfreezer(OthelloGPT(ModelConfig(16,1,2)),7,
                                  initial_percent=.1,granularity=10,min_hold_updates=0)
        c=RecordController(2);c.observe(0,np.zeros((3,2)))
        e=c.switch(2,mask)
        self.assertEqual(e['restored_step'],0);self.assertFalse(e['released'])
        self.assertIsNone(c.best);self.assertEqual(c.learner,1)
        c.observe(2,np.full((3,2),-1))
        self.assertEqual(c.best['score'],1.)
        e=c.switch(4,mask)
        self.assertTrue(e['released']);self.assertEqual(mask.summary()['percent'],.2)
        saved=c.state_dict();loaded=RecordController(2);loaded.load_state_dict(saved)
        self.assertEqual(c.summary(),loaded.summary())


class RecordIntegrationTest(unittest.TestCase):
    def run_trainer(self,directory,updates,resume=False,extras=(),sequence=None,extra_args=()):
        import record_train
        real=record_train.record_evaluation
        calls=0
        def evaluate(*args,**kwargs):
            nonlocal calls
            margins,pairings=real(*args,**kwargs)
            margins[:]=sequence[calls] if sequence else 0
            calls+=1
            return margins,pairings
        argv=['record_train.py','--out',directory,'--population','3','--width','16','--layers','1',
            '--heads','2','--games-per-pair','2','--record-games-per-pair','2','--record-every','1',
            '--eval-games-per-model','2','--eval-every','100','--teacher-games-per-opponent','2',
            '--sharing-rounds','2','--sharing-positions','4','--diversity-positions','8',
            '--initial-percent','10','--patience','2','--device','cpu','--updates',str(updates)]
        if resume:argv+=['--resume',str(Path(directory)/'latest.pt')]
        argv+=list(extra_args)
        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(patch('sys.argv',argv));stack.enter_context(patch('sys.stdout',io.StringIO()))
            stack.enter_context(patch.object(record_train,'record_evaluation',side_effect=evaluate))
            for extra in extras:stack.enter_context(extra)
            record_train.main()
        return torch.load(Path(directory)/'latest.pt',map_location='cpu',weights_only=False)

    def test_frozen_opponent_shared_updates_and_no_statistical_or_entropy_state(self):
        with tempfile.TemporaryDirectory() as d:
            s=self.run_trainer(d,1)
            initial=torch.load(Path(d)/'initial.pt',map_location='cpu',weights_only=False)['groups']
            for name,value in s['groups'][1]['weights'].items():
                torch.testing.assert_close(value,initial[1][name],atol=0,rtol=0)
            self.assertTrue(any(not torch.equal(v,initial[0][n]) for n,v in s['groups'][0]['weights'].items()))
            self.assertEqual(s['sharing_totals'],[2,0]);self.assertEqual(s['alien_totals'][1],0)
            self.assertEqual(s['training_games'],6);self.assertEqual(s['record_games'],36)
            self.assertNotIn('test',s['controller']);self.assertNotIn('loss_test',s['controller'])
            config=json.loads((Path(d)/'config.json').read_text(encoding='utf-8'))
            self.assertFalse(config['entropy_reward_enabled']);self.assertFalse(config['p_values_enabled'])
            self.assertFalse(config['streaks_enabled'])
            self.assertFalse(config['alien_learning_enabled'])
            self.assertFalse(config['diversity']['enabled'])
            self.assertIsNone(config['diversity']['reward'])
            self.assertEqual(config['matchmaking'],'per_game_permutation')
            rows=[json.loads(x) for x in (Path(d)/'metrics.jsonl').read_text().splitlines()]
            pairing=np.asarray(rows[0]['pairing'])
            self.assertEqual(pairing.shape,(3,2))
            for column in pairing.T:self.assertEqual(sorted(column),[0,1,2])
            evaluations=[json.loads(x) for x in (Path(d)/'record_evaluations.jsonl').read_text().splitlines()]
            for e in evaluations:
                for pairing in e['pairings']:self.assertEqual(sorted(pairing),[0,1,2])
                for i in range(3):self.assertEqual({p[i] for p in e['pairings']},{0,1,2})

    def test_rollback_restores_entire_population_all_optimizers_and_reward_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            s=self.run_trainer(d,3,sequence=[0,1,0,0],extra_args=('--alien-learning',))
            best=torch.load(Path(d)/'best.pt',map_location='cpu',weights_only=False)
            self.assertEqual(best['step'],1)
            self.assertEqual(s['controller']['learner'],1)
            event=s['controller']['events'][0]
            self.assertEqual(event['step'],3);self.assertEqual(event['restored_step'],1)
            for name,value in s['groups'][0]['weights'].items():
                torch.testing.assert_close(value,best['group']['weights'][name],atol=0,rtol=0)
            np.testing.assert_array_equal(s['groups'][0]['baseline'],best['group']['baseline'])
            for key in ('competition_optimizer','sharing_optimizer','alien_optimizer'):
                actual,wanted=s['groups'][0][key],best['group'][key]
                self.assertEqual(actual['param_groups'],wanted['param_groups'])
                for index,values in actual['state'].items():
                    for name,value in values.items():
                        torch.testing.assert_close(value,wanted['state'][index][name],atol=0,rtol=0)

    def test_200_rule_switch_resume_and_B_to_A_release(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_trainer(d,2)
            s=self.run_trainer(d,4,True)
            self.assertEqual([e['step'] for e in s['controller']['events']],[2,4])
            self.assertEqual(s['learner_steps'],[2,2])
            self.assertEqual(s['mask']['percent'],101)
            initial=torch.load(Path(d)/'initial.pt',map_location='cpu',weights_only=False)['groups']
            for g,expected in zip(s['groups'],initial):
                for n,v in g['weights'].items():torch.testing.assert_close(v,expected[n],atol=0,rtol=0)
                self.assertTrue((g['baseline']==0).all())
                for key in ('competition_optimizer','sharing_optimizer','alien_optimizer'):
                    self.assertEqual(len(g[key]['state']),0)

    def test_partial_sharing_resume_matches_continuous_training(self):
        import record_train
        real=record_train.sharing_update
        with tempfile.TemporaryDirectory() as d,tempfile.TemporaryDirectory() as reference:
            def stop_after_round(*args,**kwargs):
                result=real(*args,**kwargs)
                (Path(d)/'STOP').touch()
                return result
            partial=self.run_trainer(d,1,extras=(patch.object(record_train,'sharing_update',side_effect=stop_after_round),))
            self.assertEqual(partial['step'],0)
            self.assertEqual(partial['pending']['sharing_round'],1)
            (Path(d)/'STOP').unlink()
            resumed=self.run_trainer(d,1,True);expected=self.run_trainer(reference,1)
            self.assertEqual(resumed['training_games'],6)
            self.assertEqual(resumed['teacher_games'],12)
            self.assertEqual(resumed['sharing_totals'],expected['sharing_totals'])
            for actual,wanted in zip(resumed['groups'],expected['groups']):
                for n,v in actual['weights'].items():torch.testing.assert_close(v,wanted['weights'][n],atol=0,rtol=0)

    def test_disabled_alien_never_updates_even_below_initial_diversity_after_resume(self):
        import record_train
        real=record_train.population_diversity
        calls=0
        def below_initial(*args,**kwargs):
            nonlocal calls
            calls+=1
            result=real(*args,**kwargs)
            if calls>2:result['value']=0.
            return result
        with tempfile.TemporaryDirectory() as d:
            with patch.object(record_train,'population_diversity',side_effect=below_initial), \
                 patch.object(record_train,'alien_update',side_effect=AssertionError('Alien update is disabled')) as alien:
                self.run_trainer(d,2,extra_args=('--patience','200'))
                s=self.run_trainer(d,3,True,extra_args=('--patience','200'))
                alien.assert_not_called()
            self.assertEqual(s['step'],3)
            self.assertEqual(s['alien_totals'],[0,0])
            self.assertEqual(s['diversity'][0]['value'],0.)
            self.assertGreater(s['targets'][0],0.)
            self.assertEqual(s['sharing_totals'],[6,0])
            self.assertEqual(s['training_games'],18)
            for group in s['groups']:self.assertEqual(group['alien_optimizer']['state'],{})
            self.assertFalse((Path(d)/'alien_events.jsonl').exists())
            self.assertFalse((Path(d)/'alien_sessions.jsonl').exists())
            initial=torch.load(Path(d)/'initial.pt',map_location='cpu',weights_only=False)['groups']
            for n,w in s['groups'][1]['weights'].items():
                torch.testing.assert_close(w,initial[1][n],atol=0,rtol=0)


if __name__=='__main__':unittest.main(verbosity=2)
