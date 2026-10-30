import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import compact_checkpoint
import individual_train
from individual_control import WinRateController
from individual_train import balanced_opponents,experiment,parse_args,play_individual
from model import ModelConfig,OthelloGPT


class ThresholdTest(unittest.TestCase):
    def test_exactly_55_percent_does_not_pass_draws_are_not_half_wins(self):
        c=WinRateController(3,.55,20,np.random.default_rng(1))
        other=np.flatnonzero(np.arange(3)!=c.learner);opp=np.repeat(other,20)
        margins=np.r_[np.ones(22),np.zeros(18)]
        self.assertFalse(c.observe(1,margins,opp));self.assertEqual(c.current['win_rate'],.55)
        margins[22]=1
        self.assertTrue(c.observe(2,margins,opp))
        self.assertLess(c.current['per_opponent'][1]['win_rate'],.55)
        previous=c.learner;event=c.reselect(2,np.random.default_rng(2))
        self.assertNotEqual(c.learner,previous);self.assertFalse(event['released']);self.assertFalse(event['rollback'])
        self.assertIsNone(c.current)
        state=c.state_dict();restored=WinRateController(3,.55,20,np.random.default_rng(9));restored.load_state_dict(state)
        self.assertEqual(c.summary(),restored.summary())

    def test_actual_2432_game_boundary_and_all_nineteen_opponents(self):
        rng=np.random.default_rng(2);c=WinRateController(20,.55,128,rng)
        opp=balanced_opponents(20,c.learner,128,rng)
        margins=-np.ones(2432);margins[:1337]=1
        self.assertFalse(c.observe(1,margins,opp))
        margins[1337]=1;self.assertTrue(c.observe(2,margins,opp))
        self.assertEqual(len(c.current['per_opponent']),19)
        self.assertTrue(all(r['games']==128 for r in c.current['per_opponent']))

    def test_pairing_has_exactly_64_games_in_each_color_against_every_opponent(self):
        for learner in (0,11,19):
            opp=balanced_opponents(20,learner,128,np.random.default_rng(learner))
            self.assertEqual(len(opp),2432)
            for color in (0,1):
                counts=np.bincount(opp[color::2],minlength=20)
                self.assertEqual(counts[learner],0)
                self.assertTrue(all(counts[i]==64 for i in range(20) if i!=learner))
        with self.assertRaises(ValueError):balanced_opponents(20,0,127,np.random.default_rng(0))

    def test_controller_rejects_unequal_coverage_and_self_opponents(self):
        rng=np.random.default_rng(2);c=WinRateController(3,.55,4,rng)
        other=np.flatnonzero(np.arange(3)!=c.learner)
        for opp in (np.repeat(other,[2,6]),np.full(8,c.learner)):
            with self.assertRaises(ValueError):c.observe(1,np.ones(8),opp)


class FullTrainingTest(unittest.TestCase):
    def test_learner_evaluation_every_step_keeps_frozen_individual_values_and_measurement_times(self):
        with tempfile.TemporaryDirectory() as folder:
            state,_=self.run_training(folder,3,outcomes=[-2,2,-2])
            history=[json.loads(line) for line in (Path(folder)/'evaluations.jsonl').read_text(encoding='utf8').splitlines()]
            self.assertEqual([r['step'] for r in history],[0,1,2,3])
            first=state['controller']['initial_learner'];second=state['controller']['learner']
            self.assertEqual([r['measured_ids'] for r in history[1:]],[[first],[first],[second]])
            old=[next(r for r in e['individuals'] if r['id']==first) for e in history]
            self.assertEqual(old[1]['random'],old[2]['random'])
            self.assertEqual(old[2],old[3])
            self.assertEqual(old[3]['evaluated_step'],2)
            for e in history:
                self.assertEqual(len(e['individuals']),3)
                for name in ('random','position','initial'):
                    self.assertEqual(e['aggregate'][name]['games'],6)
                    self.assertEqual(e['aggregate'][name]['wins'],sum(r[name]['wins'] for r in e['individuals']))

    def test_partial_diagnostic_preserves_actual_model_id_and_evaluation_seeds(self):
        torch.set_num_threads(1);torch.manual_seed(5)
        config=ModelConfig(16,1,2);models=[OthelloGPT(config) for _ in range(3)]
        initial=[copy.deepcopy(m.state_dict()) for m in models]
        full=individual_train.diagnostic(models,initial,config,4,222,5)
        single=individual_train.diagnostic(models,initial,config,4,222,5,indices=[2])
        self.assertEqual(single['individuals'],[full['individuals'][2]])
        self.assertEqual(single['measured_ids'],[2])

    def run_training(self,folder,steps,resume=False,outcomes=None):
        cli=['--selection','win_rate','--algorithm','ppo','--initial-percent','100',
             '--models','3','--games-per-opponent','4','--eval-games','2','--eval-every','100',
             '--save-every','1','--width','16','--layers','1','--heads','2',
             '--ppo-epochs','2','--ppo-minibatch','64','--grad-chunk','32',
             '--device','cpu','--updates',str(steps),'--compact-checkpoints','--out',folder]
        if resume:cli+=['--resume',str(Path(folder)/'latest.pt.gz')]
        real=individual_train.play_individual;rounds=0
        def controlled(*args,**kwargs):
            nonlocal rounds
            match=real(*args,**kwargs)
            if outcomes is not None:
                match['margins'][:]=outcomes[rounds];rounds+=1
            return match
        with patch.object(individual_train,'play_individual',side_effect=controlled):
            for _ in experiment(parse_args(cli)):pass
        initial=torch.load(Path(folder)/'initial.pt',weights_only=False)['models']
        saved=compact_checkpoint.load(Path(folder)/'latest.pt.gz',initial)
        return saved,initial

    def test_selected_model_and_critic_train_without_masks_and_other_models_stay_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            state,initial=self.run_training(folder,1,outcomes=[-2])
            learner=state['controller']['learner']
            self.assertIsNone(state['mask']);self.assertEqual(state['trained_model_ids'],[learner])
            self.assertEqual(state['training_games'],8)
            self.assertEqual(state['checkpoint_format'],compact_checkpoint.FULL_FORMAT)
            self.assertTrue(any(not torch.equal(w,initial[learner][n]) for n,w in state['models'][learner].items()))
            self.assertTrue(any(bool(v.abs().sum()) for v in state['critics'][learner].values()))
            for i in set(range(3))-{learner}:
                for n,w in state['models'][i].items():torch.testing.assert_close(w,initial[i][n],atol=0,rtol=0)
                self.assertEqual(state['optimizers'][i]['state'],{})
                self.assertTrue(all(not bool(v.abs().sum()) for v in state['critics'][i].values()))
            config=json.loads((Path(folder)/'config.json').read_text(encoding='utf8'))
            self.assertTrue(config['model']['all_parameters_trainable'])
            self.assertEqual(config['selection']['score'],'wins / games; draws are not wins')

    def test_passing_round_switches_and_preserves_the_tested_weights_without_extra_update(self):
        with tempfile.TemporaryDirectory() as folder:
            before,initial=self.run_training(folder,1,outcomes=[-2])
            after,_=self.run_training(folder,2,True,outcomes=[2])
            event=after['controller']['events'][0]
            self.assertEqual(event['step'],2);self.assertEqual(event['result']['win_rate'],1.)
            self.assertNotEqual(event['from'],event['to'])
            self.assertEqual(after['latest_training']['update']['optimizer_steps'],0)
            self.assertEqual(after['learner_steps'][event['from']],1)
            for a,b in zip(before['models'],after['models']):
                for name,w in a.items():torch.testing.assert_close(w,b[name],atol=0,rtol=0)

    def test_full_checkpoint_resume_matches_uninterrupted_training_and_switch_rng(self):
        with tempfile.TemporaryDirectory() as folder,tempfile.TemporaryDirectory() as ref:
            self.run_training(folder,1,outcomes=[-2])
            actual,_=self.run_training(folder,3,True,outcomes=[2,-2])
            expected,_=self.run_training(ref,3,outcomes=[-2,2,-2])
            self.assertEqual(actual['controller'],expected['controller'])
            self.assertEqual(actual['rngs'],expected['rngs']);self.assertEqual(actual['ppo_rng'],expected['ppo_rng'])
            for key in ('models','critics'):
                for a,b in zip(actual[key],expected[key]):
                    for name,w in a.items():torch.testing.assert_close(w,b[name],atol=0,rtol=0)
            for a,b in zip(actual['optimizers'],expected['optimizers']):
                self.assertEqual(a['param_groups'],b['param_groups'])
                for pid,values in a['state'].items():
                    for name,w in values.items():torch.testing.assert_close(w,b['state'][pid][name],atol=0,rtol=0)

    def test_passing_initial_round_never_changes_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            state,initial=self.run_training(folder,1,outcomes=[1])
            self.assertEqual(state['trained_model_ids'],[]);self.assertEqual(sum(state['learner_steps']),0)
            for a,b in zip(state['models'],initial):
                for name,w in a.items():torch.testing.assert_close(w,b[name],atol=0,rtol=0)


if __name__=='__main__':unittest.main(verbosity=2)
