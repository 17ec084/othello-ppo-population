import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from entropy import normalized_entropy, future_entropy_returns
from model import masked_log_probs
from train import update_policy


class EntropyTest(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_legal_uniform_forced_and_concentrated_policies(self):
        legal=torch.tensor([[True,True,False],[True,False,False],[True,True,False]])
        logits=torch.tensor([[0.,0.,99.],[3.,99.,99.],[30.,-30.,99.]],requires_grad=True)
        h,n=normalized_entropy(masked_log_probs(logits,legal),legal)
        torch.testing.assert_close(n,torch.tensor([1.,0.,0.]),atol=1e-12,rtol=1e-6)
        h.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad[~legal]==0).all())

    def test_future_rewards_follow_each_trajectory_and_exclude_current_entropy(self):
        future,total=future_entropy_returns([.2,.6,.3,.1,.5],np.array([0,1,0,1,0]),2)
        np.testing.assert_allclose(future,[.8,.1,.5,0,0])
        np.testing.assert_allclose(total,[1.,.7])

    def test_entropy_bonus_flattens_a_policy_with_zero_stone_reward(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.logits=torch.nn.Parameter(torch.tensor([2.,-2.,10.]))
            def forward(self,t):return self.logits[None].expand(len(t),-1)
        model=Toy();optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        rows=np.zeros((2,8),dtype=np.int64);legal=np.array([[True,True,False]]*2)
        records=[(rows,legal,np.array([0,0]),np.array([0,1]))]
        old=model.logits.detach().clone()
        result=update_policy(model,optimizer,records,np.zeros(2),np.array([0,1]),0,
                             [0.,0.],entropy_coef=1.)
        self.assertLess(float((model.logits[0]-model.logits[1]).detach()),float(old[0]-old[1]))
        self.assertEqual(float(model.logits[2].detach()),float(old[2]))
        self.assertGreater(result['mean_entropy_bonus'],0)
        self.assertEqual(result['mean_reward'],0)

    def run_trainer(self,path,updates,resume=False):
        import alternating_train
        real=alternating_train.play_batch
        def drawn_evaluation(*args,**kwargs):
            result=real(*args,**kwargs)
            if not kwargs.get('collect'):
                result['rewards'][0][:]=0;result['rewards'][1][:]=0
            return result
        argv=['entropy_train.py','--out',path,'--width','16','--layers','1','--heads','2',
              '--device','cpu','--games-per-batch','4','--test-games','4','--eval-games','2',
              '--entropy-positions','8','--entropy-coef','.5','--initial-percent','10',
              '--patience','3','--eval-every','100','--updates',str(updates)]
        if resume:argv+=['--resume',str(Path(path)/'latest.pt')]
        with patch('sys.argv',argv),patch('sys.stdout',io.StringIO()), \
             patch.object(alternating_train,'play_batch',side_effect=drawn_evaluation):
            alternating_train.main(entropy_mode=True)
        return torch.load(Path(path)/'latest.pt',map_location='cpu',weights_only=False)

    def test_new_run_freezes_opponent_and_closed_weights_entropy_does_not_enter_test(self):
        with tempfile.TemporaryDirectory() as d:
            saved=self.run_trainer(d,1)
            initial=torch.load(Path(d)/'initial.pt',map_location='cpu',weights_only=False)['transformer']
            self.assertEqual(saved['protocol'],'alternating-entropy-margin-v1')
            for name,w in saved['models'][1].items():
                self.assertTrue(torch.equal(w,initial[1][name]))
            from model import ModelConfig,OthelloGPT
            from curriculum import ProgressiveUnfreezer
            mask=ProgressiveUnfreezer(OthelloGPT(ModelConfig(16,1,2)),saved['mask']['seed'],
                                      initial_percent=10,granularity=10,min_hold_updates=0)
            mask.load_state_dict(saved['mask'])
            for name,w in saved['models'][0].items():
                self.assertTrue(torch.equal(w[~mask.masks[name]],initial[0][name][~mask.masks[name]]))
            progress=json.loads((Path(d)/'progress.json').read_text(encoding='utf-8'))
            self.assertEqual(progress['alternating']['test']['p_value'],1)
            self.assertEqual(progress['alternating']['test']['draws'],4)
            row=json.loads((Path(d)/'metrics.jsonl').read_text(encoding='utf-8').splitlines()[0])
            self.assertGreater(row['agents']['A']['mean_entropy_bonus'],0)
            self.assertEqual(saved['learner_steps'],[1,0])

    def test_resume_matches_continuous_training_and_switches_and_releases(self):
        with tempfile.TemporaryDirectory() as interrupted,tempfile.TemporaryDirectory() as continuous:
            self.run_trainer(interrupted,3)
            resumed=self.run_trainer(interrupted,6,True)
            expected=self.run_trainer(continuous,6)
            self.assertEqual(resumed['learner_steps'],[3,3])
            self.assertEqual(resumed['mask']['percent'],101)
            for actual,wanted in zip(resumed['models'],expected['models']):
                for name,w in actual.items():
                    torch.testing.assert_close(w,wanted[name],atol=0,rtol=0)


if __name__=='__main__':unittest.main(verbosity=2)
