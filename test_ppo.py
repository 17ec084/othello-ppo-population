import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

import compact_checkpoint
from curriculum import ProgressiveUnfreezer
from individual_train import experiment,parse_args,play_individual
from model import ModelConfig,OthelloGPT
from ppo import gae,clipped_policy_loss,update_ppo,value_head


class PPOMathTest(unittest.TestCase):
    def test_interleaved_trajectories_terminal_on_last_own_action(self):
        # game 0: values [1,2,3], reward 8; game 1: [4,5], reward -2.
        values=np.array([1,4,2,5,3]);ids=np.array([0,1,0,1,0])
        a,t=gae(values,ids,[8,-2],lam=1)
        np.testing.assert_allclose(a,[7,-6,6,-7,5]);np.testing.assert_allclose(t,[8,-2,8,-2,8])
        a,t=gae(values,ids,[8,-2],lam=0)
        np.testing.assert_allclose(a,[1,1,1,-7,5]);np.testing.assert_allclose(t,[2,5,3,-2,8])

    def test_clipping_blocks_only_improvements_beyond_clip(self):
        logp=torch.log(torch.tensor([1.5,.5,1.5,.5],requires_grad=True))
        losses=clipped_policy_loss(logp,torch.zeros(4),torch.tensor([1.,1.,-1.,-1.]),.2)
        torch.testing.assert_close(losses,torch.tensor([-1.2,-.5,1.5,.8]))
        grad=torch.autograd.grad(losses.sum(),logp)[0]
        torch.testing.assert_close(grad,torch.tensor([0.,-.5,1.5,0.]))

    def test_value_head_does_not_advance_rng_or_change_policy(self):
        torch.manual_seed(31);model=OthelloGPT(ModelConfig(16,1,2))
        rng=torch.get_rng_state().clone();before=copy.deepcopy(model.state_dict());critic=value_head(16,'cpu')
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for n,w in model.state_dict().items():self.assertTrue(torch.equal(w,before[n]))
        self.assertTrue(torch.equal(critic(torch.ones(3,16)),torch.zeros(3,1)))

    def test_update_preserves_frozen_actor_and_opponents_and_trains_critic(self):
        torch.set_num_threads(1);torch.manual_seed(9)
        models=[OthelloGPT(ModelConfig(16,1,2)) for _ in range(3)]
        before=[copy.deepcopy(m.state_dict()) for m in models]
        critic=value_head(16,'cpu');mask=ProgressiveUnfreezer(models[0],5,initial_percent=10,granularity=10,min_hold_updates=0)
        optimizer=torch.optim.Adam(list(models[0].parameters())+list(critic.parameters()),lr=.001)
        match=play_individual(models,0,4,np.random.default_rng(5),collect=True)
        result=update_ppo(models[0],critic,optimizer,match['records'],np.array([20,-3,5,-2]),
                          np.random.default_rng(6),mask,epochs=2,minibatch=64,chunk_size=16)
        self.assertGreater(result['optimizer_steps'],2)
        self.assertTrue(any(bool(p.abs().sum()) for p in critic.parameters()))
        self.assertTrue(any(not torch.equal(w,before[0][n]) for n,w in models[0].state_dict().items()))
        for n,w in models[0].state_dict().items():torch.testing.assert_close(w[~mask.masks[n]],before[0][n][~mask.masks[n]],atol=0,rtol=0)
        for i in (1,2):
            for n,w in models[i].state_dict().items():self.assertTrue(torch.equal(w,before[i][n]))


class PairedIntegrationTest(unittest.TestCase):
    def run_method(self,d,method,steps,resume=False,compact=True,base=None):
        cli=['--out',d,'--algorithm',method,'--models','3','--games','4','--record-games','4',
             '--record-every','1','--patience','2','--eval-games','2','--eval-every','100','--save-every','1',
             '--width','16','--layers','1','--heads','2','--initial-percent','10','--device','cpu',
             '--updates',str(steps),'--ppo-epochs','2','--ppo-minibatch','64']
        if compact:cli+=['--compact-checkpoints']
        if base:cli+=['--initial-checkpoint',str(base)]
        if resume:cli+=['--resume',str(Path(d)/('latest.pt.gz' if compact else 'latest.pt'))]
        import individual_train
        real=individual_train.play_individual
        def controlled(*args,**kwargs):
            match=real(*args,**kwargs)
            if not kwargs.get('collect',False):match['margins'][:]=0
            return match
        with patch.object(individual_train,'play_individual',side_effect=controlled):
            for _ in experiment(parse_args(cli)):pass
        initial=torch.load(base or Path(d)/'initial.pt',weights_only=False)['models']
        return (compact_checkpoint.load(Path(d)/'latest.pt.gz',initial) if compact else
                torch.load(Path(d)/'latest.pt',weights_only=False))

    def test_same_initial_policies_first_games_opponents_and_rewards(self):
        with tempfile.TemporaryDirectory() as root:
            a=self.run_method(str(Path(root)/'reinforce'),'reinforce',1,base=Path(root)/'initial.pt')
            b=self.run_method(str(Path(root)/'ppo'),'ppo',1,base=Path(root)/'initial.pt')
            self.assertEqual(a['initial_policy_sha256'],b['initial_policy_sha256'])
            self.assertEqual(a['latest_training']['rollout_sha256'],b['latest_training']['rollout_sha256'])
            self.assertEqual(a['latest_training']['opponents'],b['latest_training']['opponents'])
            self.assertEqual(a['latest_training']['mean_reward'],b['latest_training']['mean_reward'])
            self.assertEqual(a['controller']['initial_learner'],b['controller']['initial_learner'])

    def test_ppo_sparse_resume_matches_continuous_across_reselection(self):
        with tempfile.TemporaryDirectory() as d,tempfile.TemporaryDirectory() as ref:
            self.run_method(d,'ppo',2)
            a=self.run_method(d,'ppo',4,True);b=self.run_method(ref,'ppo',4)
            self.assertEqual(a['controller'],b['controller']);self.assertEqual(a['ppo_rng'],b['ppo_rng'])
            self.assertEqual(a['mask']['percent'],102)
            for key in ('models','critics'):
                for x,y in zip(a[key],b[key]):
                    for name,w in x.items():torch.testing.assert_close(w,y[name],atol=0,rtol=0)
            for x,y in zip(a['optimizers'],b['optimizers']):
                self.assertEqual(x['param_groups'],y['param_groups'])
                for pid,state in x['state'].items():
                    for name,w in state.items():torch.testing.assert_close(w,y['state'][pid][name],atol=0,rtol=0)

    def test_sparse_and_full_checkpoint_restore_identical_learning(self):
        with tempfile.TemporaryDirectory() as d,tempfile.TemporaryDirectory() as ref:
            a=self.run_method(d,'ppo',2);b=self.run_method(ref,'ppo',2,compact=False)
            for x,y in zip(a['models'],b['models']):
                for name,w in x.items():torch.testing.assert_close(w,y[name],atol=0,rtol=0)
            self.assertLess((Path(d)/'latest.pt.gz').stat().st_size,(Path(ref)/'latest.pt').stat().st_size)

    def test_low_disk_preserves_complete_checkpoint_without_creating_temp(self):
        with tempfile.TemporaryDirectory() as d:
            state=self.run_method(d,'ppo',1)
            path=Path(d)/'latest.pt.gz';before=path.read_bytes()
            model=OthelloGPT(ModelConfig(16,1,2));mask=ProgressiveUnfreezer(model,20261523,initial_percent=10,granularity=10,min_hold_updates=0)
            usage=type('Usage',(),{'free':0})()
            with patch('compact_checkpoint.shutil.disk_usage',return_value=usage),self.assertRaises(OSError):
                compact_checkpoint.save(path,state,mask.masks)
            self.assertEqual(path.read_bytes(),before);self.assertFalse(Path(str(path)+'.tmp').exists())

    def test_save_pressure_waits_then_retries_same_pending_step(self):
        real=compact_checkpoint.save;calls=[]
        def pressure(path,data,masks,*args,**kwargs):
            calls.append(data['step'])
            if len(calls)==1:raise OSError(28,'simulated full disk')
            return real(path,data,masks,*args,**kwargs)
        with tempfile.TemporaryDirectory() as d:
            with patch('compact_checkpoint.save',side_effect=pressure),patch('individual_train.time.sleep') as wait,patch('sys.stderr',io.StringIO()):
                result=self.run_method(d,'ppo',1)
            self.assertEqual(calls[:2],[0,0]);wait.assert_called_once_with(5)
            self.assertEqual(result['step'],1)
            self.assertEqual(json.loads((Path(d)/'progress.json').read_text(encoding='utf8'))['status'],'completed')

    def test_partial_compressed_write_keeps_previous_file_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as d:
            state=self.run_method(d,'ppo',1);path=Path(d)/'latest.pt.gz';before=path.read_bytes()
            model=OthelloGPT(ModelConfig(16,1,2));mask=ProgressiveUnfreezer(model,20261523,initial_percent=10,granularity=10,min_hold_updates=0)
            def failure(data,stream):stream.write(b'partial');raise OSError(28,'disk full')
            with patch('compact_checkpoint.torch.save',side_effect=failure),self.assertRaises(OSError):
                compact_checkpoint.save(path,state,mask.masks)
            self.assertEqual(path.read_bytes(),before);self.assertFalse(Path(str(path)+'.tmp').exists())

    def test_modified_initial_checkpoint_is_rejected_before_resume(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_method(d,'ppo',1)
            initial=torch.load(Path(d)/'initial.pt',weights_only=False)['models']
            initial[0]['position'].view(-1)[0]+=1
            with self.assertRaisesRegex(ValueError,'hash differs'):
                compact_checkpoint.load(Path(d)/'latest.pt.gz',initial)


if __name__=='__main__':unittest.main(verbosity=2)
