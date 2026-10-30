import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from adaptive import AdaptiveController, copy_opponent
from curriculum import ProgressiveUnfreezer
from monitor import timeline_events


class AdaptiveTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.models = [torch.nn.Linear(4, 2) for _ in range(2)]
        self.mask = ProgressiveUnfreezer(self.models[0], 15, initial_percent=.1,
                                        min_hold_updates=0, granularity=10)

    def test_switch_before_50_not_49_and_only_B_to_A_releases(self):
        c = AdaptiveController()
        weights = copy.deepcopy([m.state_dict() for m in self.models])
        for step in range(1, 50):
            self.assertIsNone(c.before_step(step, self.mask))
            self.assertIsNone(c.observe(step, [0, 0], self.mask))
        event = c.before_step(50, self.mask)
        self.assertEqual((event['from'], event['to'], event['timing']), ('A', 'B', 'before_update'))
        self.assertEqual(event['reason'], 'plateau_50')
        self.assertFalse(event['released'])
        self.assertEqual(c.t, 50)
        self.assertIsNone(c.before_step(99, self.mask))
        self.assertTrue(c.before_step(100, self.mask)['released'])
        self.assertEqual(self.mask.summary()['percent'], .2)
        for m, old in zip(self.models, weights):
            for name, value in m.state_dict().items():
                self.assertTrue(torch.equal(value, old[name]))

    def test_improvement_restarts_clock_and_win_switches_normally(self):
        c = AdaptiveController()
        self.assertIsNone(c.observe(20, [1, 1], self.mask))
        self.assertEqual(c.t, 20)
        self.assertLess(c.anchor_p, 1)
        self.assertIsNone(c.before_step(69, self.mask))
        self.assertEqual(c.before_step(70, self.mask)['to'], 'B')
        event = c.observe(71, np.full(200, -16), self.mask)
        self.assertEqual(event['reason'], 'p_threshold')
        self.assertEqual(c.learner, 0)
        self.assertEqual(c.loss_test.p_value, 1)

    def test_inferiority_uses_opposite_direction_and_keeps_learner(self):
        for learner, margins in ((0, -16), (1, 16)):
            c = AdaptiveController()
            c.learner = learner
            event = c.observe(2, np.full(200, margins), self.mask)
            self.assertEqual(event['type'], 'opponent_copy')
            self.assertLessEqual(event['loss_test']['p_value'], .05)
            self.assertGreater(event['test']['p_value'], .05)
            self.assertEqual(c.learner, learner)
            self.assertEqual((c.test.p_value, c.loss_test.p_value, c.t), (1, 1, 2))
            self.assertEqual(c.events, [])
            self.assertIsNone(c.before_step(51, self.mask))
            self.assertIsNotNone(c.before_step(52, self.mask))

    def test_copy_includes_frozen_values_resets_only_learner_and_preserves_mask(self):
        optimizers = [torch.optim.Adam(m.parameters()) for m in self.models]
        source = copy.deepcopy(self.models[1].state_dict())
        for optimizer, model in zip(optimizers, self.models):
            model(torch.ones(1, 4)).sum().backward()
            optimizer.step()
        source = copy.deepcopy(self.models[1].state_dict())
        opponent_state = copy.deepcopy(optimizers[1].state_dict())
        masks = copy.deepcopy(self.mask.masks)
        baselines = [[9., -2.], [3., 4.]]
        copy_opponent(self.models, optimizers, baselines, 0)
        for name, value in self.models[0].state_dict().items():
            self.assertTrue(torch.equal(value, source[name]))
            self.assertTrue(torch.equal(self.models[1].state_dict()[name], source[name]))
            self.assertTrue(torch.equal(self.mask.masks[name], masks[name]))
        self.assertEqual(len(optimizers[0].state), 0)
        self.assertEqual(len(optimizers[1].state), len(opponent_state['state']))
        self.assertEqual(baselines, [[0., 0.], [3., 4.]])

    def test_resume_preserves_both_tests_and_clock(self):
        c = AdaptiveController()
        c.observe(5, [1, 1, -1, -1, 0, 0], self.mask)
        restored = AdaptiveController()
        restored.load_state_dict(copy.deepcopy(c.state_dict()))
        self.assertEqual(c.summary(), restored.summary())
        self.assertEqual(c.plateau_summary(40), restored.plateau_summary(40))
        c.observe(6, [-1, -1], self.mask)
        restored.observe(6, [-1, -1], self.mask)
        self.assertEqual(c.summary(), restored.summary())

    def test_timeline_keeps_mutations_copies_switch_order_and_ignores_partial_line(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'switches.jsonl').write_text(json.dumps({'step': 5, 'from': 'A', 'to': 'B',
                    'timing': 'before_update'})+'\n{', encoding='utf-8')
            (root/'plateau_events.jsonl').write_text('\n'.join(json.dumps(e) for e in
                    [{'step': 2, 'type': 'p_improved'}, {'step': 3, 'type': 'mutation'}]), encoding='utf-8')
            (root/'copies.jsonl').write_text('\n'.join(json.dumps(e) for e in
                    [{'step': 5, 'learner': 'B'}, {'step': 10, 'learner': 'A'}]), encoding='utf-8')
            events = timeline_events(root, 5)
            self.assertEqual([e['type'] for e in events], ['mutation', 'learner_switch', 'opponent_copy'])

    def run_trainer(self, directory, updates, outcome, patience=3, resume=False):
        import alternating_train
        real_play = alternating_train.play_batch
        def synthetic_evaluation(*args, **kwargs):
            result = real_play(*args, **kwargs)
            if not kwargs.get('collect', False):
                # Only held-out outcomes are synthetic; exercise real training.
                learner = 0 if next(args[0][0].parameters()).requires_grad else 1
                result['rewards'][0][:] = outcome * (1 if learner == 0 else -1)
                result['rewards'][1][:] = -result['rewards'][0]
            return result
        argv = ['alternating_train.py', '--out', directory, '--device', 'cpu', '--width', '16',
                '--layers', '1', '--heads', '2', '--games-per-batch', '2', '--test-games', '32',
                '--eval-games', '2', '--eval-every', '100', '--patience', str(patience), '--updates', str(updates)]
        checkpoint = Path(directory)/'latest.pt'
        if resume:
            argv += ['--resume', str(checkpoint)]
        with patch.object(alternating_train, 'play_batch', side_effect=synthetic_evaluation), \
             patch('sys.argv', argv), patch('sys.stdout', io.StringIO()):
            alternating_train.main()
        return torch.load(checkpoint, map_location='cpu', weights_only=False)

    def test_trainer_switch_before_training_resume_and_no_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_trainer(d, 4, 0)
            saved = self.run_trainer(d, 8, 0, resume=True)
            events = [json.loads(s) for s in (Path(d)/'switches.jsonl').read_text().splitlines()]
            rows = [json.loads(s) for s in (Path(d)/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual([e['step'] for e in events], [3, 6])
            self.assertEqual(rows[2]['learner'], 'B')
            self.assertEqual(rows[5]['learner'], 'A')
            self.assertEqual(saved['controller']['learner'], 0)
            self.assertEqual(saved['mask']['percent'], 2)
            self.assertFalse((Path(d)/'plateau_events.jsonl').exists())
            self.assertEqual(saved['protocol'], 'alternating-anytime-copy-v3')

    def test_trainer_exact_copy_reset_and_continued_same_learner(self):
        with tempfile.TemporaryDirectory() as d:
            saved = self.run_trainer(d, 3, -1, patience=50)
            events = [json.loads(s) for s in (Path(d)/'copies.jsonl').read_text().splitlines()]
            self.assertEqual([e['step'] for e in events], [1, 2, 3])
            self.assertEqual(saved['controller']['learner'], 0)
            self.assertEqual(saved['controller']['t'], 3)
            self.assertEqual(saved['controller']['test']['pairs'], 0)
            self.assertEqual(saved['controller']['loss_test']['pairs'], 0)
            self.assertEqual(len(saved['optimizers'][0]['state']), 0)
            self.assertEqual(saved['baselines'][0], [0., 0.])
            for name, value in saved['models'][0].items():
                self.assertTrue(torch.equal(value, saved['models'][1][name]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
