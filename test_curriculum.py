"""Stone-difference reward, exact freezing, leader changes, and resume checks."""
import copy
import io
import unittest

import numpy as np
import torch

from curriculum import ProgressiveUnfreezer
from model import ModelConfig, OthelloGPT
from othello import initial_boards, legal_moves, bits_to_mask, row_tokens, terminal_rewards
from train import play_batch, summarize, update_policy


class CurriculumTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(24)
        self.model = OthelloGPT(ModelConfig(width=16, layers=1, heads=2))

    def test_masks_are_exact_nested_distributed_and_reach_100_percent(self):
        schedule = ProgressiveUnfreezer(self.model, 114)
        previous = {name: torch.zeros_like(mask) for name, mask in schedule.masks.items()}
        for percent in range(1, 101):
            schedule.percent = percent
            schedule._build_masks()
            self.assertEqual(sum(int(mask.sum()) for mask in schedule.masks.values()),
                             schedule.total * percent // 100)
            for name, mask in schedule.masks.items():
                self.assertFalse((previous[name] & ~mask).any(), name)
            previous = {name: mask.clone() for name, mask in schedule.masks.items()}
        self.assertEqual(schedule.summary()['frozen_parameters'], 0)
        start = ProgressiveUnfreezer(self.model, 114)
        # All layers get access at the first stage, including input and output.
        for prefix in ('embedding', 'blocks.0', 'head'):
            self.assertGreater(sum(int(m.sum()) for n, m in start.masks.items() if n.startswith(prefix)), 0)

    def test_15_observations_and_3_reversals_release_once_then_reset(self):
        schedule = ProgressiveUnfreezer(self.model, 114, min_hold_updates=0)
        # A -> B -> A -> B = 3 switches, but never release before 15 samples.
        sequence = ['A']*4 + ['B']*4 + ['A']*4 + ['B']*3
        for step, winner in enumerate(sequence, 1):
            released = schedule.observe(step, 70 if winner == 'A' else 58,
                                        58 if winner == 'A' else 70)
            self.assertEqual(released, step == 15)
        self.assertEqual(schedule.percent, 2)
        self.assertEqual(schedule.events[-1]['reversals'], 3)
        self.assertEqual(schedule.history, [])
        self.assertFalse(schedule.observe(16, 70, 58))
        self.assertEqual(len(schedule.history), 1)

    def test_ties_and_reversals_outside_window_do_not_trigger_release(self):
        schedule = ProgressiveUnfreezer(self.model, 114)
        for step in range(1, 31):
            schedule.observe(step, 64, 64)
        self.assertEqual(schedule.percent, 1)
        self.assertEqual(schedule.latest_reversals, 0)
        sequence = ['A', 'B'] + ['A']*13  # only two switches
        for step, winner in enumerate(sequence, 31):
            schedule.observe(step, 70 if winner == 'A' else 58, 58 if winner == 'A' else 70)
        self.assertEqual(schedule.percent, 1)
        for step in range(46, 61):
            schedule.observe(step, 70, 58)
        self.assertEqual(schedule.latest_reversals, 0)
        self.assertEqual(schedule.percent, 1)

    def test_frozen_entries_never_move_even_with_existing_adam_momentum(self):
        black, white = initial_boards(2)
        tokens = row_tokens(black, white)
        masks = bits_to_mask(legal_moves(black, white))
        records = [(tokens, masks, np.array([19, 19]), np.array([0, 1]))]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        # Create genuine momentum before freezing; a gradient mask alone would
        # allow the previously trained frozen coordinates to continue drifting.
        update_policy(self.model, optimizer, records, np.array([16., 16.], dtype=np.float32),
                      np.array([0, 1]), 0, [0., 0.])
        schedule = ProgressiveUnfreezer(self.model, 114)
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        for _ in range(3):
            update_policy(self.model, optimizer, records, np.array([16., 16.], dtype=np.float32),
                          np.array([0, 1]), 0, [0., 0.], unfreezer=schedule)
        changed_active = 0
        for name, parameter in self.model.named_parameters():
            mask = schedule.masks[name]
            self.assertTrue(torch.equal(parameter.detach()[~mask], before[name][~mask]), name)
            changed_active += int((parameter.detach()[mask] != before[name][mask]).sum())
        self.assertGreater(changed_active, 0)

    def test_saved_masks_history_and_next_release_restore_exactly(self):
        schedule = ProgressiveUnfreezer(self.model, 114, min_hold_updates=0)
        for step, wins in enumerate([70, 70, 58, 58, 70, 70, 58, 58], 1):
            schedule.observe(step, wins, 128-wins)
        buffer = io.BytesIO()
        torch.save(schedule.state_dict(), buffer)
        buffer.seek(0)
        restored = ProgressiveUnfreezer(copy.deepcopy(self.model), 114, min_hold_updates=0)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        for name in schedule.masks:
            self.assertTrue(torch.equal(schedule.masks[name], restored.masks[name]))
        for step in range(9, 16):
            self.assertEqual(schedule.observe(step, 58, 70), restored.observe(step, 58, 70))
        self.assertEqual(restored.percent, 2)
        self.assertEqual(schedule.summary(), restored.summary())

    def test_minimum_50_updates_at_start_and_after_every_release(self):
        schedule = ProgressiveUnfreezer(self.model, 114)
        for step in range(1, 101):
            released = schedule.observe(step, 70 if step % 2 else 58, 58 if step % 2 else 70)
            self.assertEqual(released, step in (50, 100))
        self.assertEqual([e['step'] for e in schedule.events], [50, 100])
        self.assertEqual(schedule.percent, 3)
        self.assertEqual(schedule.summary()['hold_remaining'], 50)

    def test_hold_end_uses_current_window_not_old_switches(self):
        schedule = ProgressiveUnfreezer(self.model, 114)
        for step in range(1, 16):
            schedule.observe(step, 70 if step % 2 else 58, 58 if step % 2 else 70)
        for step in range(16, 51):
            self.assertFalse(schedule.observe(step, 70, 58))
        self.assertEqual(schedule.percent, 1)
        self.assertEqual(schedule.latest_reversals, 0)
        self.assertFalse(schedule.observe(51, 58, 70))
        self.assertFalse(schedule.observe(52, 70, 58))
        self.assertTrue(schedule.observe(53, 58, 70))

    def test_50_update_hold_survives_resume_and_legacy_migration(self):
        schedule = ProgressiveUnfreezer(self.model, 114)
        for step in range(1, 31):
            schedule.observe(step, 70 if step % 2 else 58, 58 if step % 2 else 70)
        state = copy.deepcopy(schedule.state_dict())
        restored = ProgressiveUnfreezer(self.model, 114)
        restored.load_state_dict(state)
        self.assertEqual(restored.summary()['hold_remaining'], 20)
        for step in range(31, 51):
            self.assertEqual(restored.observe(step, 70 if step % 2 else 58, 58 if step % 2 else 70), step == 50)
        # Old checkpoints predate the waiting period; preserve masks and count
        # elapsed updates since the historical release instead of resetting it.
        legacy = ProgressiveUnfreezer(self.model, 114, min_hold_updates=0)
        for step in range(1, 21):
            legacy.observe(step, 70 if step % 2 else 58, 58 if step % 2 else 70)
        legacy_state = copy.deepcopy(legacy.state_dict())
        for key in ('min_hold_updates', 'stage_start_step', 'latest_step'):
            del legacy_state[key]
        migrated = ProgressiveUnfreezer(self.model, 114)
        migrated.load_state_dict(legacy_state)
        self.assertEqual(migrated.percent, legacy.percent)
        self.assertEqual(migrated.summary()['stage_start_step'], 15)
        self.assertEqual(migrated.summary()['hold_remaining'], 45)


class MarginRewardTest(unittest.TestCase):
    def test_actual_margin_not_win_indicator_and_opposite_rewards(self):
        black = np.array([15, 3, 1, 7], dtype=np.uint64)
        white = np.array([16, 12, 6, 8], dtype=np.uint64)
        ids = np.array([0, 0, 0, 1])
        a = terminal_rewards(black, white, ids, 0, mode='margin')
        b = terminal_rewards(black, white, ids, 1, mode='margin')
        np.testing.assert_array_equal(a, [3, 0, -1, -2])
        np.testing.assert_array_equal(a, -b)

    def test_match_counts_and_evaluation_are_not_sums_of_margin_rewards(self):
        match = play_batch(['random', 'random'], 16, np.random.default_rng(21), 'cpu', reward_mode='margin')
        self.assertEqual(match['a_wins'] + match['b_wins'] + match['draws'], 16)
        self.assertEqual(match['a_wins'], int((match['rewards'][0] > 0).sum()))
        np.testing.assert_array_equal(match['rewards'][0], -match['rewards'][1])
        scores = summarize(match)
        self.assertEqual(scores['win_rate'], match['a_wins'] / 16)
        self.assertEqual(sum(v['wins'] for v in scores['by_color'].values()), match['a_wins'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
