"""Independent rules oracle and learning regression checks (unittest only)."""
import unittest
import numpy as np
import torch

from model import ComplexTaylor, ModelConfig, OthelloGPT, masked_log_probs
from othello import (U64, apply_moves, bits_to_mask, initial_boards, legal_moves,
                     row_tokens, terminal_rewards)
from train import play_batch, update_policy


def oracle_moves(own, opp):
    board = [1 if own >> i & 1 else -1 if opp >> i & 1 else 0 for i in range(64)]
    result = {}
    for i, cell in enumerate(board):
        if cell:
            continue
        captured = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                r, c = i // 8 + dr, i % 8 + dc
                ray = []
                while 0 <= r < 8 and 0 <= c < 8 and board[8*r+c] == -1:
                    ray.append(8*r+c)
                    r, c = r+dr, c+dc
                if ray and 0 <= r < 8 and 0 <= c < 8 and board[8*r+c] == 1:
                    captured += ray
        if captured:
            flips = sum(1 << j for j in set(captured))
            result[i] = (own | (1 << i) | flips, opp & ~flips)
    return result


class RulesTest(unittest.TestCase):
    def test_rules_against_independent_coordinate_oracle(self):
        rng = np.random.default_rng(1123)
        saw_pass, saw_terminal_empty = False, False
        for _ in range(24):
            black, white = initial_boards(1)
            own, opp = int(black[0]), int(white[0])
            for turn in range(122):
                reference = oracle_moves(own, opp)
                legal = int(legal_moves(np.array([own], dtype=np.uint64), np.array([opp], dtype=np.uint64))[0])
                self.assertEqual(legal, sum(1 << i for i in reference))
                if not reference:
                    if not oracle_moves(opp, own):
                        saw_terminal_empty |= (own | opp).bit_count() < 64
                        break
                    saw_pass = True
                    own, opp = opp, own
                    continue
                action = int(rng.choice(list(reference)))
                after = apply_moves(np.array([own], dtype=np.uint64), np.array([opp], dtype=np.uint64),
                                    np.array([1 << action], dtype=np.uint64))
                self.assertEqual((int(after[0][0]), int(after[1][0])), reference[action])
                self.assertEqual(int(after[0][0] & after[1][0]), 0)
                own, opp = int(after[1][0]), int(after[0][0])
            else:
                self.fail('Game did not terminate')
        self.assertTrue(saw_pass)

    def test_terminal_reward_is_win_only_and_color_correct(self):
        black = np.array([15, 3, 1, 7], dtype=np.uint64)
        white = np.array([16, 12, 6, 8], dtype=np.uint64)
        agents = np.array([0, 0, 0, 1])
        np.testing.assert_array_equal(terminal_rewards(black, white, agents, 0), [1, 0, 0, 0])
        np.testing.assert_array_equal(terminal_rewards(black, white, agents, 1), [0, 0, 1, 1])
        # An early game end with empty cells is still scored by actual stones.
        self.assertEqual(terminal_rewards(np.array([7], dtype=np.uint64),
                                         np.array([0], dtype=np.uint64), np.array([0]), 0)[0], 1)

    def test_row_token_encoding_and_initial_legal_moves(self):
        black, white = initial_boards(1)
        legal = np.flatnonzero(bits_to_mask(legal_moves(black, white))[0])
        np.testing.assert_array_equal(legal, [19, 26, 37, 44])
        tokens = row_tokens(black, white)
        decoded = (tokens[..., None] // (3 ** np.arange(8))) % 3
        expected = bits_to_mask(black).astype(int) + 2 * bits_to_mask(white)
        np.testing.assert_array_equal(decoded.reshape(1, 64), expected)


class LearningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_mask_has_zero_probability_for_illegal_actions_and_no_pass(self):
        logits = torch.tensor([[100., 1., -2.]])
        legal = torch.tensor([[False, True, True]])
        probs = masked_log_probs(logits, legal).exp()
        self.assertEqual(probs[0, 0], 0)
        self.assertAlmostEqual(float(probs.sum()), 1., places=6)
        with self.assertRaises(ValueError):
            masked_log_probs(logits, torch.zeros_like(legal))

    def test_frozen_trajectory_recomputation_updates_both_complete_models(self):
        for family in ('transformer', 'taylor'):
            with self.subTest(family=family):
                torch.manual_seed(724)
                make = (lambda: OthelloGPT(ModelConfig(width=16, layers=1, heads=2))) if family == 'transformer' else (lambda: ComplexTaylor(2))
                models = [make(), make()]
                match = play_batch(models, 8, np.random.default_rng(51), 'cpu', collect=True)
                self.assertEqual(match['a_wins'] + match['b_wins'] + match['draws'], 8)
                for agent, model in enumerate(models):
                    old = {n: p.detach().clone() for n, p in model.named_parameters()}
                    metrics = update_policy(model, torch.optim.Adam(model.parameters(), lr=1e-4),
                                            match['records'][agent], match['rewards'][agent],
                                            match['black_agents'], agent, [.5, .5], chunk_size=64)
                    self.assertGreater(metrics['parameter_delta_l2'], 0)
                    self.assertGreater(metrics['gradient_norm'], 0)
                    for name, p in model.named_parameters():
                        self.assertIsNotNone(p.grad, name)
                        self.assertTrue(torch.isfinite(p.grad).all(), name)
                        self.assertTrue(torch.isfinite(p).all(), name)
                    # Both input embedding/function maps and readout must learn.
                    keys = ('embedding.weight', 'head.weight') if family == 'transformer' else ('a0', 'a1', 'a2', 'observation', 'next_a1')
                    for key in keys:
                        self.assertFalse(torch.equal(old[key], dict(model.named_parameters())[key]), key)

    def test_reward_increases_selected_action_probability(self):
        model = OthelloGPT(ModelConfig(width=16, layers=1, heads=2))
        black, white = initial_boards(2)
        tokens = row_tokens(black, white)
        mask = bits_to_mask(legal_moves(black, white))
        actions = np.array([19, 19])
        records = [(tokens, mask, actions, np.array([0, 1]))]
        tensor_tokens = torch.tensor(tokens)
        tensor_mask = torch.tensor(mask)
        before = masked_log_probs(model(tensor_tokens), tensor_mask)[:, 19].mean().item()
        update_policy(model, torch.optim.Adam(model.parameters(), lr=1e-4), records,
                      np.ones(2, dtype=np.float32), np.array([0, 1]), 0, [0., 0.])
        after = masked_log_probs(model(tensor_tokens), tensor_mask)[:, 19].mean().item()
        self.assertGreater(after, before)

    def test_taylor_measurement_is_normalized_and_autograd_reaches_frame(self):
        torch.manual_seed(31)
        model = ComplexTaylor(4)
        scores = model(torch.randint(0, 6561, (8, 8)))
        torch.testing.assert_close(scores.exp().sum(-1), torch.ones(8), rtol=1e-5, atol=1e-6)
        (-scores[:, 0].sum()).backward()
        self.assertGreater(float(model.observation.grad.norm()), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
