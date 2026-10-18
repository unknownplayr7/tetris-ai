"""Regression tests for the board rules, heuristic agent, and RL environment."""
import unittest

from ai import find_best_move
from board import HEIGHT, WIDTH, clear_lines, new_board, simulate_drop
from pieces import SHAPES
from tetris_env import TetrisEnv


class BoardTests(unittest.TestCase):
    def test_clears_a_full_bottom_row(self):
        board = new_board()
        board[-1] = [1] * WIDTH

        result, cleared = clear_lines(board)

        self.assertEqual(cleared, 1)
        self.assertEqual(result[0], [0] * WIDTH)
        self.assertEqual(result[-1], [0] * WIDTH)

    def test_heuristic_returns_a_legal_placement_with_lookahead(self):
        board = new_board()
        move = find_best_move(board, SHAPES["T"], next_shape=SHAPES["I"])

        self.assertIsNotNone(move)
        _, rotated, x, result, lines = move
        expected, expected_lines = simulate_drop(board, rotated, x)
        self.assertEqual(result, expected)
        self.assertEqual(lines, expected_lines)
        self.assertEqual(len(result), HEIGHT)
        self.assertTrue(all(len(row) == WIDTH for row in result))


class EnvironmentTests(unittest.TestCase):
    def test_action_mask_has_legal_moves_and_step_accepts_one(self):
        env = TetrisEnv(seed=7)
        observation = env.reset()
        mask = env.valid_action_mask()

        self.assertEqual(observation.shape, (HEIGHT * WIDTH + len(SHAPES),))
        self.assertTrue(mask.any())

        action = int(mask.nonzero()[0][0])
        next_observation, reward, done, info = env.step(action)

        self.assertEqual(next_observation.shape, observation.shape)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertFalse(info["invalid"])


if __name__ == "__main__":
    unittest.main()
