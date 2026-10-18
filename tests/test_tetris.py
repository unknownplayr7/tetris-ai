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
    def test_seven_bag_contains_each_piece_once_and_is_seeded(self):
        def sequence(seed):
            env = TetrisEnv(seed=seed)
            env.reset()
            names = []
            for _ in range(14):
                names.append(env.piece_name)
                env._next_piece()
            return names

        first = sequence(41)
        self.assertEqual(first, sequence(41))
        self.assertEqual(set(first[:7]), set(SHAPES))
        self.assertEqual(set(first[7:14]), set(SHAPES))

    def test_peeking_next_piece_does_not_consume_it(self):
        def sequence(peek):
            env = TetrisEnv(seed=51)
            env.reset()
            names = []
            for _ in range(21):
                if peek:
                    env.peek_next_shape()
                names.append(env.piece_name)
                env._next_piece()
            return names

        self.assertEqual(sequence(False), sequence(True))

    def test_negative_action_is_rejected(self):
        env = TetrisEnv(seed=4)
        env.reset()
        _observation, reward, done, info = env.step(-10)
        self.assertTrue(done)
        self.assertEqual(reward, -5.0)
        self.assertTrue(info["invalid"])

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
