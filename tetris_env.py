"""Fast placement-based Tetris environment for reinforcement learning."""

import random
import numpy as np

from board import WIDTH, HEIGHT, new_board, simulate_drop
from pieces import SHAPES, rotations

PIECE_NAMES = tuple(SHAPES.keys())
ACTIONS_PER_ROTATION = WIDTH
MAX_ACTIONS = 4 * WIDTH


def board_features(board):
    return np.asarray(board, dtype=np.float32)


class TetrisEnv:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.board = None
        self.piece_name = None
        self.shape = None
        self.steps = 0

    def reset(self):
        self.board = new_board()
        self.steps = 0
        self._next_piece()
        return self.observation()

    def _next_piece(self):
        self.piece_name = self.rng.choice(PIECE_NAMES)
        self.shape = [row[:] for row in SHAPES[self.piece_name]]

    def observation(self):
        board = board_features(self.board).reshape(-1)
        piece = np.zeros(len(PIECE_NAMES), dtype=np.float32)
        piece[PIECE_NAMES.index(self.piece_name)] = 1.0
        return np.concatenate((board, piece))

    def valid_action_mask(self):
        mask = np.zeros(MAX_ACTIONS, dtype=np.bool_)
        rots = rotations(self.shape)
        for r, rotated in enumerate(rots):
            shape_width = len(rotated[0])
            for x in range(WIDTH - shape_width + 1):
                _, lines = simulate_drop(self.board, rotated, x)
                # A legal spawn may clear zero lines, so check the result directly.
                result, _ = simulate_drop(self.board, rotated, x)
                if result is not None:
                    mask[r * WIDTH + x] = True
        return mask

    def step(self, action):
        self.steps += 1
        rotation_index = int(action) // WIDTH
        x = int(action) % WIDTH
        rots = rotations(self.shape)

        if rotation_index >= len(rots):
            return self.observation(), -5.0, True, {"lines": 0, "invalid": True}

        rotated = rots[rotation_index]
        if x > WIDTH - len(rotated[0]):
            return self.observation(), -5.0, True, {"lines": 0, "invalid": True}

        result, lines = simulate_drop(self.board, rotated, x)
        if result is None:
            return self.observation(), -5.0, True, {"lines": 0, "invalid": True}

        self.board = result
        heights = self._heights()
        holes = self._holes()
        max_height = max(heights)

        # Dense reward makes early learning much less miserable.
        reward = 0.05 + lines * lines * 1.0 - holes * 0.08 - max_height * 0.01
        self._next_piece()

        # If the next piece has no legal placement, the episode ends.
        done = not self.valid_action_mask().any()
        if done:
            reward -= 2.0

        return self.observation(), float(reward), done, {"lines": lines, "invalid": False}

    def _heights(self):
        heights = []
        for x in range(WIDTH):
            height = 0
            for y in range(HEIGHT):
                if self.board[y][x]:
                    height = HEIGHT - y
                    break
            heights.append(height)
        return heights

    def _holes(self):
        holes = 0
        for x in range(WIDTH):
            seen = False
            for y in range(HEIGHT):
                if self.board[y][x]:
                    seen = True
                elif seen:
                    holes += 1
        return holes
