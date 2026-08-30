"""PyTorch policy/value network for Tetris placement actions."""

import torch
from torch import nn

from tetris_env import HEIGHT, WIDTH, PIECE_NAMES, MAX_ACTIONS

OBS_SIZE = HEIGHT * WIDTH + len(PIECE_NAMES)


class TetrisActorCritic(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(OBS_SIZE, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, MAX_ACTIONS)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x):
        z = self.backbone(x)
        return self.policy(z), self.value(z).squeeze(-1)

    def masked_logits(self, observations, action_masks):
        logits, value = self(observations)
        logits = logits.masked_fill(~action_masks.bool(), -1e9)
        return logits, value
