"""PyTorch policy/value network for Tetris placement actions.

Observation schema v2 appends a one-hot encoding of the next piece. Legacy
checkpoints are upgraded by zero-padding their first-layer inputs, preserving
their original policy until the new preview features are learned.
"""
import copy

import torch
from torch import nn

from tetris_env import HEIGHT, WIDTH, PIECE_NAMES, MAX_ACTIONS

LEGACY_OBS_SIZE = HEIGHT * WIDTH + len(PIECE_NAMES)
OBS_SIZE = HEIGHT * WIDTH + 2 * len(PIECE_NAMES)
OBSERVATION_SCHEMA_VERSION = 2


class TetrisActorCritic(nn.Module):
    def __init__(self, hidden=512, observation_size=OBS_SIZE):
        super().__init__()
        self.observation_size = observation_size
        self.backbone = nn.Sequential(
            nn.Linear(observation_size, hidden),
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


def upgrade_legacy_checkpoint(checkpoint):
    """Upgrade a v1 checkpoint to the v2 next-piece observation schema.

    Old observations were [board, current-piece one-hot]. New observations are
    [board, current-piece one-hot, next-piece one-hot]. Zero-padding the new
    input columns preserves old model outputs exactly. Matching Adam moments are
    padded too, allowing training to resume without discarding optimizer state.
    """
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("Checkpoint must be a training checkpoint with 'model_state'.")

    upgraded = copy.deepcopy(checkpoint)
    model_state = upgraded["model_state"]
    first_weight_key = "backbone.0.weight"
    if first_weight_key not in model_state:
        raise ValueError("Checkpoint is missing the actor-critic input layer.")

    first_weight = model_state[first_weight_key]
    if first_weight.ndim != 2:
        raise ValueError("Checkpoint input-layer weights have an invalid shape.")
    input_size = int(first_weight.shape[1])

    if input_size == OBS_SIZE:
        upgraded.setdefault("observation_schema_version", OBSERVATION_SCHEMA_VERSION)
        return upgraded
    if input_size != LEGACY_OBS_SIZE:
        raise ValueError(
            f"Unsupported checkpoint observation size {input_size}; expected "
            f"{LEGACY_OBS_SIZE} (legacy) or {OBS_SIZE} (current)."
        )

    expanded = first_weight.new_zeros((first_weight.shape[0], OBS_SIZE))
    expanded[:, :LEGACY_OBS_SIZE] = first_weight
    model_state[first_weight_key] = expanded

    optimizer_state = upgraded.get("optimizer_state")
    if isinstance(optimizer_state, dict):
        for parameter_state in optimizer_state.get("state", {}).values():
            for key, value in list(parameter_state.items()):
                if torch.is_tensor(value) and tuple(value.shape) == tuple(first_weight.shape):
                    padded = value.new_zeros((value.shape[0], OBS_SIZE))
                    padded[:, :LEGACY_OBS_SIZE] = value
                    parameter_state[key] = padded

    upgraded["observation_schema_version"] = OBSERVATION_SCHEMA_VERSION
    upgraded["checkpoint_migration"] = "v1_to_v2_next_piece_preview"
    return upgraded
