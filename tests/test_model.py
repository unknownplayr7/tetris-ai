"""Tests for next-piece model inputs and backward-compatible checkpoints."""
import unittest

import torch

from model import (
    LEGACY_OBS_SIZE,
    OBS_SIZE,
    TetrisActorCritic,
    upgrade_legacy_checkpoint,
)


class CheckpointMigrationTests(unittest.TestCase):
    def test_legacy_upgrade_preserves_policy_and_value_outputs(self):
        torch.manual_seed(123)
        legacy = TetrisActorCritic(observation_size=LEGACY_OBS_SIZE)
        legacy.eval()
        legacy_state = {key: value.clone() for key, value in legacy.state_dict().items()}
        checkpoint = {"model_state": legacy_state, "update": 12}

        upgraded = upgrade_legacy_checkpoint(checkpoint)
        current = TetrisActorCritic()
        current.load_state_dict(upgraded["model_state"])
        current.eval()

        old_observation = torch.randn(3, LEGACY_OBS_SIZE)
        new_observation = torch.cat(
            [old_observation, torch.randn(3, OBS_SIZE - LEGACY_OBS_SIZE)], dim=1
        )
        with torch.no_grad():
            old_logits, old_values = legacy(old_observation)
            new_logits, new_values = current(new_observation)

        torch.testing.assert_close(old_logits, new_logits, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(old_values, new_values, rtol=1e-6, atol=1e-6)
        self.assertEqual(upgraded["observation_schema_version"], 2)
        self.assertEqual(checkpoint["model_state"]["backbone.0.weight"].shape[1], LEGACY_OBS_SIZE)

    def test_legacy_optimizer_moments_are_padded(self):
        torch.manual_seed(7)
        legacy = TetrisActorCritic(observation_size=LEGACY_OBS_SIZE)
        optimizer = torch.optim.Adam(legacy.parameters(), lr=1e-3)
        inputs = torch.randn(2, LEGACY_OBS_SIZE)
        logits, values = legacy(inputs)
        (logits.square().mean() + values.square().mean()).backward()
        optimizer.step()

        old_state = optimizer.state_dict()
        old_first_weight = old_state["state"][0]["exp_avg"].clone()
        checkpoint = {
            "model_state": legacy.state_dict(),
            "optimizer_state": old_state,
        }
        upgraded = upgrade_legacy_checkpoint(checkpoint)
        migrated = upgraded["optimizer_state"]["state"][0]["exp_avg"]

        self.assertEqual(tuple(migrated.shape), (512, OBS_SIZE))
        self.assertTrue(torch.equal(migrated[:, :LEGACY_OBS_SIZE], old_first_weight))
        self.assertEqual(float(migrated[:, LEGACY_OBS_SIZE:].abs().sum()), 0.0)

    def test_unknown_observation_size_fails_clearly(self):
        checkpoint = {
            "model_state": {
                "backbone.0.weight": torch.zeros(512, LEGACY_OBS_SIZE - 1)
            }
        }
        with self.assertRaisesRegex(ValueError, "Unsupported checkpoint observation size"):
            upgrade_legacy_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
